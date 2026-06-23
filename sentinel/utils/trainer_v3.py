# ============================================================
# sentinel/utils/trainer_v3.py
# Training loop for SENTINELv3
# ============================================================

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score,
                              precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1.  YOBDM Threshold
# ─────────────────────────────────────────────

def compute_youden_threshold(scores, labels):
    precisions, recalls, thresholds = precision_recall_curve(
        labels, scores)
    f1s     = (2 * precisions[:-1] * recalls[:-1]
               / (precisions[:-1] + recalls[:-1] + 1e-8))
    best    = np.argmax(f1s)
    tau     = float(thresholds[best])
    print(f"YOBDM threshold: {tau:.4f}  (val F1={f1s[best]:.4f})")
    return tau


# ─────────────────────────────────────────────
# 2.  Metrics
# ─────────────────────────────────────────────

def compute_metrics(scores, labels, threshold):
    preds = (scores >= threshold).astype(int)
    p  = precision_score(labels, preds, zero_division=0)
    r  = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float('nan')
    return {'precision': p, 'recall': r, 'f1': f1,
            'auc': auc, 'threshold': threshold}


def print_metrics(metrics, split='Test'):
    print(f"\n{'='*45}")
    print(f"  {split} Results")
    print(f"{'='*45}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.4f}")
    print(f"{'='*45}")


# ─────────────────────────────────────────────
# 3.  Score Collection
# ─────────────────────────────────────────────

def collect_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Scoring", leave=False):
            seq     = batch['seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            graph   = batch['graph'].to(device)
            labels  = batch['label']
            scores  = model.score(seq, seq_len, graph)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ─────────────────────────────────────────────
# 4.  Training Epoch
# ─────────────────────────────────────────────

def train_epoch_v3(model, loader, optimizer, device, scaler=None):
    model.train()
    totals = {'total': 0, 'flow': 0, 'contrast': 0, 'recon': 0}

    for batch in tqdm(loader, desc="  Train", leave=False):
        # Move to device
        batch_dev = {
            'seq_norm':   batch['seq_norm'].to(device),
            'seq_aug':    batch['seq_aug'].to(device),
            'len_norm':   batch['len_norm'].to(device),
            'len_aug':    batch['len_aug'].to(device),
            'graph_norm': batch['graph_norm'].to(device),
            'graph_aug':  batch['graph_aug'].to(device),
        }

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                total, fl, cl, rl = model(batch_dev)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total, fl, cl, rl = model(batch_dev)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        totals['total']    += total.item()
        totals['flow']     += fl.item()
        totals['contrast'] += cl.item()
        totals['recon']    += rl.item()

    n = len(loader)
    return {k: v/n for k, v in totals.items()}


# ─────────────────────────────────────────────
# 5.  Validation Loss
# ─────────────────────────────────────────────

def val_epoch_v3(model, loader, device):
    """
    Val loss computed on normal sessions only
    (identified by label=0 in val loader).
    We approximate by running full val batch through
    forward pass and using total loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            # For val we only have seq/graph (eval dataset)
            # Use score variance as proxy for separation
            seq     = batch['seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            graph   = batch['graph'].to(device)
            scores  = model.score(seq, seq_len, graph)
            # Higher score variance = better separation
            val_loss += -scores.std().item()
    return val_loss / len(loader)


# ─────────────────────────────────────────────
# 6.  Full Training Loop
# ─────────────────────────────────────────────

def train_sentinel_v3(
    model,
    train_loader,
    val_loader,
    epochs:        int   = 50,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-5,
    patience:      int   = 15,
    use_amp:       bool  = True,
    device:        str   = 'cuda',
    checkpoint_dir: str  = '/content/drive/MyDrive/SENTINEL/checkpoints',
    dataset_name:  str   = 'HDFS',
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir, f'sentinel_v3_{dataset_name}_best.pt')

    model     = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.01)
    scaler    = torch.amp.GradScaler('cuda') \
                if use_amp and device == 'cuda' else None

    best_val   = float('inf')
    no_improve = 0
    train_history = []

    print(f"\nSENTINEL v3 Training — {dataset_name}")
    print(f"Parameters    : {model.count_parameters():,}")
    print(f"Epochs        : {epochs} | LR: {lr} | Device: {device}")
    print(f"Margin        : {model.margin} | "
          f"λ_contrast: {model.lambda_contrast} | "
          f"λ_recon: {model.lambda_recon}\n")

    for epoch in range(1, epochs + 1):
        # Train
        losses = train_epoch_v3(
            model, train_loader, optimizer, device, scaler)
        train_history.append(losses)

        # Val (score variance proxy)
        val_loss = val_epoch_v3(model, val_loader, device)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Total: {losses['total']:8.4f} | "
              f"Flow: {losses['flow']:8.4f} | "
              f"Contrast: {losses['contrast']:.4f} | "
              f"Recon: {losses['recon']:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model (lowest val loss = highest score variance)
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    val_loss,
            }, ckpt_path)
            print(f"  ✓ Best saved (epoch={epoch}, "
                  f"val={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    return model, train_history


# ─────────────────────────────────────────────
# 7.  Plots
# ─────────────────────────────────────────────

def plot_training_v3(history, save_path=None, dataset_name='HDFS'):
    epochs     = range(1, len(history) + 1)
    flow_loss  = [h['flow']     for h in history]
    cont_loss  = [h['contrast'] for h in history]
    recon_loss = [h['recon']    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, flow_loss, color='steelblue')
    axes[0].set_title('Flow Loss (NLL)')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, cont_loss, color='tomato')
    axes[1].set_title('Contrastive Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, recon_loss, color='seagreen')
    axes[2].set_title('Reconstruction Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'SENTINEL v3 Training — {dataset_name}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved → {save_path}")
    plt.close()


def plot_score_dist(scores, labels, threshold=None,
                    save_path=None, dataset_name='HDFS'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels==0], bins=80, alpha=0.6,
            label='Normal', color='steelblue', density=True)
    ax.hist(scores[labels==1], bins=80, alpha=0.6,
            label='Anomalous', color='tomato', density=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--',
                   linewidth=1.5,
                   label=f'YOBDM τ={threshold:.3f}')
    ax.set_xlabel('Anomaly Score (−log p(z))')
    ax.set_ylabel('Density')
    ax.set_title(f'SENTINEL v3 Score Distribution — {dataset_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Score dist saved → {save_path}")
    plt.close()
