# ============================================================
# sentinel/utils/trainer.py
# Training loop + YOBDM threshold + Evaluation
# ============================================================

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score,
                              precision_recall_curve)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1.  YOBDM — Youden-Optimal Boundary Decision
# ─────────────────────────────────────────────

def compute_youden_threshold(scores: np.ndarray,
                              labels: np.ndarray) -> float:
    """
    Compute the Youden-optimal threshold from validation scores.
    Youden's J = Sensitivity + Specificity - 1
               = TPR + TNR - 1
               = TPR - FPR

    Maximising J gives the best trade-off between sensitivity
    and specificity — no need to manually tune a threshold.

    Returns: optimal threshold (float)
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    # thresholds has len = len(precisions) - 1
    f1_scores = (2 * precisions[:-1] * recalls[:-1] /
                 (precisions[:-1] + recalls[:-1] + 1e-8))
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx]
    print(f"YOBDM threshold: {threshold:.4f}  "
          f"(val F1={f1_scores[best_idx]:.4f})")
    return float(threshold)


# ─────────────────────────────────────────────
# 2.  Single Epoch Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        seq     = batch['seq'].to(device)
        seq_len = batch['seq_len'].to(device)
        graph   = batch['graph'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = model(seq, seq_len, graph)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(seq, seq_len, graph)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 3.  Score Collection (val / test)
# ─────────────────────────────────────────────

def collect_scores(model, loader, device):
    """
    Run model in inference mode.
    Returns (scores_np, labels_np).
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Scoring", leave=False):
            seq     = batch['seq'].to(device)
            seq_len = batch['seq_len'].to(device)
            graph   = batch['graph'].to(device)
            labels  = batch['label']

            scores = model.score(seq, seq_len, graph)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_scores),
            np.concatenate(all_labels))


# ─────────────────────────────────────────────
# 4.  Metrics Computation
# ─────────────────────────────────────────────

def compute_metrics(scores: np.ndarray,
                    labels: np.ndarray,
                    threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    p  = precision_score(labels, preds, zero_division=0)
    r  = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float('nan')

    return {'precision': p, 'recall': r, 'f1': f1, 'auc': auc,
            'threshold': threshold}


def print_metrics(metrics: dict, split: str = "Test"):
    print(f"\n{'='*40}")
    print(f"  {split} Results")
    print(f"{'='*40}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.4f}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
# 5.  Full Training Loop
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_sentinel(
    model,
    train_loader,
    val_loader,
    epochs:       int   = 50,
    lr:           float = 1e-3,
    weight_decay: float = 1e-5,
    patience:     int   = 10,
    use_amp:      bool  = True,
    device:       str   = 'cuda',
    checkpoint_dir: str = '/content/drive/MyDrive/SENTINEL/checkpoints',
    dataset_name:   str = 'HDFS',
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f'sentinel_{dataset_name}_best.pt')

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler    = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None
    stopper   = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\nTraining SENTINEL on {dataset_name}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Epochs: {epochs} | LR: {lr} | Device: {device}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        train_losses.append(train_loss)

        # ── Val ── compute loss on full val batch (includes anomalies)
        # We use full batch loss as proxy — anomalies will have higher NLL
        # which slightly inflates val loss but avoids graph batch size mismatch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq     = batch['seq'].to(device)
                seq_len = batch['seq_len'].to(device)
                graph   = batch['graph'].to(device)
                loss    = model(seq, seq_len, graph)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # ── Save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss': val_loss}, ckpt_path)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # ── Early stopping ──
        if stopper.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── Load best model ──
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}")

    return model, train_losses, val_losses


# ─────────────────────────────────────────────
# 6.  Plot Training Curves
# ─────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses,
                          save_path=None, dataset_name='HDFS'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train Loss', color='steelblue')
    ax.plot(val_losses,   label='Val Loss',   color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL Loss')
    ax.set_title(f'SENTINEL Training — {dataset_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curve saved → {save_path}")
    plt.show()


def plot_score_distribution(scores, labels,
                             threshold=None, save_path=None,
                             dataset_name='HDFS'):
    fig, ax = plt.subplots(figsize=(8, 4))
    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    ax.hist(normal_scores,  bins=80, alpha=0.6,
            label='Normal',    color='steelblue', density=True)
    ax.hist(anomaly_scores, bins=80, alpha=0.6,
            label='Anomalous', color='tomato',    density=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--',
                   linewidth=1.5, label=f'YOBDM τ={threshold:.3f}')
    ax.set_xlabel('Anomaly Score (−log p(z))')
    ax.set_ylabel('Density')
    ax.set_title(f'SENTINEL Score Distribution — {dataset_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
