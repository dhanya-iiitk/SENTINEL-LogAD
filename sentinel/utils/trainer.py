# ============================================================
# sentinel/utils/trainer.py
# Training loop + YOBDM threshold + Evaluation
# For SENTINEL: Semi-supervised Log Anomaly Detection
# ============================================================

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve)


# ─────────────────────────────────────────────
# 1. YOBDM — Youden-Optimal Boundary Decision
# ─────────────────────────────────────────────

def compute_youden_threshold(
        scores: np.ndarray,
        labels: np.ndarray) -> float:
    """
    Compute the Youden-optimal threshold.
    Youden's J = TPR - FPR
    Maximising J gives best trade-off between
    sensitivity and specificity without manual
    parameter selection.
    Returns: optimal threshold (float)
    """
    fpr, tpr, thresholds = roc_curve(
        labels, scores)
    j        = tpr - fpr
    best_idx = np.argmax(j)
    threshold = float(thresholds[best_idx])
    print(f"  YOBDM threshold : {threshold:.4f}")
    print(f"  Youden J        : {j[best_idx]:.4f}")
    print(f"  TPR             : {tpr[best_idx]:.4f}")
    print(f"  FPR             : {fpr[best_idx]:.4f}")
    return threshold


# ─────────────────────────────────────────────
# 2. Collect Anomaly Scores
# ─────────────────────────────────────────────

def collect_scores(model, loader, device):
    """
    Collect anomaly scores for all sessions.
    Returns: (scores, labels) as numpy arrays
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            scores = model.anomaly_score(
                batch['seq'].to(device),
                batch['seq_len'].to(device))
            all_scores.append(
                scores.cpu().numpy())
            all_labels.append(
                batch['label'].numpy())
    return (np.concatenate(all_scores),
            np.concatenate(all_labels))


# ─────────────────────────────────────────────
# 3. Single Epoch Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer,
                device, scaler=None):
    """
    Training loop for sequential model.
    No graph/GNN dependency.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader,
                       desc="  Train",
                       leave=False):
        seq     = batch['seq'].to(device)
        seq_len = batch['seq_len'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(seq, seq_len)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(seq, seq_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────

def evaluate(model, loader, threshold,
             device, prefix="Val"):
    """
    Evaluate model using precomputed threshold.
    Returns: dict with P, R, F1, AUC
    """
    scores, labels = collect_scores(
        model, loader, device)
    preds = (scores >= threshold).astype(int)

    p   = precision_score(
        labels, preds, zero_division=0)
    r   = recall_score(
        labels, preds, zero_division=0)
    f1  = f1_score(
        labels, preds, zero_division=0)
    auc = roc_auc_score(labels, scores)

    print(f"  {prefix}: P={p:.4f} R={r:.4f} "
          f"F1={f1:.4f} AUC={auc:.4f}")
    return {'precision': p, 'recall': r,
            'f1': f1, 'auc': auc}


# ─────────────────────────────────────────────
# 5. Full Training Loop
# ─────────────────────────────────────────────

def train_model(model, train_loader,
                val_loader, optimizer,
                scheduler, device,
                save_path, max_epochs=30,
                patience=15):
    """
    Full training loop with early stopping.
    Uses YOBDM for threshold calibration.
    Saves best model checkpoint.

    Args:
        model       : SENTINELSeqFeatureExtract
        train_loader: normal sessions only
        val_loader  : mixed normal + anomalous
        optimizer   : AdamW
        scheduler   : CosineAnnealingLR
        device      : cuda or cpu
        save_path   : checkpoint save path
        max_epochs  : maximum training epochs
        patience    : early stopping patience
    """
    best_f1    = 0.0
    best_epoch = 0
    no_improve = 0
    scaler     = torch.cuda.amp.GradScaler()

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        # Train one epoch
        train_loss = train_epoch(
            model, train_loader,
            optimizer, device, scaler)
        print(f"  Train loss: {train_loss:.4f}")

        # Collect validation scores
        val_scores, val_labels = collect_scores(
            model, val_loader, device)

        # Compute YOBDM threshold
        threshold = compute_youden_threshold(
            val_scores, val_labels)

        # Compute validation F1
        val_preds = (
            val_scores >= threshold).astype(int)
        val_f1 = f1_score(
            val_labels, val_preds,
            zero_division=0)
        print(f"  Val F1: {val_f1:.4f}")

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Early stopping + save best
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'threshold'  : threshold,
                'val_f1'     : val_f1,
            }, save_path)
            print(f"  ✓ Best saved "
                  f"(F1={val_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improve "
                  f"({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping "
                      f"at epoch {epoch}")
                break

    print(f"\nBest epoch : {best_epoch}")
    print(f"Best val F1: {best_f1:.4f}")
    return best_f1, best_epoch
