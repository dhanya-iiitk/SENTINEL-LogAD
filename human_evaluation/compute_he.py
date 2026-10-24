import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

cases = [
    'BGL – Anomaly 1 (Microloader Assertion)',
    'BGL – Anomaly 2 (TLB Error)',
    'BGL – Normal',
    'BGL – Borderline (TLB + Recovery)',
    'HDFS – Anomaly 1 (Block Write Failure)',
    'HDFS – Anomaly 2 (Socket Timeout + Exception)',
    'TB – Anomaly 1 (InfiniBand MAD + Fatal Error)',
    'TB – Anomaly 2 (MAD Failure Storm)',
]

# ── INPUT: Evaluator 1 ───────────────────────────────────────────────
# Each row: [Accuracy, Clarity, Relevance, Actionability]  (scale 1–5)
eval1 = [
    [4, 5, 5, 4],  # Case 1
    [5, 5, 5, 5],  # Case 2
    [5, 5, 5, 5],  # Case 3
    [4, 5, 4, 5],  # Case 4
    [4, 5, 4, 5],  # Case 5
    [4, 5, 5, 5],  # Case 6
    [5, 5, 4, 5],  # Case 7
    [5, 5, 4, 5],  # Case 8
]

# ── INPUT: Evaluator 2 ───────────────────────────────────────────────
eval2 = [
    [4, 5, 5, 4],  # Case 1
    [5, 5, 5, 5],  # Case 2
    [5, 5, 5, 5],  # Case 3
    [5, 5, 4, 5],  # Case 4
    [4, 5, 4, 5],  # Case 5
    [4, 5, 5, 5],  # Case 6
    [5, 5, 4, 5],  # Case 7
    [5, 5, 4, 5],  # Case 8
]

# ── Compute HE scores ────────────────────────────────────────────────
e1 = np.array(eval1, dtype=float)
e2 = np.array(eval2, dtype=float)

he_e1 = e1.mean(axis=1)   # mean of 4 criteria per case, Evaluator 1
he_e2 = e2.mean(axis=1)   # mean of 4 criteria per case, Evaluator 2
he    = (he_e1 + he_e2) / 2  # average across both evaluators

# ── Cohen's κ ────────────────────────────────────────────────────────
# Computed over all 8×4 = 32 individual ratings (per-criterion, per-case)
# weighted='quadratic' is standard for ordinal Likert scales
kappa = cohen_kappa_score(
    e1.flatten().astype(int),
    e2.flatten().astype(int),
    weights='quadratic'
)

# ── Results table ────────────────────────────────────────────────────
df = pd.DataFrame({
    'Case'      : cases,
    'HE (E1)'   : he_e1.round(2),
    'HE (E2)'   : he_e2.round(2),
    'HE (Mean)' : he.round(2),
})

print(df.to_string(index=False))
print(f"\nOverall HE Mean : {he.mean():.3f}")
print(f"Cohen's κ       : {kappa:.3f}  (quadratic-weighted, 32 ratings)")
