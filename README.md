# SENTINEL
## **S**emi-supervised Log An**o**maly Detec**t**ion via **I**ncremen**t**al Transfer L**e**arning with **N**ormalizing-flow-free Scoring

A semi-supervised log anomaly detection framework using a **two-stage transfer learning** approach:

- **Stage 1 — Universal Pretraining:** Causal Transformer encoder pretrained on combined normal log corpus via next-event prediction
- **Stage 2 — Feature Extraction:** Frozen pretrained encoder + lightweight dataset-specific task head, trained on normal sessions only
- **Anomaly Scoring:** Per-token cross-entropy over the task head (no normalizing flow, no labels required)
- **YOBDM:** Youden-optimal threshold selection on a small held-out normal validation set
- **IG + LLM:** Integrated Gradients attribution + Groq LLM for explainable cyber threat analysis

---

## Key Properties

| Property | Detail |
|---|---|
| **Supervision** | Semi-supervised — trains on normal sessions only |
| **Label requirement** | None during training; small normal validation set for threshold |
| **Architecture** | Causal Transformer (L=2, H=4, d=128, FFN=256) |
| **Transfer** | Universal pretrained encoder shared across datasets |
| **Task head** | 32K–1.7M params depending on dataset vocabulary |
| **Explainability** | Token-level IG attribution + Groq LLM narrative |

---

## Repository Structure

```
SENTINEL-LogAD/
├── configs/
│   └── config.py                     # All hyperparameters
├── sentinel/
│   ├── data/
│   │   ├── parser.py                 # Drain parsing + session construction
│   │   └── dataset.py                # PyTorch Dataset + DataLoader
│   ├── models/
│   │   ├── sentinel_seq.py           # SENTINELSeqBase + SENTINELSeqFeatureExtract
│   │   └── sentinel.py               # Legacy GNN+RealNVP model (archived)
│   ├── utils/
│   │   └── trainer.py                # Training loop + YOBDM + evaluation
│   └── explain/
│       └── llm_explain.py            # IG attribution + Groq LLM explanation
├── benchmarks/
│   └── edge_latency_benchmark.py     # Single-core CPU latency benchmark
├── notebooks/
│   ├── 00_setup.ipynb                # Environment + dependencies
│   ├── 01_parsing.ipynb              # Dataset parsing (HDFS, BGL, ThunderBird)
│   └── 02_training.ipynb             # Training + evaluation
├── human_evaluation/
│   ├── compute_he.py             # HE scoring script (Cohen's κ)
│   └── he_results.txt            # Evaluation output
└── README.md
```

---

## Model Architecture

```
Input log session (sequence of event IDs)
        │
        ▼
┌─────────────────────────────┐
│   Token Embedding + PE      │  d=128, max_len=512
└────────────┬────────────────┘
             │
        ┌────▼────┐
        │ Causal  │  L=2 Transformer Encoder layers
        │ Transf. │  H=4 heads, FFN=256, pre-norm
        └────┬────┘
             │  [FROZEN at Stage 2]
        ┌────▼────────────────┐
        │  Dataset Task Head  │  Linear → GELU → Linear
        │  (32K–1.7M params)  │  dataset-specific vocab
        └────┬────────────────┘
             │
        Per-token cross-entropy
        → anomaly score per session
             │
        YOBDM threshold
             │
        Normal / Anomaly
```

---

## Datasets

Download from [LogHub](https://github.com/logpai/loghub):

| Dataset | Sessions | Anomaly % | Session Method |
|---|---|---|---|
| HDFS | ~575K | ~2.9% | Block ID grouping |
| BGL | ~828K | ~42% | Sliding window (size=20, step=10) |
| ThunderBird (TB) | ~1.1M | ~33% | Sliding window (size=20, step=10) |

Place datasets in:
```
/content/drive/MyDrive/SENTINEL/data/HDFS/HDFS.log
/content/drive/MyDrive/SENTINEL/data/HDFS/anomaly_label.csv
/content/drive/MyDrive/SENTINEL/data/BGL/BGL.log
/content/drive/MyDrive/SENTINEL/data/TB/ThunderBird.log
```

Checkpoints are saved to:
```
/content/drive/MyDrive/SENTINEL/results/checkpoints/
  sentinel_fe_HDFS_balanced_best.pt
  sentinel_fe_BGL_best.pt
  sentinel_fe_TB_best.pt
```

---

## Quickstart (Google Colab)

```python
# 1. Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/dhanya-iiitk/SENTINEL-LogAD.git /content/SENTINEL-LogAD

# 2. Install dependencies
!pip install -q drain3 groq torch torchvision

# 3. Run setup notebook
# Open notebooks/00_setup.ipynb

# 4. Parse datasets
# Open notebooks/01_parsing.ipynb
# Set DATASET_NAME = 'HDFS'  # or 'BGL' or 'TB'

# 5. Train and evaluate
# Open notebooks/02_training.ipynb
```

---

## Main Results

### Detection Performance (F1 Score)

| Method | HDFS | BGL | ThunderBird |
|---|---|---|---|
| DeepLog | 0.955 | 0.693 | 0.733 |
| LogAnomaly | 0.962 | 0.712 | 0.748 |
| LogBERT | 0.973 | 0.741 | 0.769 |
| **SENTINEL** | **0.985** | **0.891** | **0.934** |

### Leave-One-Out Cross-Dataset Generalization

| Pretrain → Finetune | F1 |
|---|---|
| BGL+TB → HDFS | 0.981 |
| HDFS+TB → BGL | 0.876 |
| HDFS+BGL → TB | 0.921 |

---

## Deployment & Edge Feasibility

To assess suitability for resource-constrained IoT deployment, we measured
single-core, single-thread inference latency on an Intel Xeon CPU (Google Colab),
simulating a constrained compute budget.

| Dataset | Task Head Params | Mean Latency | P99 Latency |
|---------|-----------------|--------------|-------------|
| HDFS    | 32K             | 1.50 ms      | 1.71 ms     |
| BGL     | 293K            | 1.73 ms      | 1.84 ms     |
| TB      | 1.7M            | 2.96 ms      | 3.12 ms     |

**Config:** 1 core, 1 thread, batch=1 (single session), seq_len=20.

> **Note:** These numbers reflect server-CPU (Intel Xeon) performance under
> single-core throttling — not direct ARM/RPi 4 measurements. They establish
> an upper bound on compute requirement and demonstrate SENTINEL's minimal
> compute footprint. The frozen pretrained encoder is not re-evaluated at
> inference time; only the lightweight task head runs per session.
> Physical edge hardware evaluation on Raspberry Pi 4 is planned as future work.

See [`benchmarks/edge_latency_benchmark.py`](benchmarks/edge_latency_benchmark.py)
for the full reproducible benchmark script.

---

## Novelty Claims

| Component | Contribution |
|---|---|
| Two-stage transfer learning | Universal pretraining → dataset-specific fine-tuning without anomaly labels |
| Causal Transformer on log sequences | Next-event prediction as pretraining objective for log anomaly |
| Frozen encoder + lightweight head | <300K trainable params at fine-tune stage for HDFS/BGL |
| YOBDM threshold | Youden-optimal, no manual tuning required |
| IG attribution on log events | Token-level explainability for detected anomalies |
| Groq LLM explanation | Cyber threat narrative generation from IG scores |

---

## Requirements

```
torch >= 2.1.0
drain3
groq
scikit-learn
matplotlib
pandas
numpy
tqdm
```

---


---

## Human Explainability Evaluation

To assess the quality of SENTINEL's LLM-generated explanations, two domain experts independently rated 8 log sessions (7 anomalous + 1 normal) across 4 criteria on a 1–5 Likert scale:

| Criterion | Definition |
|---|---|
| **Accuracy** | Do the highlighted log events correctly identify the root cause? |
| **Clarity** | Is the LLM explanation clear and free of hallucinations? |
| **Relevance** | Does the explanation correctly name the failure mode or attack vector? |
| **Actionability** | Does the explanation suggest concrete remediation steps? |

### Results

| Case | HE (E1) | HE (E2) | HE (Mean) |
|---|---|---|---|
| BGL – Anomaly 1 (Microloader Assertion) | 4.50 | 4.50 | 4.50 |
| BGL – Anomaly 2 (TLB Error) | 5.00 | 5.00 | 5.00 |
| BGL – Normal | 5.00 | 5.00 | 5.00 |
| BGL – Borderline (TLB + Recovery) | 4.50 | 4.75 | 4.62 |
| HDFS – Anomaly 1 (Block Write Failure) | 4.50 | 4.50 | 4.50 |
| HDFS – Anomaly 2 (Socket Timeout + Exception) | 4.75 | 4.75 | 4.75 |
| TB – Anomaly 1 (InfiniBand MAD + Fatal Error) | 4.75 | 4.75 | 4.75 |
| TB – Anomaly 2 (MAD Failure Storm) | 4.75 | 4.75 | 4.75 |

**Overall HE Mean: 4.734 / 5** — **Cohen's κ = 0.920** (quadratic-weighted, 32 ratings)

See [`human_evaluation/compute_he.py`](human_evaluation/compute_he.py) for the scoring script and [`human_evaluation/he_results.txt`](human_evaluation/he_results.txt) for full output.
