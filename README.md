# SENTINEL
## **SEM**antic-graph **T**ransformer with **N**ormalizing flow for **L**og anomaly

A semi-supervised log anomaly detection framework combining:
- **Drain** log parsing → structured event templates
- **GNN Encoder** → structural session graph representation
- **Transformer Encoder** → sequential session representation  
- **Fusion Layer** → unified latent representation
- **RealNVP Normalizing Flow** → density-based anomaly scoring
- **YOBDM** → Youden-optimal threshold
- **IG + Groq LLM** → explainable cyber threat analysis

---

## Repository Structure

```
SENTINEL/
├── configs/
│   └── config.py              # All hyperparameters
├── sentinel/
│   ├── data/
│   │   ├── parser.py          # Drain parsing + session construction
│   │   └── dataset.py         # PyTorch Dataset + graph builder + DataLoader
│   ├── models/
│   │   └── sentinel.py        # Full SENTINEL model
│   ├── utils/
│   │   └── trainer.py         # Training loop + YOBDM + evaluation
│   └── explain/
│       └── llm_explain.py     # IG attribution + Groq LLM explanation
├── notebooks/
│   ├── 00_setup.ipynb         # Environment + dependencies
│   ├── 01_parsing.ipynb       # Dataset parsing (HDFS, BGL, OpenStack)
│   └── 02_training.ipynb      # Training + evaluation
└── README.md
```

---

## Datasets

Download from [LogHub](https://github.com/logpai/loghub):

| Dataset | Sessions | Anomaly % | Session Method |
|---|---|---|---|
| HDFS | ~575K | ~2.9% | Block ID grouping |
| BGL | ~828 | ~42% | Sliding window (size=20) |
| OpenStack | ~10K | variable | Request ID grouping |

Place datasets in:
```
/content/drive/MyDrive/SENTINEL/data/HDFS/HDFS.log
/content/drive/MyDrive/SENTINEL/data/HDFS/anomaly_label.csv
/content/drive/MyDrive/SENTINEL/data/BGL/BGL.log
/content/drive/MyDrive/SENTINEL/data/OpenStack/*.log
```

---

## Quickstart (Colab Pro+)

```python
# 1. Run Notebook 00 — setup
# 2. Run Notebook 01 — parse datasets
# 3. Run Notebook 02 — train + evaluate

# Change dataset:
DATASET_NAME = 'HDFS'    # or 'BGL' or 'OpenStack'
DEV_MODE     = True      # False for full dataset (paper results)
```

---

## Novelty Claims

| Component | Contribution |
|---|---|
| GNN + Transformer dual encoding | Structural + sequential log representation |
| RealNVP flow on log latent space | **Novel** — no prior log anomaly paper uses NF as primary scorer |
| YOBDM threshold | Youden-optimal, no manual tuning |
| IG attribution on log events | Token-level explainability for log anomaly |
| Groq LLM explanation | Cyber threat narrative generation |

---

## Citation

```
@article{sentinel2025,
  title   = {SENTINEL: A Semi-Supervised Log Anomaly Detection Framework
             via Semantic-Graph Transformer and Normalizing Flow},
  author  = {Dhanya P and Arun Cyril Jose},
  journal = {IEEE Transactions on ...},
  year    = {2025}
}
```

---

## Requirements

```
torch >= 2.1.0
torch-geometric
drain3
groq
scikit-learn
matplotlib
pandas
numpy
tqdm
```
