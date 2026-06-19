# ============================================================
# sentinel/data/dataset.py
# PyTorch Dataset + Graph construction + DataLoader
# ============================================================

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from collections import defaultdict
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# 1.  Train / Val / Test Split
# ─────────────────────────────────────────────

def split_sessions(
    sessions: List[List[int]],
    labels:   List[int],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
    dev_mode:    bool  = True,
    dev_samples: int   = 2000,
) -> Tuple[dict, dict, dict]:
    """
    Semi-supervised split:
      Train  : normal sessions only (no anomalies seen)
      Val    : 10% normal + 50% anomalous
      Test   : remaining normal + remaining anomalous

    Returns dicts with keys 'sessions' and 'labels'.
    """
    np.random.seed(seed)

    normal_idx  = [i for i, l in enumerate(labels) if l == 0]
    anomaly_idx = [i for i, l in enumerate(labels) if l == 1]

    np.random.shuffle(normal_idx)
    np.random.shuffle(anomaly_idx)

    # Dev mode — subsample for fast iteration
    if dev_mode:
        n_normal  = min(int(dev_samples * 0.85), len(normal_idx))
        n_anomaly = min(int(dev_samples * 0.15), len(anomaly_idx))
        normal_idx  = normal_idx[:n_normal]
        anomaly_idx = anomaly_idx[:n_anomaly]
        print(f"[DEV MODE] Using {n_normal} normal + {n_anomaly} anomalous sessions")

    n  = len(normal_idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = normal_idx[:n_train]

    val_normal  = normal_idx[n_train: n_train + n_val]
    val_anomaly = anomaly_idx[: len(anomaly_idx) // 2]
    val_idx     = val_normal + val_anomaly

    test_normal  = normal_idx[n_train + n_val:]
    test_anomaly = anomaly_idx[len(anomaly_idx) // 2:]
    test_idx     = test_normal + test_anomaly

    def make_split(indices):
        return {
            'sessions': [sessions[i] for i in indices],
            'labels':   [labels[i]   for i in indices],
        }

    train = make_split(train_idx)
    val   = make_split(val_idx)
    test  = make_split(test_idx)

    print(f"\nSplit summary:")
    print(f"  Train  : {len(train['sessions'])} sessions "
          f"(normal={sum(1 for l in train['labels'] if l==0)}, "
          f"anomaly={sum(1 for l in train['labels'] if l==1)})")
    print(f"  Val    : {len(val['sessions'])} sessions "
          f"(normal={sum(1 for l in val['labels'] if l==0)}, "
          f"anomaly={sum(1 for l in val['labels'] if l==1)})")
    print(f"  Test   : {len(test['sessions'])} sessions "
          f"(normal={sum(1 for l in test['labels'] if l==0)}, "
          f"anomaly={sum(1 for l in test['labels'] if l==1)})")

    return train, val, test


# ─────────────────────────────────────────────
# 2.  Session-level Graph Construction
# ─────────────────────────────────────────────

def build_session_graph(event_ids: List[int]) -> Data:
    """
    Build a directed graph for a single log session.

    Nodes  : unique event types in the session
    Edges  : sequential co-occurrence (E_i → E_{i+1})
    Node features : event ID (used as embedding lookup index)

    Returns torch_geometric.data.Data object.
    """
    unique_events = list(dict.fromkeys(event_ids))   # preserve order, deduplicate
    event_to_node = {e: i for i, e in enumerate(unique_events)}

    # Node features: event IDs (integers for embedding lookup)
    x = torch.tensor(unique_events, dtype=torch.long)   # [num_nodes]

    # Edges: consecutive pairs in sequence
    edge_set = set()
    for i in range(len(event_ids) - 1):
        src = event_to_node[event_ids[i]]
        dst = event_to_node[event_ids[i + 1]]
        if src != dst:
            edge_set.add((src, dst))

    if len(edge_set) == 0:
        # Self-loop fallback for single-node graphs
        edge_set = {(0, 0)}

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────
# 3.  SENTINEL Dataset Class
# ─────────────────────────────────────────────

class SENTINELDataset(Dataset):
    """
    PyTorch Dataset for SENTINEL.

    Each item returns:
      seq      : padded event-ID sequence  [max_seq_len]
      seq_len  : actual (unpadded) length  scalar
      graph    : torch_geometric Data object
      label    : 0 or 1
    """

    def __init__(
        self,
        sessions:    List[List[int]],
        labels:      List[int],
        max_seq_len: int = 50,
        pad_idx:     int = 0,
    ):
        assert len(sessions) == len(labels)
        self.sessions    = sessions
        self.labels      = labels
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        eids  = self.sessions[idx]
        label = self.labels[idx]

        # Truncate
        eids = eids[:self.max_seq_len]
        seq_len = len(eids)

        # Pad
        padded = eids + [self.pad_idx] * (self.max_seq_len - seq_len)
        seq    = torch.tensor(padded, dtype=torch.long)

        # Graph
        graph = build_session_graph(eids)

        return {
            'seq':    seq,
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'graph':  graph,
            'label':  torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 4.  Custom Collate (handles PyG graphs)
# ─────────────────────────────────────────────

def sentinel_collate(batch):
    """
    Custom collate function to handle mixed
    tensor + PyG Data objects in a batch.
    """
    seqs     = torch.stack([b['seq']     for b in batch])
    seq_lens = torch.stack([b['seq_len'] for b in batch])
    labels   = torch.stack([b['label']   for b in batch])
    graphs   = Batch.from_data_list([b['graph'] for b in batch])

    return {
        'seq':    seqs,
        'seq_len': seq_lens,
        'graph':  graphs,
        'label':  labels,
    }


# ─────────────────────────────────────────────
# 5.  DataLoader Factory
# ─────────────────────────────────────────────

def get_dataloaders(
    train_data:  dict,
    val_data:    dict,
    test_data:   dict,
    max_seq_len: int = 50,
    pad_idx:     int = 0,
    batch_size:  int = 64,
    num_workers: int = 2,
):
    """
    Create train / val / test DataLoaders.
    Train loader shuffles; val/test do not.
    """
    train_ds = SENTINELDataset(train_data['sessions'], train_data['labels'],
                                max_seq_len, pad_idx)
    val_ds   = SENTINELDataset(val_data['sessions'],   val_data['labels'],
                                max_seq_len, pad_idx)
    test_ds  = SENTINELDataset(test_data['sessions'],  test_data['labels'],
                                max_seq_len, pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=sentinel_collate,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=sentinel_collate,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=sentinel_collate,
                              num_workers=num_workers, pin_memory=True)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader
