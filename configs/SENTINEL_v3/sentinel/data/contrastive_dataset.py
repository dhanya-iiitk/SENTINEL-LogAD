# ============================================================
# sentinel/data/contrastive_dataset.py
# Contrastive Dataset — returns (normal, augmented) pairs
# ============================================================

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# 1.  Session Augmentation Strategies
# ─────────────────────────────────────────────

def augment_session(
    event_ids:  List[int],
    vocab_size: int,
    strategy:   str = 'random',
    pad_idx:    int = 0,
) -> List[int]:
    """
    Create a pseudo-anomalous session from a normal session.

    Three strategies simulate different anomaly types:

    1. shuffle  → breaks sequential dependencies
                  (simulates out-of-order execution)
    2. replace  → injects random events
                  (simulates unexpected system behavior)
    3. insert   → inserts foreign events
                  (simulates intrusion/injection)

    All strategies produce sessions that look structurally
    different from normal — forcing the encoder to map them
    to different regions of the latent space.
    """
    # Filter out PAD tokens before augmentation
    eids = [e for e in event_ids if e != pad_idx]
    if len(eids) == 0:
        return event_ids

    if strategy == 'random':
        strategy = random.choice(['shuffle', 'replace', 'insert'])

    eids = eids.copy()

    if strategy == 'shuffle':
        # Shuffle entire sequence — breaks all sequential patterns
        # Keep first and last event to maintain some structure
        if len(eids) > 2:
            middle = eids[1:-1]
            random.shuffle(middle)
            eids = [eids[0]] + middle + [eids[-1]]
        else:
            random.shuffle(eids)

    elif strategy == 'replace':
        # Replace 30-50% of events with random events
        n_replace = max(1, int(len(eids) * random.uniform(0.3, 0.5)))
        indices   = random.sample(range(len(eids)), 
                                  min(n_replace, len(eids)))
        for idx in indices:
            # Replace with random event (not PAD, not same event)
            new_event = random.randint(1, vocab_size - 1)
            while new_event == eids[idx]:
                new_event = random.randint(1, vocab_size - 1)
            eids[idx] = new_event

    elif strategy == 'insert':
        # Insert random foreign events at random positions
        n_insert = max(1, int(len(eids) * 0.3))
        for _ in range(n_insert):
            pos       = random.randint(0, len(eids))
            new_event = random.randint(1, vocab_size - 1)
            eids.insert(pos, new_event)
        # Truncate to original length
        eids = eids[:len(event_ids)]

    return eids


def augment_batch(
    event_ids_list: List[List[int]],
    vocab_size:     int,
    pad_idx:        int = 0,
) -> List[List[int]]:
    """Augment a batch of sessions with random strategies."""
    return [augment_session(eids, vocab_size, 'random', pad_idx)
            for eids in event_ids_list]


# ─────────────────────────────────────────────
# 2.  Train/Val/Test Split
# ─────────────────────────────────────────────

def split_sessions(
    sessions:    List[List[int]],
    labels:      List[int],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
    dev_mode:    bool  = True,
    dev_samples: int   = 2000,
) -> Tuple[dict, dict, dict]:
    """
    Semi-supervised split:
      Train  : normal only
      Val    : normal + anomalous
      Test   : normal + anomalous
    """
    np.random.seed(seed)
    random.seed(seed)

    normal_idx  = [i for i, l in enumerate(labels) if l == 0]
    anomaly_idx = [i for i, l in enumerate(labels) if l == 1]

    np.random.shuffle(normal_idx)
    np.random.shuffle(anomaly_idx)

    if dev_mode:
        n_normal  = min(int(dev_samples * 0.85), len(normal_idx))
        n_anomaly = min(int(dev_samples * 0.15), len(anomaly_idx))
        normal_idx  = normal_idx[:n_normal]
        anomaly_idx = anomaly_idx[:n_anomaly]
        print(f"[DEV MODE] {n_normal} normal + {n_anomaly} anomalous")

    n       = len(normal_idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx   = normal_idx[:n_train]
    val_normal  = normal_idx[n_train: n_train + n_val]
    val_anomaly = anomaly_idx[:len(anomaly_idx) // 2]
    val_idx     = val_normal + val_anomaly
    test_normal  = normal_idx[n_train + n_val:]
    test_anomaly = anomaly_idx[len(anomaly_idx) // 2:]
    test_idx     = test_normal + test_anomaly

    def make(indices):
        return {'sessions': [sessions[i] for i in indices],
                'labels':   [labels[i]   for i in indices]}

    train = make(train_idx)
    val   = make(val_idx)
    test  = make(test_idx)

    print(f"\nSplit summary:")
    print(f"  Train : {len(train['sessions']):,} "
          f"(normal={sum(1 for l in train['labels'] if l==0):,}, "
          f"anomaly={sum(1 for l in train['labels'] if l==1):,})")
    print(f"  Val   : {len(val['sessions']):,} "
          f"(normal={sum(1 for l in val['labels'] if l==0):,}, "
          f"anomaly={sum(1 for l in val['labels'] if l==1):,})")
    print(f"  Test  : {len(test['sessions']):,} "
          f"(normal={sum(1 for l in test['labels'] if l==0):,}, "
          f"anomaly={sum(1 for l in test['labels'] if l==1):,})")

    return train, val, test


# ─────────────────────────────────────────────
# 3.  Graph Construction
# ─────────────────────────────────────────────

def build_session_graph(event_ids: List[int]) -> Data:
    """
    Build directed session graph.
    Nodes = unique event types
    Edges = sequential co-occurrence
    """
    unique_events  = list(dict.fromkeys(event_ids))
    event_to_node  = {e: i for i, e in enumerate(unique_events)}
    x              = torch.tensor(unique_events, dtype=torch.long)

    edge_set = set()
    for i in range(len(event_ids) - 1):
        src = event_to_node[event_ids[i]]
        dst = event_to_node[event_ids[i + 1]]
        if src != dst:
            edge_set.add((src, dst))

    if len(edge_set) == 0:
        edge_set = {(0, 0)}

    edge_index = torch.tensor(
        list(edge_set), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ─────────────────────────────────────────────
# 4.  Contrastive Training Dataset
# ─────────────────────────────────────────────

class ContrastiveLogDataset(Dataset):
    """
    Training dataset that returns (normal, augmented) pairs.

    Each item:
      seq_norm  : padded normal event sequence   [T]
      seq_aug   : padded augmented sequence      [T]
      len_norm  : actual length of normal seq    scalar
      len_aug   : actual length of augmented seq scalar
      graph_norm: PyG graph of normal session
      graph_aug : PyG graph of augmented session
    """
    def __init__(
        self,
        sessions:    List[List[int]],
        vocab_size:  int,
        max_seq_len: int = 50,
        pad_idx:     int = 0,
    ):
        self.sessions    = sessions
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx

    def __len__(self):
        return len(self.sessions)

    def _prepare(self, eids: List[int]):
        eids    = eids[:self.max_seq_len]
        seq_len = len(eids)
        padded  = eids + [self.pad_idx] * (self.max_seq_len - seq_len)
        seq     = torch.tensor(padded, dtype=torch.long)
        graph   = build_session_graph(eids)
        return seq, torch.tensor(seq_len, dtype=torch.long), graph

    def __getitem__(self, idx):
        normal_eids = self.sessions[idx]

        # Create augmented (pseudo-anomalous) version
        aug_eids = augment_session(
            normal_eids, self.vocab_size,
            strategy='random', pad_idx=self.pad_idx
        )

        seq_n, len_n, graph_n = self._prepare(normal_eids)
        seq_a, len_a, graph_a = self._prepare(aug_eids)

        return {
            'seq_norm':   seq_n,
            'seq_aug':    seq_a,
            'len_norm':   len_n,
            'len_aug':    len_a,
            'graph_norm': graph_n,
            'graph_aug':  graph_a,
        }


# ─────────────────────────────────────────────
# 5.  Evaluation Dataset (no augmentation)
# ─────────────────────────────────────────────

class EvalLogDataset(Dataset):
    """Standard dataset for val/test — no augmentation."""
    def __init__(self, sessions, labels,
                 max_seq_len=50, pad_idx=0):
        self.sessions    = sessions
        self.labels      = labels
        self.max_seq_len = max_seq_len
        self.pad_idx     = pad_idx

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        eids    = self.sessions[idx][:self.max_seq_len]
        seq_len = len(eids)
        padded  = eids + [self.pad_idx] * (self.max_seq_len - seq_len)
        return {
            'seq':    torch.tensor(padded,   dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'graph':  build_session_graph(eids),
            'label':  torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 6.  Collate Functions
# ─────────────────────────────────────────────

def contrastive_collate(batch):
    """Collate contrastive training pairs."""
    from torch_geometric.data import Batch as PyGBatch
    return {
        'seq_norm':   torch.stack([b['seq_norm']  for b in batch]),
        'seq_aug':    torch.stack([b['seq_aug']   for b in batch]),
        'len_norm':   torch.stack([b['len_norm']  for b in batch]),
        'len_aug':    torch.stack([b['len_aug']   for b in batch]),
        'graph_norm': PyGBatch.from_data_list([b['graph_norm'] for b in batch]),
        'graph_aug':  PyGBatch.from_data_list([b['graph_aug']  for b in batch]),
    }


def eval_collate(batch):
    """Collate evaluation batches."""
    from torch_geometric.data import Batch as PyGBatch
    return {
        'seq':    torch.stack([b['seq']    for b in batch]),
        'seq_len': torch.stack([b['seq_len'] for b in batch]),
        'graph':  PyGBatch.from_data_list([b['graph'] for b in batch]),
        'label':  torch.stack([b['label']  for b in batch]),
    }


# ─────────────────────────────────────────────
# 7.  DataLoader Factory
# ─────────────────────────────────────────────

def get_dataloaders_v3(
    train_data:  dict,
    val_data:    dict,
    test_data:   dict,
    vocab_size:  int,
    max_seq_len: int = 50,
    pad_idx:     int = 0,
    batch_size:  int = 64,
    num_workers: int = 2,
):
    """
    Create contrastive train loader + standard val/test loaders.
    """
    train_ds = ContrastiveLogDataset(
        train_data['sessions'], vocab_size, max_seq_len, pad_idx)
    val_ds   = EvalLogDataset(
        val_data['sessions'], val_data['labels'], max_seq_len, pad_idx)
    test_ds  = EvalLogDataset(
        test_data['sessions'], test_data['labels'], max_seq_len, pad_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=contrastive_collate,
        num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=eval_collate,
        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=eval_collate,
        num_workers=num_workers, pin_memory=True)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"  Test batches  : {len(test_loader):,}")

    return train_loader, val_loader, test_loader
