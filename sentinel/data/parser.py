# ============================================================
# sentinel/data/parser.py
# Drain-based log parsing + session construction
# for HDFS, BGL, OpenStack datasets
# ============================================================

import re
import os
import json
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1.  Drain Parser Wrapper
# ─────────────────────────────────────────────

class DrainParser:
    """
    Thin wrapper around drain3.TemplateMiner.
    Parses raw log lines → (template_id, template_str) pairs.
    """

    def __init__(self, depth: int = 4, st: float = 0.5, max_children: int = 100):
        config = TemplateMinerConfig()
        config.drain_depth        = depth
        config.drain_sim_th       = st
        config.drain_max_children = max_children
        config.parametrize_numeric_tokens = True
        self.miner = TemplateMiner(config=config)
        self.template2id: Dict[str, int] = {}   # template_str → int ID
        self._next_id = 1                        # 0 reserved for PAD

    def parse_line(self, log_line: str) -> Tuple[int, str]:
        """Parse one log line. Returns (event_id, template_str)."""
        result   = self.miner.add_log_message(log_line.strip())
        template = result["template_mined"]
        if template not in self.template2id:
            self.template2id[template] = self._next_id
            self._next_id += 1
        return self.template2id[template], template

    def parse_lines(self, lines: List[str]) -> Tuple[List[int], List[str]]:
        """Parse a list of log lines. Returns (event_ids, templates)."""
        ids, templates = [], []
        for line in tqdm(lines, desc="Drain parsing", leave=False):
            eid, tmpl = self.parse_line(line)
            ids.append(eid)
            templates.append(tmpl)
        return ids, templates

    @property
    def vocab_size(self) -> int:
        return self._next_id   # includes PAD=0

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'template2id': self.template2id,
                         'next_id': self._next_id}, f)
        print(f"Parser vocabulary saved → {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.template2id = obj['template2id']
        self._next_id    = obj['next_id']
        print(f"Parser vocabulary loaded ← {path}  (vocab={self.vocab_size})")


# ─────────────────────────────────────────────
# 2.  Log Preprocessors (dataset-specific)
# ─────────────────────────────────────────────

def preprocess_hdfs(log_line: str) -> Tuple[str, Optional[str]]:
    """
    Extract (cleaned_line, block_id) from HDFS raw log.
    Example raw line:
      081109 203615 148 INFO dfs.DataNode$PacketResponder: blk_-1608999687919862906 ...
    """
    blk_pattern = re.compile(r'(blk_-?\d+)')
    match = blk_pattern.search(log_line)
    block_id = match.group(1) if match else None
    # Remove timestamps and numeric fields for Drain
    cleaned = re.sub(r'^\d+\s+\d+\s+\d+\s+', '', log_line.strip())
    return cleaned, block_id


def preprocess_bgl(log_line: str) -> str:
    """
    Strip alert tag and leading fields from BGL raw log.
    Example: - 1117838570 2005.06.03 ... PROGRAM FATAL ...
    """
    # Remove leading '-' or alert tag
    cleaned = re.sub(r'^[-\w]+\s+\d+\s+[\d.]+\s+', '', log_line.strip())
    return cleaned


def preprocess_openstack(log_line: str) -> Tuple[str, Optional[str]]:
    """
    Extract (cleaned_line, request_id) from OpenStack raw log.
    Example: nova-api.log ... req-xxxxxxxx-xxxx-... ...
    """
    req_pattern = re.compile(r'req-([\w-]+)')
    match = req_pattern.search(log_line)
    request_id = match.group(0) if match else None
    cleaned = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+', '', log_line.strip())
    return cleaned, request_id


# ─────────────────────────────────────────────
# 3.  Session Constructors
# ─────────────────────────────────────────────

def build_sessions_hdfs(
    raw_log_path: str,
    label_path:   str,
    parser:       DrainParser,
    max_seq_len:  int = 50,
    min_seq_len:  int = 2,
) -> Tuple[List[List[int]], List[int], List[str]]:
    """
    HDFS: group log lines by Block ID.
    Returns (sessions, labels, block_ids).
    sessions : list of event-ID sequences
    labels   : 0=normal, 1=anomalous
    """
    print("Building HDFS sessions...")

    # Load anomaly labels
    label_df = pd.read_csv(label_path)   # columns: BlockId, Label
    label_map = dict(zip(label_df['BlockId'],
                         label_df['Label'].map({'Normal': 0, 'Anomaly': 1})))

    # Group lines by block_id
    block_lines: Dict[str, List[str]] = defaultdict(list)
    with open(raw_log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading HDFS logs"):
            cleaned, block_id = preprocess_hdfs(line)
            if block_id:
                block_lines[block_id].append(cleaned)

    # Parse and build sessions
    sessions, labels, block_ids = [], [], []
    for block_id, lines in tqdm(block_lines.items(), desc="Parsing HDFS blocks"):
        if len(lines) < min_seq_len:
            continue
        eids, _ = parser.parse_lines(lines)
        # Truncate / keep as-is (padding done in Dataset class)
        eids = eids[:max_seq_len]
        label = label_map.get(block_id, 0)
        sessions.append(eids)
        labels.append(label)
        block_ids.append(block_id)

    print(f"HDFS → {len(sessions)} sessions | "
          f"Normal: {sum(1 for l in labels if l==0)} | "
          f"Anomalous: {sum(1 for l in labels if l==1)}")
    return sessions, labels, block_ids


def build_sessions_bgl(
    raw_log_path: str,
    parser:       DrainParser,
    window_size:  int = 20,
    stride:       int = 1,
    max_seq_len:  int = 50,
    min_seq_len:  int = 2,
) -> Tuple[List[List[int]], List[int]]:
    """
    BGL: sliding window session construction.
    Each log line has an alert label (non '-' = anomalous).
    Window is anomalous if ANY line in it is anomalous.
    Returns (sessions, labels).
    """
    print("Building BGL sessions (sliding window)...")

    raw_lines, raw_labels = [], []
    with open(raw_log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading BGL logs"):
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            alert_tag = parts[0]
            log_body  = parts[1] if len(parts) > 1 else ''
            is_anomaly = 0 if alert_tag == '-' else 1
            cleaned = preprocess_bgl(line)
            raw_lines.append(cleaned)
            raw_labels.append(is_anomaly)

    # Parse all lines first (more efficient)
    print("Running Drain on BGL...")
    eids, _ = parser.parse_lines(raw_lines)

    # Sliding window
    sessions, labels = [], []
    for start in range(0, len(eids) - window_size + 1, stride):
        window_eids   = eids[start: start + window_size]
        window_labels = raw_labels[start: start + window_size]
        if len(window_eids) < min_seq_len:
            continue
        window_eids = window_eids[:max_seq_len]
        sessions.append(window_eids)
        labels.append(1 if any(l == 1 for l in window_labels) else 0)

    print(f"BGL → {len(sessions)} sessions | "
          f"Normal: {sum(1 for l in labels if l==0)} | "
          f"Anomalous: {sum(1 for l in labels if l==1)}")
    return sessions, labels


def build_sessions_openstack(
    log_dir:     str,
    parser:      DrainParser,
    max_seq_len: int = 100,
    min_seq_len: int = 2,
) -> Tuple[List[List[int]], List[int], List[str]]:
    """
    OpenStack: group by request_id across multiple log files.
    Returns (sessions, labels, request_ids).
    Labels derived from 'error' keyword presence (unsupervised proxy)
    or from a provided label file if available.
    """
    print("Building OpenStack sessions...")

    req_lines:  Dict[str, List[str]] = defaultdict(list)
    req_errors: Dict[str, int]       = defaultdict(int)

    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    for fname in log_files:
        fpath = os.path.join(log_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                cleaned, req_id = preprocess_openstack(line)
                if req_id:
                    req_lines[req_id].append(cleaned)
                    if 'error' in line.lower() or 'exception' in line.lower():
                        req_errors[req_id] += 1

    sessions, labels, req_ids = [], [], []
    for req_id, lines in tqdm(req_lines.items(), desc="Parsing OpenStack requests"):
        if len(lines) < min_seq_len:
            continue
        eids, _ = parser.parse_lines(lines)
        eids = eids[:max_seq_len]
        label = 1 if req_errors[req_id] > 0 else 0
        sessions.append(eids)
        labels.append(label)
        req_ids.append(req_id)

    print(f"OpenStack → {len(sessions)} sessions | "
          f"Normal: {sum(1 for l in labels if l==0)} | "
          f"Anomalous: {sum(1 for l in labels if l==1)}")
    return sessions, labels, req_ids


# ─────────────────────────────────────────────
# 4.  Save / Load parsed sessions
# ─────────────────────────────────────────────

def save_parsed(sessions, labels, path: str, extra=None):
    obj = {'sessions': sessions, 'labels': labels}
    if extra:
        obj.update(extra)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved {len(sessions)} sessions → {path}")


def load_parsed(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded {len(obj['sessions'])} sessions ← {path}")
    return obj
