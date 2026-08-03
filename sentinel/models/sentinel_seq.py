"""
SENTINEL — Semi-supervised Log Anomaly Detection
Stage 1: Universal pretraining (SENTINELSeqBase)
Stage 2: Feature extraction (SENTINELSeqFeatureExtract)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SENTINELSeqBase(nn.Module):
    """
    Causal Transformer encoder pretrained on
    combined normal log corpus via next-event
    prediction. Serves as universal log feature
    extractor after pretraining.
    Architecture: L=2, H=4, d=128, FFN=256
    """
    def __init__(self, vocab_size, embed_dim=128,
                 num_layers=2, num_heads=4,
                 ffn_dim=256, dropout=0.1,
                 max_len=512, pad_idx=0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(
            0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def _encode(self, seq, seq_len):
        B, T = seq.shape
        x    = self.embedding(seq) + self.pe[:, :T]
        causal_mask = torch.triu(
            torch.ones(T, T, device=seq.device),
            diagonal=1).bool()
        pad_mask = torch.zeros(
            B, T, dtype=torch.bool,
            device=seq.device)
        for i in range(B):
            actual = seq_len[i].item()
            if actual < T:
                pad_mask[i, actual:] = True
        h = self.transformer(
            x, mask=causal_mask,
            src_key_padding_mask=pad_mask)
        return self.norm(h)

    def forward(self, seq, seq_len):
        h      = self._encode(seq, seq_len)
        logits = self.head(h[:, :-1, :])
        target = seq[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.pad_idx)

    def anomaly_score(self, seq, seq_len):
        with torch.no_grad():
            h      = self._encode(seq, seq_len)
            logits = self.head(h[:, :-1, :])
            target = seq[:, 1:]
            B, T1, V = logits.shape
            per_token = F.cross_entropy(
                logits.reshape(-1, V),
                target.reshape(-1),
                ignore_index=self.pad_idx,
                reduction='none').reshape(B, T1)
            mask = (target != self.pad_idx).float()
            return (per_token * mask).sum(dim=1) / \
                   (mask.sum(dim=1) + 1e-9)

    def count_parameters(self):
        return sum(p.numel() for p in
                   self.parameters()
                   if p.requires_grad)


class SENTINELSeqFeatureExtract(nn.Module):
    """
    Feature extraction fine-tuning stage.
    Frozen pretrained encoder + lightweight
    dataset-specific prediction head.
    Trained on normal sessions only.
    Head params: <300K for HDFS/BGL
    """
    def __init__(self, base_model,
                 task_vocab_size, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.base    = base_model
        for param in self.base.parameters():
            param.requires_grad_(False)
        embed_dim      = base_model.embed_dim
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, task_vocab_size))
        nn.init.xavier_uniform_(
            self.task_head[0].weight)
        nn.init.xavier_uniform_(
            self.task_head[2].weight)
        head_params = sum(
            p.numel() for p in
            self.task_head.parameters())
        print(f"Task head params: {head_params:,}")

    def _encode(self, seq, seq_len):
        with torch.no_grad():
            return self.base._encode(seq, seq_len)

    def forward(self, seq, seq_len):
        h      = self._encode(seq, seq_len)
        logits = self.task_head(h[:, :-1, :])
        target = seq[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.pad_idx)

    def anomaly_score(self, seq, seq_len):
        with torch.no_grad():
            h      = self._encode(seq, seq_len)
            logits = self.task_head(h[:, :-1, :])
            target = seq[:, 1:]
            B, T1, V = logits.shape
            per_token = F.cross_entropy(
                logits.reshape(-1, V),
                target.reshape(-1),
                ignore_index=self.pad_idx,
                reduction='none').reshape(B, T1)
            mask = (target != self.pad_idx).float()
            return (per_token * mask).sum(dim=1) / \
                   (mask.sum(dim=1) + 1e-9)

    def count_parameters(self):
        return sum(p.numel() for p in
                   self.parameters()
                   if p.requires_grad)
