# ============================================================
# sentinel/models/sentinel.py
# Full SENTINEL model:
#   GNN Encoder + Transformer Encoder + Fusion + RealNVP Flow
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math


# ─────────────────────────────────────────────
# 1.  Shared Event Embedding
# ─────────────────────────────────────────────

class EventEmbedding(nn.Module):
    """
    Learned embedding for discrete log event IDs.
    Shared between Transformer and GNN encoders.
    PAD index 0 has zero embedding (padding_idx).
    """
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        # Reset PAD embedding to zero
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0)

    def forward(self, x):
        return self.embedding(x)


# ─────────────────────────────────────────────
# 2.  Transformer Encoder
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, embed_dim]

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for log event sequences.
    Uses [CLS] token pooling for session-level representation.
    Input : event ID sequence [B, T]
    Output: session embedding  [B, embed_dim]
    """
    def __init__(
        self,
        embedding:   EventEmbedding,
        embed_dim:   int = 128,
        num_layers:  int = 2,
        num_heads:   int = 4,
        ffn_dim:     int = 256,
        dropout:     float = 0.1,
        max_len:     int = 512,
    ):
        super().__init__()
        self.embedding = embedding
        self.pos_enc   = PositionalEncoding(embed_dim, max_len, dropout)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        seq     : [B, T]  event ID sequences (padded)
        seq_len : [B]     actual sequence lengths
        Returns : [B, embed_dim]
        """
        B, T = seq.shape

        # Embed + positional encoding
        x = self.embedding(seq)           # [B, T, D]
        x = self.pos_enc(x)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)          # [B, T+1, D]

        # Padding mask (True = ignore position)
        # CLS position (index 0) is never masked
        pad_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=seq.device)
        for i in range(B):
            # Positions after actual length + 1 (CLS offset) are padding
            actual = seq_len[i].item()
            if actual < T:
                pad_mask[i, actual + 1:] = True

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        return x[:, 0, :]   # [B, D] — CLS token representation


# ─────────────────────────────────────────────
# 3.  GNN Encoder
# ─────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    2-layer GCN encoder for session-level log event graphs.
    Input : PyG Batch of session graphs
    Output: graph-level embedding [B, gnn_hidden]
    """
    def __init__(
        self,
        embedding:  EventEmbedding,
        gnn_hidden: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.embedding = embedding
        embed_dim = embedding.embedding.embedding_dim

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        in_dim = embed_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, gnn_hidden))
            self.bns.append(nn.BatchNorm1d(gnn_hidden))
            in_dim = gnn_hidden

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch) -> torch.Tensor:
        """
        graph_batch : PyG Batch object
        Returns     : [B, gnn_hidden]
        """
        x          = self.embedding(graph_batch.x)   # node embeddings
        edge_index = graph_batch.edge_index
        batch      = graph_batch.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global mean pooling → graph-level vector
        out = global_mean_pool(x, batch)   # [B, gnn_hidden]
        return out


# ─────────────────────────────────────────────
# 4.  Fusion Layer
# ─────────────────────────────────────────────

class FusionLayer(nn.Module):
    """
    Fuse Transformer and GNN representations.
    [z_transformer || z_gnn] → latent vector z
    """
    def __init__(self, transformer_dim: int, gnn_dim: int, latent_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim + gnn_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """
        z_t : [B, transformer_dim]
        z_g : [B, gnn_dim]
        Returns : [B, latent_dim]
        """
        return self.fc(torch.cat([z_t, z_g], dim=-1))


# ─────────────────────────────────────────────
# 5.  RealNVP Normalizing Flow
# ─────────────────────────────────────────────

class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.
    Splits input in half; one half transforms the other.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        half = input_dim // 2
        self.net_s = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - half),
            nn.Tanh(),   # bound scale
        )
        self.net_t = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - half),
        )
        self.half = half

    def forward(self, x):
        """Forward pass. Returns (z, log_det_jacobian)."""
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s = self.net_s(x1)
        t = self.net_t(x1)
        z2 = x2 * torch.exp(s) + t
        z  = torch.cat([x1, z2], dim=-1)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """Inverse pass for sampling."""
        z1, z2 = z[:, :self.half], z[:, self.half:]
        s = self.net_s(z1)
        t = self.net_t(z1)
        x2 = (z2 - t) * torch.exp(-s)
        x  = torch.cat([z1, x2], dim=-1)
        return x


class RealNVP(nn.Module):
    """
    RealNVP normalizing flow for density estimation.
    Trained on normal session embeddings only.
    Anomaly score = negative log-likelihood under the learned distribution.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        # Permutation matrices (fixed alternating)
        perms = []
        for i in range(num_layers):
            perm = torch.arange(input_dim)
            if i % 2 == 1:
                perm = torch.flip(perm, dims=[0])
            perms.append(perm)
        self.register_buffer('perms', torch.stack(perms))

    def forward(self, x: torch.Tensor):
        """
        x       : [B, input_dim]  fused latent representation
        Returns : (z, total_log_det)  for training
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        for i, layer in enumerate(self.layers):
            z = z[:, self.perms[i]]   # permute
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) under the flow.
        = log p_z(f(x)) + log |det J_f(x)|
        where p_z = standard Normal.
        Returns : [B]  log-likelihood per sample
        """
        z, log_det = self.forward(x)
        # Log-likelihood under standard Normal prior
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_pz + log_det

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score = negative log-likelihood.
        Higher score → more anomalous.
        Returns : [B]
        """
        return -self.log_prob(x)


# ─────────────────────────────────────────────
# 6.  Full SENTINEL Model
# ─────────────────────────────────────────────

class SENTINEL(nn.Module):
    """
    Full SENTINEL framework:
      EventEmbedding (shared)
        ├── TransformerEncoder  → z_t
        └── GNNEncoder          → z_g
              ↓
          FusionLayer           → z
              ↓
          RealNVP Flow          → anomaly score
    """
    def __init__(
        self,
        vocab_size:      int   = 300,
        embed_dim:       int   = 128,
        pad_idx:         int   = 0,
        transformer_layers: int = 2,
        num_heads:       int   = 4,
        ffn_dim:         int   = 256,
        gnn_hidden:      int   = 128,
        gnn_layers:      int   = 2,
        latent_dim:      int   = 256,
        flow_layers:     int   = 4,
        flow_hidden:     int   = 256,
        dropout:         float = 0.1,
        max_len:         int   = 512,
    ):
        super().__init__()

        # Shared embedding
        self.embedding = EventEmbedding(vocab_size, embed_dim, pad_idx)

        # Dual encoders
        self.transformer = TransformerEncoder(
            self.embedding, embed_dim, transformer_layers,
            num_heads, ffn_dim, dropout, max_len,
        )
        self.gnn = GNNEncoder(
            self.embedding, gnn_hidden, gnn_layers, dropout,
        )

        # Fusion
        self.fusion = FusionLayer(embed_dim, gnn_hidden, latent_dim, dropout)

        # Normalizing Flow
        self.flow = RealNVP(latent_dim, flow_hidden, flow_layers)

    def encode(self, seq, seq_len, graph) -> torch.Tensor:
        """Returns fused latent representation z. [B, latent_dim]"""
        z_t = self.transformer(seq, seq_len)
        z_g = self.gnn(graph)
        z   = self.fusion(z_t, z_g)
        return z

    def forward(self, seq, seq_len, graph):
        """
        Training forward pass.
        Returns negative log-likelihood loss (scalar).
        """
        z    = self.encode(seq, seq_len, graph)
        loss = -self.flow.log_prob(z).mean()
        return loss

    def score(self, seq, seq_len, graph) -> torch.Tensor:
        """
        Inference: compute anomaly scores. [B]
        """
        with torch.no_grad():
            z = self.encode(seq, seq_len, graph)
            return self.flow.anomaly_score(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
