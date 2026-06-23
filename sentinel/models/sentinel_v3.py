# ============================================================
# sentinel/models/sentinel_v3.py
# SENTINELv3 — Contrastive Augmentation + Flow Scoring
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
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0)

    def forward(self, x):
        return self.embedding(x)


# ─────────────────────────────────────────────
# 2.  Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────
# 3.  Transformer Encoder
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, embedding, embed_dim=128, num_layers=2,
                 num_heads=4, ffn_dim=256, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding  = embedding
        self.pos_enc    = PositionalEncoding(embed_dim, max_len, dropout)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, seq, seq_len):
        B, T   = seq.shape
        x      = self.pos_enc(self.embedding(seq))
        cls    = self.cls_token.expand(B, -1, -1)
        x      = torch.cat([cls, x], dim=1)

        pad_mask = torch.zeros(B, T+1, dtype=torch.bool,
                               device=seq.device)
        for i in range(B):
            actual = seq_len[i].item()
            if actual < T:
                pad_mask[i, actual+1:] = True

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.norm(x)[:, 0, :]   # CLS token


# ─────────────────────────────────────────────
# 4.  GNN Encoder
# ─────────────────────────────────────────────

class GNNEncoder(nn.Module):
    def __init__(self, embedding, gnn_hidden=128,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        embed_dim      = embedding.embedding.embedding_dim
        self.convs     = nn.ModuleList()
        self.bns       = nn.ModuleList()
        in_dim         = embed_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_dim, gnn_hidden))
            self.bns.append(nn.BatchNorm1d(gnn_hidden))
            in_dim = gnn_hidden
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch):
        x          = self.embedding(graph_batch.x)
        edge_index = graph_batch.edge_index
        batch      = graph_batch.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.dropout(F.relu(bn(conv(x, edge_index))))
        return global_mean_pool(x, batch)


# ─────────────────────────────────────────────
# 5.  Fusion
# ─────────────────────────────────────────────

class FusionLayer(nn.Module):
    def __init__(self, t_dim, g_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(t_dim + g_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t, z_g):
        return self.fc(torch.cat([z_t, z_g], dim=-1))


# ─────────────────────────────────────────────
# 6.  RealNVP Normalizing Flow
# ─────────────────────────────────────────────

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        half          = input_dim // 2
        self.net_s    = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - half), nn.Tanh(),
        )
        self.net_t    = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - half),
        )
        self.half = half

    def forward(self, x):
        x1, x2  = x[:, :self.half], x[:, self.half:]
        s, t     = self.net_s(x1), self.net_t(x1)
        z2       = x2 * torch.exp(s) + t
        return torch.cat([x1, z2], dim=-1), s.sum(dim=-1)

    def inverse(self, z):
        z1, z2 = z[:, :self.half], z[:, self.half:]
        s, t   = self.net_s(z1), self.net_t(z1)
        x2     = (z2 - t) * torch.exp(-s)
        return torch.cat([z1, x2], dim=-1)


class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        perms = []
        for i in range(num_layers):
            perm = torch.arange(input_dim)
            if i % 2 == 1:
                perm = torch.flip(perm, dims=[0])
            perms.append(perm)
        self.register_buffer('perms', torch.stack(perms))

    def log_prob(self, x):
        log_det = torch.zeros(x.size(0), device=x.device)
        z       = x
        for i, layer in enumerate(self.layers):
            z = z[:, self.perms[i]]
            z, ld = layer(z)
            log_det += ld
        log_pz = -0.5 * (z**2 + math.log(2*math.pi)).sum(dim=-1)
        return log_pz + log_det

    def anomaly_score(self, x):
        return -self.log_prob(x)


# ─────────────────────────────────────────────
# 7.  SENTINELv3 — Full Model
# ─────────────────────────────────────────────

class SENTINELv3(nn.Module):
    """
    SENTINELv3: Contrastive Augmentation + Flow Scoring

    Training:
      - Flow loss      : maximize log p(z_normal)
      - Contrastive    : push z_normal and z_aug apart
      - Reconstruction : next-event prediction (anti-collapse)

    Inference:
      - Anomaly score = -log p(z)  (flow log-likelihood)
    """
    def __init__(
        self,
        vocab_size:         int,
        embed_dim:          int   = 128,
        pad_idx:            int   = 0,
        transformer_layers: int   = 2,
        num_heads:          int   = 4,
        ffn_dim:            int   = 256,
        gnn_hidden:         int   = 128,
        gnn_layers:         int   = 2,
        latent_dim:         int   = 256,
        flow_layers:        int   = 6,
        flow_hidden:        int   = 512,
        dropout:            float = 0.1,
        max_len:            int   = 512,
        margin:             float = 10.0,   # contrastive margin
        lambda_contrast:    float = 0.5,    # contrastive loss weight
        lambda_recon:       float = 0.1,    # reconstruction loss weight
    ):
        super().__init__()
        self.pad_idx         = pad_idx
        self.margin          = margin
        self.lambda_contrast = lambda_contrast
        self.lambda_recon    = lambda_recon

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

        # Fusion + normalization
        self.fusion      = FusionLayer(embed_dim, gnn_hidden,
                                        latent_dim, dropout)
        self.latent_norm = nn.LayerNorm(latent_dim)

        # Normalizing flow
        self.flow = RealNVP(latent_dim, flow_hidden, flow_layers)

        # Reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size),
        )

    def encode(self, seq, seq_len, graph):
        """Encode a session to latent vector z."""
        z_t = self.transformer(seq, seq_len)
        z_g = self.gnn(graph)
        z   = self.fusion(z_t, z_g)
        return self.latent_norm(z)

    def forward(self, batch):
        """
        Contrastive training forward pass.
        batch contains: seq_norm, seq_aug, len_norm, len_aug,
                        graph_norm, graph_aug
        Returns: (total_loss, flow_loss, contrast_loss, recon_loss)
        """
        seq_n  = batch['seq_norm']
        seq_a  = batch['seq_aug']
        len_n  = batch['len_norm']
        len_a  = batch['len_aug']
        graph_n = batch['graph_norm']
        graph_a = batch['graph_aug']

        # Encode both normal and augmented sessions
        z_n = self.encode(seq_n, len_n, graph_n)   # normal
        z_a = self.encode(seq_a, len_a, graph_a)   # augmented

        # ── 1. Flow Loss (on normal sessions only) ──
        flow_loss = -self.flow.log_prob(z_n).mean()

        # ── 2. Contrastive Loss ──
        # Push normal and augmented embeddings apart
        # Margin loss: penalize if ||z_n - z_a|| < margin
        dist          = torch.norm(z_n - z_a, dim=-1)   # [B]
        contrast_loss = F.relu(self.margin - dist).mean()

        # ── 3. Reconstruction Loss (anti-collapse) ──
        emb    = self.embedding(seq_n)              # [B, T, D]
        logits = self.recon_head(emb[:, :-1, :])   # [B, T-1, V]
        target = seq_n[:, 1:]                        # [B, T-1]
        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.pad_idx,
        )

        # ── Total Loss ──
        total = (flow_loss
                 + self.lambda_contrast * contrast_loss
                 + self.lambda_recon    * recon_loss)

        return total, flow_loss, contrast_loss, recon_loss

    def score(self, seq, seq_len, graph):
        """Inference: anomaly score = -log p(z)."""
        with torch.no_grad():
            z = self.encode(seq, seq_len, graph)
            return self.flow.anomaly_score(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)
