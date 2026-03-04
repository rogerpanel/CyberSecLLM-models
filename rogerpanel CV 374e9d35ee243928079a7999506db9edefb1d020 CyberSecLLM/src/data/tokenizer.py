"""
Network Flow Tokenizer for CyberSecLLM.

Converts structured network flow records into unified embedding
representations through parallel encoding pathways for continuous,
categorical, temporal, and topological features.

Reference: Section III-A (Network Flow Tokenization) of the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousFeatureEncoder(nn.Module):
    """Two-layer MLP encoder for continuous numerical features (Eq. 1)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CategoricalFeatureEncoder(nn.Module):
    """Learned embedding tables for categorical features (Eq. 2)."""

    def __init__(self, vocab_sizes, embedding_dims):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.total_dim = 0
        for name, (vocab, dim) in zip(
            vocab_sizes.keys(),
            zip(vocab_sizes.values(), embedding_dims.values()),
        ):
            self.embeddings[name] = nn.Embedding(vocab, dim)
            self.total_dim += dim

    def forward(self, categorical_features):
        embedded = []
        for name, emb in self.embeddings.items():
            if name in categorical_features:
                embedded.append(emb(categorical_features[name]))
        return torch.cat(embedded, dim=-1)


class TemporalEncoder(nn.Module):
    """Continuous-time positional encoding with learnable frequencies (Eq. 3)."""

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        half_dim = output_dim // 2
        self.omega = nn.Parameter(torch.randn(half_dim) * 0.01)
        self.phi = nn.Parameter(torch.zeros(half_dim))

    def forward(self, timestamps):
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        phase = timestamps * self.omega.unsqueeze(0) + self.phi.unsqueeze(0)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TopologicalEncoder(nn.Module):
    """Edge embedding aggregating one-hop neighbor information (Eq. 4)."""

    def __init__(self, edge_dim, node_dim, output_dim):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim + node_dim, output_dim)

    def forward(self, edge_features, neighbor_embeddings, neighbor_mask=None):
        combined = torch.cat([edge_features, neighbor_embeddings], dim=-1)
        projected = self.edge_proj(combined)
        if neighbor_mask is not None:
            projected = projected * neighbor_mask.unsqueeze(-1)
            counts = neighbor_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            return projected.sum(dim=-2) / counts
        return projected.mean(dim=-2)


class NetworkFlowTokenizer(nn.Module):
    """
    Complete network flow tokenizer combining four parallel encoding
    pathways into unified token representations (Eq. 5).

    Given a network flow record r_i containing continuous features c_i,
    categorical features k_i, timestamp t_i, and topological context,
    produces a unified embedding x_i in R^d.
    """

    def __init__(
        self,
        continuous_dim=80,
        hidden_dim=256,
        embedding_dim=3072,
        vocab_sizes=None,
        temporal_dim=768,
        edge_dim=32,
        node_dim=64,
    ):
        super().__init__()

        if vocab_sizes is None:
            vocab_sizes = {"protocol": 256, "port": 65536, "flag": 64}

        quarter_dim = embedding_dim // 4
        embedding_dims = {k: quarter_dim // len(vocab_sizes) for k in vocab_sizes}
        cat_total = sum(embedding_dims.values())

        self.continuous_encoder = ContinuousFeatureEncoder(
            continuous_dim, hidden_dim, quarter_dim
        )
        self.categorical_encoder = CategoricalFeatureEncoder(
            vocab_sizes, embedding_dims
        )
        self.temporal_encoder = TemporalEncoder(quarter_dim)
        self.topological_encoder = TopologicalEncoder(
            edge_dim, node_dim, quarter_dim
        )

        concat_dim = quarter_dim * 3 + cat_total
        self.projection = nn.Linear(concat_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        continuous_features,
        categorical_features,
        timestamps,
        edge_features=None,
        neighbor_embeddings=None,
        neighbor_mask=None,
    ):
        f_cont = self.continuous_encoder(continuous_features)
        f_cat = self.categorical_encoder(categorical_features)
        f_time = self.temporal_encoder(timestamps)

        if edge_features is not None and neighbor_embeddings is not None:
            f_topo = self.topological_encoder(
                edge_features, neighbor_embeddings, neighbor_mask
            )
        else:
            f_topo = torch.zeros_like(f_cont)

        concatenated = torch.cat([f_cont, f_cat, f_time, f_topo], dim=-1)
        x = self.projection(concatenated)
        x = self.layer_norm(x)
        return x
