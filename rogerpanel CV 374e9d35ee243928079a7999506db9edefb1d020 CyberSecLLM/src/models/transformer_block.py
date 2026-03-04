"""
Cross-Attention Transformer Block for CyberSecLLM.

Implements the cross-attention mechanism that enables the model
to attend to entries in an external threat intelligence knowledge
base (MITRE ATT&CK, CVE, IOCs).

Reference: Section III-B (Eq. 7 - Cross-Attention).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention for knowledge-grounded reasoning.

    Queries are derived from the Mamba output, while keys and values
    come from the pre-computed threat intelligence embeddings:

    H_l^Attn = softmax(Q_l * K_TI^T / sqrt(d_k)) * V_TI + H_l^Mamba
    """

    def __init__(self, d_model, num_heads, head_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key_value.shape

        q = self.q_proj(query).view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key_value).view(
            batch_size, kv_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(key_value).view(
            batch_size, kv_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, -1
        )
        return self.out_proj(attn_output)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional causal masking."""

    def __init__(self, d_model, num_heads, head_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=False):
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)


class FeedForward(nn.Module):
    """SwiGLU feedforward network."""

    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CrossAttentionTransformerBlock(nn.Module):
    """
    Complete cross-attention transformer block combining:
    1. Self-attention for intra-sequence dependencies
    2. Cross-attention to threat intelligence knowledge base
    3. SwiGLU feedforward network

    This block enables knowledge-grounded reasoning over MITRE ATT&CK
    techniques, CVE descriptions, and structured threat indicators.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        head_dim=None,
        ff_hidden_dim=None,
        dropout=0.1,
        use_cross_attention=True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(
            d_model, num_heads, head_dim, dropout
        )

        if use_cross_attention:
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadCrossAttention(
                d_model, num_heads, head_dim, dropout
            )

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim, dropout)

    def forward(self, x, knowledge_base=None, kb_mask=None, causal_mask=False):
        h = x + self.self_attn(self.self_attn_norm(x), causal_mask=causal_mask)

        if self.use_cross_attention and knowledge_base is not None:
            h = h + self.cross_attn(
                self.cross_attn_norm(h), knowledge_base, mask=kb_mask
            )

        h = h + self.ff(self.ff_norm(h))
        return h
