"""
Mixture-of-Experts (MoE) Feedforward Block for CyberSecLLM.

Implements task-specialized expert routing where different experts
specialize in distinct security subtasks (IDS, malware analysis,
APT detection, IoT security), with shared expert isolation.

Reference: Section III-B (Eq. 8-9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network with SwiGLU activation."""

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    """
    Top-K expert routing with load balancing.

    Routes each token to the top-k experts with highest gating scores,
    incorporating auxiliary load-balancing loss to prevent expert collapse.
    """

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        return top_k_weights, top_k_indices, logits

    def compute_load_balance_loss(self, logits):
        """
        Compute auxiliary load-balancing loss (Eq. 9).

        L_bal = N * sum_i(f_i * p_i)

        where f_i is the fraction of tokens routed to expert i
        and p_i is the average gating probability for expert i.
        """
        probs = F.softmax(logits, dim=-1)

        flat_logits = logits.view(-1, self.num_experts)
        flat_probs = probs.view(-1, self.num_experts)

        _, top_indices = flat_logits.topk(self.top_k, dim=-1)

        expert_mask = torch.zeros_like(flat_probs)
        expert_mask.scatter_(1, top_indices, 1.0)

        tokens_per_expert = expert_mask.mean(dim=0)
        avg_prob_per_expert = flat_probs.mean(dim=0)

        loss = self.num_experts * (tokens_per_expert * avg_prob_per_expert).sum()
        return loss


class MixtureOfExperts(nn.Module):
    """
    Complete Mixture-of-Experts block with shared expert.

    Implements task-specialized MoE routing (Eq. 8):
    MoE(x) = sum_{i in TopK(G(x), k)} G(x)_i * E_i(x)

    Following DeepSeekMoE, includes one shared expert always activated
    alongside routed experts to ensure common security knowledge
    is accessible regardless of routing decisions.
    """

    def __init__(
        self,
        d_model,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=None,
        num_shared_experts=1,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = num_shared_experts

        expert_hidden = expert_hidden_dim or 4 * d_model

        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden) for _ in range(num_experts)
        ])

        self.shared_experts = nn.ModuleList([
            Expert(d_model, expert_hidden) for _ in range(num_shared_experts)
        ])

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._load_balance_loss = None

    def forward(self, x):
        residual = x
        x = self.norm(x)

        batch_size, seq_len, d_model = x.shape

        weights, indices, logits = self.router(x)
        self._load_balance_loss = self.router.compute_load_balance_loss(logits)

        flat_x = x.view(-1, d_model)
        flat_weights = weights.view(-1, self.top_k)
        flat_indices = indices.view(-1, self.top_k)

        output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_indices = flat_indices[:, k]
            expert_weights = flat_weights[:, k].unsqueeze(-1)
            for i in range(self.num_experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

        for shared in self.shared_experts:
            output = output + shared(flat_x) / self.num_shared

        output = output.view(batch_size, seq_len, d_model)
        output = self.dropout(output)
        return output + residual

    @property
    def load_balance_loss(self):
        return self._load_balance_loss
