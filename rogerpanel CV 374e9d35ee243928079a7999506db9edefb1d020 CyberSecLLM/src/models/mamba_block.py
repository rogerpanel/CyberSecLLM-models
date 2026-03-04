"""
Mamba Selective State Space Model Block for CyberSecLLM.

Implements the Mamba SSM block with input-dependent selection
mechanism for efficient linear-time sequence processing of
long network trace sequences.

Reference: Section III-B (Hybrid Mamba-Transformer Architecture, Eq. 6).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core computation.

    Implements the discretized state space model with input-dependent
    parameters Delta, B, C for selective information propagation:

    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C * h_t

    where A_bar, B_bar are discretized using input-dependent Delta.
    """

    def __init__(self, d_model, d_state=16, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == "auto":
            self.dt_rank = max(d_model // 16, 1)
        else:
            self.dt_rank = dt_rank

        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(d_model, -1)
            )
        )

        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_bias.log())

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        x_dbl = self.x_proj(x)
        delta = self.dt_proj(x_dbl[:, :, :self.dt_rank])
        delta = F.softplus(delta)

        B = x_dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_dbl[:, :, self.dt_rank + self.d_state:]

        A = -torch.exp(self.A_log)
        y = self._selective_scan(x, delta, A, B, C)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    def _selective_scan(self, u, delta, A, B, C):
        """Selective scan implementation (sequential for clarity)."""
        batch, seq_len, d_model = u.shape
        d_state = self.d_state

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch, d_model, d_state, device=u.device)
        outputs = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """
    Complete Mamba block with gated architecture.

    Processes input through:
    1. Linear projection to expanded dimension
    2. 1D causal convolution
    3. Selective SSM
    4. SiLU-gated output (Eq. 6)

    H_l^Mamba = SSM(Conv1D(Linear(H_{l-1}))) * SiLU(Linear(H_{l-1}))
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        expand_factor=2,
        conv_kernel=4,
        dt_rank="auto",
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(self.d_inner, d_state, dt_rank)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_conv = rearrange(x_branch, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.shape[1]]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)

        z = F.silu(z_branch)
        output = y * z

        output = self.out_proj(output)
        output = self.dropout(output)
        return output + residual
