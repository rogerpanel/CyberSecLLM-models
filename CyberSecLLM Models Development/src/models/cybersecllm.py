"""
CyberSecLLM: Hybrid Mamba-Transformer Foundation Model
for Zero-Shot Cybersecurity Threat Intelligence
with Continuous-Time Adaptive Detection.

This module integrates all architectural components:
1. Network Flow Tokenizer (parallel encoding pathways)
2. Mamba SSM Blocks (linear-time long-range processing)
3. Cross-Attention Transformer Blocks (knowledge-grounded reasoning)
4. Mixture-of-Experts Feedforward (task-specialized routing)
5. TA-BN-ODE Module (continuous-time adaptive detection)
6. Deep Spatio-Temporal Point Process (temporal event modeling)

The architecture stacks these blocks in a repeating pattern:
[Mamba x 3 -> CrossAttn-Transformer + MoE] x N_groups

Reference: Section III of the paper (Fig. 1 architecture diagram).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import MambaBlock
from .transformer_block import CrossAttentionTransformerBlock
from .moe import MixtureOfExperts
from .tabn_ode import TABN_ODE
from .point_process import DeepSpatioTemporalPointProcess


class CyberSecLLMConfig:
    """Configuration for CyberSecLLM model variants."""

    VARIANTS = {
        "3B": dict(
            hidden_dim=2048, mamba_layers=16, transformer_layers=4,
            num_heads=16, moe_experts=8, active_experts=2,
            d_state=16, context_length=524288,
        ),
        "7B": dict(
            hidden_dim=3072, mamba_layers=24, transformer_layers=6,
            num_heads=24, moe_experts=8, active_experts=2,
            d_state=16, context_length=1048576,
        ),
        "13B": dict(
            hidden_dim=4096, mamba_layers=32, transformer_layers=8,
            num_heads=32, moe_experts=16, active_experts=4,
            d_state=16, context_length=1048576,
        ),
    }

    def __init__(self, variant="7B", **kwargs):
        config = self.VARIANTS[variant].copy()
        config.update(kwargs)
        for k, v in config.items():
            setattr(self, k, v)

        self.mamba_per_group = self.mamba_layers // self.transformer_layers
        self.num_groups = self.transformer_layers
        self.total_layers = self.mamba_layers + self.transformer_layers

        self.input_dim = kwargs.get("input_dim", 80)
        self.num_marks = kwargs.get("num_marks", 50)
        self.num_classes = kwargs.get("num_classes", 50)
        self.dropout = kwargs.get("dropout", 0.1)
        self.moe_hidden = kwargs.get("moe_hidden", 4 * self.hidden_dim)
        self.shared_experts = kwargs.get("shared_experts", 1)
        self.tabn_hidden = kwargs.get("tabn_hidden", 256)
        self.tabn_blocks = kwargs.get("tabn_blocks", 2)
        self.time_constants = kwargs.get(
            "time_constants", [1e-6, 1e-3, 1.0, 3600.0]
        )
        self.pp_heads = kwargs.get("pp_heads", 8)
        self.pp_layers = kwargs.get("pp_layers", 4)


class HybridBlock(nn.Module):
    """
    Single hybrid block: multiple Mamba layers followed by
    one cross-attention transformer layer with MoE feedforward.
    """

    def __init__(self, config, mamba_count):
        super().__init__()

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=config.hidden_dim,
                d_state=config.d_state,
                expand_factor=2,
                conv_kernel=4,
                dropout=config.dropout,
            )
            for _ in range(mamba_count)
        ])

        self.transformer = CrossAttentionTransformerBlock(
            d_model=config.hidden_dim,
            num_heads=config.num_heads,
            ff_hidden_dim=config.moe_hidden,
            dropout=config.dropout,
            use_cross_attention=True,
        )

        self.moe = MixtureOfExperts(
            d_model=config.hidden_dim,
            num_experts=config.moe_experts,
            top_k=config.active_experts,
            expert_hidden_dim=config.moe_hidden,
            num_shared_experts=config.shared_experts,
            dropout=config.dropout,
        )

    def forward(self, x, knowledge_base=None, kb_mask=None):
        for mamba in self.mamba_layers:
            x = mamba(x)
        x = self.transformer(x, knowledge_base=knowledge_base, kb_mask=kb_mask)
        x = self.moe(x)
        return x

    @property
    def load_balance_loss(self):
        return self.moe.load_balance_loss


class TaskHead(nn.Module):
    """Task-specific output head for multi-task prediction."""

    def __init__(self, hidden_dim, output_dim, task_name="classification"):
        super().__init__()
        self.task_name = task_name
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.head(x)


class CyberSecLLM(nn.Module):
    """
    Complete CyberSecLLM architecture.

    Integrates the hybrid Mamba-Transformer backbone with
    TA-BN-ODE continuous-time modeling and Deep Spatio-Temporal
    Point Processes for comprehensive cybersecurity threat
    intelligence.

    Architecture flow:
    Input -> Tokenizer -> [Mamba x M -> CrossAttn + MoE] x N
          -> TA-BN-ODE -> Point Process -> Task Heads
    """

    def __init__(self, config=None, variant="7B", **kwargs):
        super().__init__()

        if config is None:
            config = CyberSecLLMConfig(variant, **kwargs)
        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

        self.hybrid_blocks = nn.ModuleList([
            HybridBlock(config, config.mamba_per_group)
            for _ in range(config.num_groups)
        ])

        self.tabn_ode = TABN_ODE(
            input_dim=config.hidden_dim,
            hidden_dim=config.tabn_hidden,
            num_blocks=config.tabn_blocks,
            time_constants=config.time_constants,
            use_adjoint=True,
        )

        self.point_process = DeepSpatioTemporalPointProcess(
            hidden_dim=config.tabn_hidden,
            num_marks=config.num_marks,
            num_heads=config.pp_heads,
            num_layers=config.pp_layers,
        )

        self.state_projector = nn.Linear(
            config.tabn_hidden, config.hidden_dim
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )

        self.detection_head = TaskHead(
            config.hidden_dim, config.num_classes, "detection"
        )
        self.triage_head = TaskHead(
            config.hidden_dim, 5, "triage"
        )
        self.attribution_head = TaskHead(
            config.hidden_dim, 193, "attribution"
        )

        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        timestamps: torch.Tensor,
        marks: Optional[torch.Tensor] = None,
        knowledge_base: Optional[torch.Tensor] = None,
        kb_mask: Optional[torch.Tensor] = None,
        task: str = "detection",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete CyberSecLLM architecture.

        Args:
            features: (B, L, D) input feature sequences
            timestamps: (B, L) event timestamps
            marks: (B, L) event mark labels (optional)
            knowledge_base: (B, M, D_kb) threat intelligence embeddings
            kb_mask: (B, M) knowledge base padding mask
            task: output task ("detection", "triage", "attribution")

        Returns:
            Dictionary containing logits, intensity, and auxiliary losses.
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        x = self.input_projection(features)

        total_balance_loss = torch.tensor(0.0, device=features.device)
        for block in self.hybrid_blocks:
            x = block(x, knowledge_base=knowledge_base, kb_mask=kb_mask)
            if block.load_balance_loss is not None:
                total_balance_loss = total_balance_loss + block.load_balance_loss

        backbone_output = x

        ode_states = self.tabn_ode(features, timestamps)
        ode_projected = self.state_projector(ode_states)

        min_len = min(backbone_output.shape[1], ode_projected.shape[1])
        backbone_trimmed = backbone_output[:, :min_len]
        ode_trimmed = ode_projected[:, :min_len]

        gate = self.fusion_gate(
            torch.cat([backbone_trimmed, ode_trimmed], dim=-1)
        )
        fused = gate * backbone_trimmed + (1 - gate) * ode_trimmed

        fused = self.final_norm(fused)

        pp_output = self.point_process(
            ode_states, timestamps, marks
        )

        if task == "detection":
            logits = self.detection_head(fused)
        elif task == "triage":
            logits = self.triage_head(fused)
        elif task == "attribution":
            logits = self.attribution_head(fused)
        else:
            logits = self.detection_head(fused)

        stability_reg = self.tabn_ode.compute_stability_regularization()

        return {
            "logits": logits,
            "intensity": pp_output["intensity"],
            "mark_logits": pp_output["mark_logits"],
            "backbone_features": backbone_output,
            "ode_states": ode_states,
            "fused_features": fused,
            "load_balance_loss": total_balance_loss,
            "stability_reg": stability_reg,
        }

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_variant(cls, variant="7B", **kwargs):
        """Create model from a named variant (3B, 7B, or 13B)."""
        config = CyberSecLLMConfig(variant, **kwargs)
        return cls(config=config)
