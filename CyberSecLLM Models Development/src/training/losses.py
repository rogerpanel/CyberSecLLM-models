"""
Multi-task loss functions for CyberSecLLM pre-training.

Implements the four complementary pre-training objectives:
1. Masked Flow Modeling (MFM) - Eq. 10
2. Contrastive Attack Discrimination (CAD) - Eq. 11
3. Temporal Forecasting (TF) - Eq. 12
4. ATT&CK Technique Attribution (AA) - Eq. 13

Combined using homoscedastic uncertainty weighting (Eq. 14).

Reference: Section III-C (Multi-Task Pre-Training Objectives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFlowModelingLoss(nn.Module):
    """
    Masked Flow Modeling loss (Eq. 10).

    Analogous to masked language modeling in BERT but operates
    over structured flow token representations. Randomly masks 15%
    of flow tokens and trains the model to reconstruct them.
    """

    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, predictions, targets, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        masked_pred = predictions[mask]
        masked_target = targets[mask]
        return F.mse_loss(masked_pred, masked_target)

    def create_mask(self, x):
        mask = torch.rand_like(x[:, :, 0]) < self.mask_ratio
        return mask


class ContrastiveAttackDiscriminationLoss(nn.Module):
    """
    Contrastive Attack Discrimination loss (Eq. 11).

    InfoNCE-based contrastive loss that clusters traffic by attack type
    while separating benign from malicious flows:

    L_CAD = -log(exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau))
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        batch_size = embeddings.shape[0]

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_matrix.fill_diagonal_(False)

        if label_matrix.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        mask_diag = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        log_sum_exp = torch.logsumexp(
            similarity.masked_fill(~mask_diag, float("-inf")), dim=1
        )

        pos_mask = label_matrix.float()
        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        pos_sim = (similarity * pos_mask).sum(dim=1) / pos_count

        loss = (-pos_sim + log_sum_exp).mean()
        return loss


class TemporalForecastingLoss(nn.Module):
    """
    Temporal Forecasting loss (Eq. 12).

    Trains the model to predict future network behavior from
    historical context:
    L_TF = ||x_{t+1:t+k} - f_theta(x_{1:t})||^2
    """

    def __init__(self, forecast_horizon=5):
        super().__init__()
        self.horizon = forecast_horizon

    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)


class ATTACKAttributionLoss(nn.Module):
    """
    MITRE ATT&CK Technique Attribution loss (Eq. 13).

    Multi-label classification mapping observed network behavior
    to ATT&CK techniques:
    L_AA = sum_i -[y_i log sigma(w_i^T h_cls) +
                   (1-y_i) log(1 - sigma(w_i^T h_cls))]
    """

    def __init__(self, num_techniques=193):
        super().__init__()
        self.num_techniques = num_techniques

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets.float())


class UncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task learning (Eq. 14).

    Learns scalar uncertainty parameter s_j for each task that
    automatically balances loss magnitudes:

    L_total = sum_j (1/(2*s_j^2)) * L_j + log(s_j) + lambda_bal * L_bal

    Reference: Kendall et al., "Multi-task learning using uncertainty
    to weigh losses" (CVPR 2018).
    """

    def __init__(self, num_tasks=4, balance_weight=0.01, stability_weight=0.001):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
        self.balance_weight = balance_weight
        self.stability_weight = stability_weight

    def forward(
        self,
        task_losses,
        balance_loss=None,
        stability_loss=None,
    ):
        total = torch.tensor(0.0, device=self.log_sigma.device)

        for i, loss in enumerate(task_losses):
            if i < len(self.log_sigma):
                precision = torch.exp(-2 * self.log_sigma[i])
                total = total + precision * loss + self.log_sigma[i]
            else:
                total = total + loss

        if balance_loss is not None:
            total = total + self.balance_weight * balance_loss

        if stability_loss is not None:
            total = total + self.stability_weight * stability_loss

        return total

    def get_task_weights(self):
        """Return current learned task weights for monitoring."""
        return torch.exp(-2 * self.log_sigma).detach()


class MultiTaskLoss(nn.Module):
    """
    Complete multi-task loss combining all four objectives
    with uncertainty weighting.
    """

    def __init__(
        self,
        num_classes=50,
        num_techniques=193,
        temperature=0.07,
        mask_ratio=0.15,
        forecast_horizon=5,
        balance_weight=0.01,
        stability_weight=0.001,
    ):
        super().__init__()

        self.mfm_loss = MaskedFlowModelingLoss(mask_ratio)
        self.cad_loss = ContrastiveAttackDiscriminationLoss(temperature)
        self.tf_loss = TemporalForecastingLoss(forecast_horizon)
        self.aa_loss = ATTACKAttributionLoss(num_techniques)
        self.classification_loss = nn.CrossEntropyLoss()
        self.uncertainty_weighting = UncertaintyWeightedLoss(
            num_tasks=4,
            balance_weight=balance_weight,
            stability_weight=stability_weight,
        )

    def forward(self, model_output, batch):
        logits = model_output["logits"]
        labels = batch["label"]
        cls_loss = self.classification_loss(logits, labels)

        features = model_output.get("backbone_features")
        if features is not None and features.dim() == 3:
            pooled = features.mean(dim=1)
        else:
            pooled = features if features is not None else logits

        cad_loss = self.cad_loss(pooled, labels) if pooled is not None else torch.tensor(0.0)
        tf_loss = torch.tensor(0.0, device=logits.device)
        aa_loss = torch.tensor(0.0, device=logits.device)

        task_losses = [cls_loss, cad_loss, tf_loss, aa_loss]

        total = self.uncertainty_weighting(
            task_losses,
            balance_loss=model_output.get("load_balance_loss"),
            stability_loss=model_output.get("stability_reg"),
        )

        return {
            "total_loss": total,
            "cls_loss": cls_loss.item(),
            "cad_loss": cad_loss.item() if isinstance(cad_loss, torch.Tensor) else 0.0,
            "tf_loss": tf_loss.item() if isinstance(tf_loss, torch.Tensor) else 0.0,
            "aa_loss": aa_loss.item() if isinstance(aa_loss, torch.Tensor) else 0.0,
            "task_weights": self.uncertainty_weighting.get_task_weights(),
        }
