"""
Evaluation metrics for CyberSecLLM.

Implements comprehensive security-specific metrics including
accuracy, F1-score, AUROC, calibration metrics (ECE, Brier score),
and operational metrics (nDCG for alert triage).

Reference: Section V (Experimental Evaluation).
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)


class SecurityMetrics:
    """Comprehensive evaluation metrics for security models."""

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "macro",
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "f1_score": f1_score(
                y_true, y_pred, average=average, zero_division=0
            ),
        }

        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    metrics["auroc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=average
                    )
                elif y_prob.ndim == 1 or y_prob.shape[1] == 2:
                    prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                    metrics["auroc"] = roc_auc_score(y_true, prob)
            except ValueError:
                metrics["auroc"] = 0.0

        return metrics

    @staticmethod
    def compute_calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute calibration metrics (ECE, Brier score).

        Expected Calibration Error measures the gap between
        predicted confidence and empirical accuracy.
        """
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        reliability = []

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * abs(bin_acc - bin_conf)
                reliability.append({
                    "bin_center": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                    "count": mask.sum(),
                })

        y_true_onehot = np.zeros_like(y_prob)
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        brier = ((y_prob - y_true_onehot) ** 2).sum(axis=1).mean()

        nll = -np.log(
            y_prob[np.arange(len(y_true)), y_true].clip(min=1e-8)
        ).mean()

        return {
            "ece": ece,
            "brier_score": brier,
            "nll": nll,
            "reliability": reliability,
        }

    @staticmethod
    def compute_coverage(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        confidence_level: float = 0.95,
    ) -> float:
        """Compute prediction interval coverage probability."""
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        threshold = np.percentile(confidences, (1 - confidence_level) * 100)
        high_conf_mask = confidences >= threshold
        if high_conf_mask.sum() == 0:
            return 0.0
        return (predictions[high_conf_mask] == y_true[high_conf_mask]).mean()

    @staticmethod
    def compute_ndcg(
        relevance_scores: np.ndarray,
        predicted_ranking: np.ndarray,
        k: int = 10,
    ) -> float:
        """
        Compute normalized discounted cumulative gain at k
        for alert triage ranking.
        """
        ranked_relevance = relevance_scores[predicted_ranking[:k]]
        dcg = np.sum(ranked_relevance / np.log2(np.arange(2, k + 2)))

        ideal_order = np.argsort(relevance_scores)[::-1][:k]
        ideal_relevance = relevance_scores[ideal_order]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def compute_per_attack_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """Compute per-attack-type performance breakdown."""
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)
        return {"classification_report": report, "confusion_matrix": cm}
