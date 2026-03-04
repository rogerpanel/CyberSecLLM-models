"""
CyberSecBench: Standardized evaluation suite for security
foundation models spanning eight diverse tasks.

Tasks:
1. CIC-IoT-2023 multi-class intrusion detection
2. Cross-dataset zero-shot transfer (CIC-IoT -> UNSW-NB15)
3. Out-of-distribution detection (held-out attacks from CIC-IDS2018)
4. CVE severity prediction
5. ATT&CK technique classification (multi-label)
6. Malware family attribution
7. Alert triage ranking (nDCG@10)
8. Incident summarization (ROUGE-L)

Reference: Section IV-B (CyberSecBench Evaluation Suite).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import SecurityMetrics

logger = logging.getLogger(__name__)


class CyberSecBench:
    """
    CyberSecBench evaluation suite.

    Evaluates a security foundation model across eight tasks
    organized into three categories: intrusion detection,
    threat intelligence, and operational security.
    """

    TASKS = {
        "intrusion_detection": [
            "cic_iot_multiclass",
            "cross_dataset_transfer",
            "ood_detection",
        ],
        "threat_intelligence": [
            "cve_severity",
            "attack_technique",
            "malware_family",
        ],
        "operational": [
            "alert_triage",
            "incident_summary",
        ],
    }

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.metrics = SecurityMetrics()

    @torch.no_grad()
    def evaluate_classification(
        self, dataloader: DataLoader, task_name: str = "detection"
    ) -> Dict[str, float]:
        """Evaluate on a classification task."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            features = batch["features"]
            timestamps = batch.get(
                "timestamp",
                torch.arange(
                    features.shape[0], device=self.device, dtype=torch.float32
                ),
            )

            if features.dim() == 2:
                features = features.unsqueeze(1)
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(1).expand(
                    -1, features.shape[1]
                )

            output = self.model(
                features=features, timestamps=timestamps, task=task_name
            )
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)

            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)

        cls_metrics = self.metrics.compute_classification_metrics(
            y_true, y_pred, y_prob
        )
        cal_metrics = self.metrics.compute_calibration_metrics(y_true, y_prob)
        coverage = self.metrics.compute_coverage(y_true, y_prob)

        results = {**cls_metrics, **cal_metrics, "coverage_95": coverage}
        logger.info(f"Task: {task_name} | Acc: {cls_metrics['accuracy']:.4f} | "
                     f"F1: {cls_metrics['f1_score']:.4f}")
        return results

    def run_full_benchmark(
        self, dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict]:
        """Run the complete CyberSecBench evaluation."""
        results = {}
        task_scores = []

        for category, tasks in self.TASKS.items():
            for task in tasks:
                if task in dataloaders:
                    logger.info(f"Evaluating task: {task}")
                    task_results = self.evaluate_classification(
                        dataloaders[task], task_name="detection"
                    )
                    results[task] = task_results
                    task_scores.append(task_results.get("accuracy", 0.0))

        if task_scores:
            results["average_accuracy"] = np.mean(task_scores)
            logger.info(f"Average accuracy: {results['average_accuracy']:.4f}")

        return results
