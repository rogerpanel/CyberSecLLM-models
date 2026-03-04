"""
Evaluation script for CyberSecLLM on CyberSecBench.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --variant 7B
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --task cic_iot_multiclass
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.cybersecllm import CyberSecLLM
from src.data.dataset import SecurityDataset
from src.evaluation.benchmark import CyberSecBench
from src.evaluation.metrics import SecurityMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, variant="7B", device="cuda"):
    """Load trained model from checkpoint."""
    model = CyberSecLLM.from_variant(variant)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model


def setup_test_loaders(data_root="data/", batch_size=256):
    """Set up test data loaders for each CyberSecBench task."""
    loaders = {}
    datasets_config = {
        "cic_iot_multiclass": ("IIS3D/CIC-IoT-2023", "CIC-IoT-2023"),
        "cross_dataset_transfer": ("IIS3D/UNSW-NB15", "UNSW-NB15"),
        "ood_detection": ("IIS3D/CIC-IDS2018", "CIC-IDS2018"),
    }

    for task_name, (path, ds_name) in datasets_config.items():
        full_path = os.path.join(data_root, path)
        ds = SecurityDataset(
            data_path=full_path,
            dataset_name=ds_name,
            split="test",
        )
        loaders[task_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4
        )

    return loaders


def main():
    parser = argparse.ArgumentParser(description="Evaluate CyberSecLLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default="7B")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--output", type=str, default="experiments/results/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info(f"Loading CyberSecLLM-{args.variant} from {args.checkpoint}")
    model = load_model(args.checkpoint, args.variant, args.device)

    logger.info("Setting up CyberSecBench evaluation suite...")
    benchmark = CyberSecBench(model, device=args.device)

    logger.info("Loading test datasets...")
    test_loaders = setup_test_loaders(args.data_root, args.batch_size)

    if args.task != "all":
        test_loaders = {k: v for k, v in test_loaders.items() if k == args.task}

    logger.info(f"Running evaluation on {len(test_loaders)} tasks...")
    results = benchmark.run_full_benchmark(test_loaders)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "cybersecbench_results.json")

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                for sk, sv in v.items()
                if not isinstance(sv, (list, np.ndarray))
            }
        else:
            serializable[k] = float(v) if isinstance(v, (np.floating, float)) else v

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("CyberSecBench Results Summary")
    logger.info("=" * 60)
    for task, task_results in results.items():
        if isinstance(task_results, dict):
            acc = task_results.get("accuracy", "N/A")
            f1 = task_results.get("f1_score", "N/A")
            logger.info(f"  {task}: Acc={acc:.4f}, F1={f1:.4f}")
        else:
            logger.info(f"  {task}: {task_results:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
