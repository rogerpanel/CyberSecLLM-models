"""
Data preprocessing script for CyberSecLLM.

Applies the three-stage feature harmonization pipeline to all six
datasets and saves the processed data in a unified format.

Usage:
    python scripts/preprocess_data.py --input_dir data/ --output_dir data/processed/
"""

import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import FeatureHarmonizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_FILES = {
    "CIC-IDS2018": {
        "collection": "IIS3D",
        "files": ["*.csv"],
    },
    "CIC-IoT-2023": {
        "collection": "IIS3D",
        "files": ["*.csv"],
    },
    "UNSW-NB15": {
        "collection": "IIS3D",
        "files": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
    },
    "Azure-Cloud": {
        "collection": "ICS3D",
        "files": ["*.csv"],
    },
    "Edge-IIoTset": {
        "collection": "ICS3D",
        "files": ["*.csv"],
    },
    "Kubernetes-Docker": {
        "collection": "ICS3D",
        "files": ["*.csv"],
    },
}


def find_csv_files(directory):
    """Find all CSV files in a directory."""
    csv_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


def process_dataset(
    input_dir, output_dir, dataset_name, harmonizer, collection
):
    """Process a single dataset through the harmonization pipeline."""
    dataset_dir = os.path.join(input_dir, collection, dataset_name.replace("-", "_"))

    if not os.path.exists(dataset_dir):
        logger.warning(f"Directory not found: {dataset_dir}. Skipping.")
        return

    csv_files = find_csv_files(dataset_dir)
    if not csv_files:
        logger.warning(f"No CSV files found in {dataset_dir}. Skipping.")
        return

    logger.info(f"Processing {dataset_name}: {len(csv_files)} files")

    chunks = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False, nrows=1000000)
            chunks.append(df)
            logger.info(f"  Loaded {csv_file}: {len(df)} records")
        except Exception as e:
            logger.error(f"  Error loading {csv_file}: {e}")

    if not chunks:
        return

    combined = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Combined: {len(combined)} records, {len(combined.columns)} features")

    harmonized, labels = harmonizer.harmonize(combined, dataset_name)

    out_dir = os.path.join(output_dir, collection, dataset_name.replace("-", "_"))
    os.makedirs(out_dir, exist_ok=True)

    if labels is not None:
        splits = harmonizer.split_temporal(harmonized, labels)
        for split_name, (split_df, split_labels) in splits.items():
            split_df["label"] = split_labels
            out_path = os.path.join(out_dir, f"{split_name}.csv")
            split_df.to_csv(out_path, index=False)
            logger.info(f"  Saved {split_name}: {len(split_df)} records -> {out_path}")
    else:
        out_path = os.path.join(out_dir, "processed.csv")
        harmonized.to_csv(out_path, index=False)
        logger.info(f"  Saved: {len(harmonized)} records -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    args = parser.parse_args()

    harmonizer = FeatureHarmonizer(n_harmonized_features=80)

    for dataset_name, config in DATASET_FILES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name} ({config['collection']})")
        logger.info(f"{'='*60}")

        process_dataset(
            args.input_dir,
            args.output_dir,
            dataset_name,
            harmonizer,
            config["collection"],
        )

    logger.info("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
