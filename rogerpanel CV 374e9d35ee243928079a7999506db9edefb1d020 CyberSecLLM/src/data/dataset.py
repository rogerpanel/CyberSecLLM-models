"""
Dataset classes for loading and processing the six security datasets
in the IIS3D and ICS3D collections.

Reference: Section IV (Datasets and Evaluation Benchmark) of the paper.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .preprocessing import FeatureHarmonizer


class SecurityDataset(Dataset):
    """
    PyTorch Dataset for a single security dataset with harmonized features.

    Handles loading, preprocessing, and batching of network flow records
    for the CyberSecLLM pre-training and evaluation pipelines.
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        harmonizer: Optional[FeatureHarmonizer] = None,
        split: str = "train",
        max_seq_len: int = 512,
        transform: Optional[callable] = None,
    ):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.transform = transform

        if harmonizer is None:
            harmonizer = FeatureHarmonizer()
        self.harmonizer = harmonizer

        self.features = None
        self.labels = None
        self.timestamps = None
        self._load_data()

    def _load_data(self):
        """Load and preprocess the dataset."""
        if os.path.exists(self.data_path):
            if self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path, low_memory=False)
            elif self.data_path.endswith(".parquet"):
                df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_csv(self.data_path, low_memory=False)

            fit = self.split == "train"
            harmonized, labels = self.harmonizer.harmonize(
                df, self.dataset_name, fit=fit
            )

            splits = self.harmonizer.split_temporal(
                harmonized, labels if labels is not None else np.zeros(len(df))
            )
            split_data, split_labels = splits[self.split]

            self.features = torch.tensor(
                split_data.values.astype(np.float32), dtype=torch.float32
            )
            self.labels = torch.tensor(split_labels, dtype=torch.long)

            if "timestamp" in harmonized.columns:
                ts_col = splits[self.split][0]["timestamp"].values
                self.timestamps = torch.tensor(
                    ts_col.astype(np.float64), dtype=torch.float32
                )
            else:
                self.timestamps = torch.arange(
                    len(self.features), dtype=torch.float32
                )
        else:
            self._generate_synthetic()

    def _generate_synthetic(self):
        """Generate synthetic data for development and testing."""
        n_samples = {"train": 10000, "val": 2000, "test": 2000}[self.split]
        n_features = 80
        n_classes = 10

        self.features = torch.randn(n_samples, n_features)
        self.labels = torch.randint(0, n_classes, (n_samples,))
        self.timestamps = torch.linspace(0, 1, n_samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "label": self.labels[idx],
            "timestamp": self.timestamps[idx],
        }
        if self.transform:
            item = self.transform(item)
        return item


class IntegratedSecurityCorpus(Dataset):
    """
    Combined corpus from all six datasets for pre-training.

    Merges IIS3D (CIC-IDS2018, CIC-IoT-2023, UNSW-NB15) and
    ICS3D (Azure Cloud, Edge-IIoTset, Kubernetes/Docker) collections
    with proportional sampling.
    """

    DATASET_CONFIGS = {
        "CIC-IDS2018": {
            "collection": "IIS3D",
            "records": 16233002,
            "features": 80,
            "attack_types": 7,
        },
        "CIC-IoT-2023": {
            "collection": "IIS3D",
            "records": 46686579,
            "features": 46,
            "attack_types": 33,
        },
        "UNSW-NB15": {
            "collection": "IIS3D",
            "records": 2540044,
            "features": 49,
            "attack_types": 9,
        },
        "Azure-Cloud": {
            "collection": "ICS3D",
            "features": 40,
            "attack_types": 8,
        },
        "Edge-IIoTset": {
            "collection": "ICS3D",
            "records": 20761991,
            "features": 61,
            "attack_types": 14,
        },
        "Kubernetes-Docker": {
            "collection": "ICS3D",
            "features": 35,
            "attack_types": 12,
        },
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_seq_len: int = 512,
        proportional_sampling: bool = True,
    ):
        self.data_root = data_root
        self.split = split
        self.harmonizer = FeatureHarmonizer()
        self.datasets = {}

        for name, config in self.DATASET_CONFIGS.items():
            collection = config["collection"]
            path = os.path.join(data_root, collection, name.replace("-", "_"))
            csv_path = os.path.join(path, f"{split}.csv")

            if not os.path.exists(csv_path):
                csv_path = path

            self.datasets[name] = SecurityDataset(
                data_path=csv_path,
                dataset_name=name,
                harmonizer=self.harmonizer,
                split=split,
                max_seq_len=max_seq_len,
            )

        all_features = []
        all_labels = []
        all_timestamps = []
        all_dataset_ids = []

        for i, (name, ds) in enumerate(self.datasets.items()):
            all_features.append(ds.features)
            all_labels.append(ds.labels)
            all_timestamps.append(ds.timestamps)
            all_dataset_ids.append(
                torch.full((len(ds),), i, dtype=torch.long)
            )

        self.features = torch.cat(all_features, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        self.timestamps = torch.cat(all_timestamps, dim=0)
        self.dataset_ids = torch.cat(all_dataset_ids, dim=0)

        if proportional_sampling:
            self._compute_sampling_weights()

    def _compute_sampling_weights(self):
        """Compute sampling weights proportional to dataset sizes."""
        counts = torch.bincount(self.dataset_ids)
        total = counts.sum().float()
        weights = total / (counts.float() * len(counts))
        self.sample_weights = weights[self.dataset_ids]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "timestamp": self.timestamps[idx],
            "dataset_id": self.dataset_ids[idx],
        }

    def get_dataloader(
        self, batch_size: int = 256, num_workers: int = 4, shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader with proportional sampling."""
        if shuffle and hasattr(self, "sample_weights"):
            sampler = torch.utils.data.WeightedRandomSampler(
                self.sample_weights, len(self), replacement=True
            )
            return DataLoader(
                self,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
