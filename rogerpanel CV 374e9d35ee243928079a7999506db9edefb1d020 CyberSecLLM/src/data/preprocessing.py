"""
Feature Harmonization Pipeline for CyberSecLLM.

Three-stage pipeline for harmonizing features across the six datasets
in the IIS3D and ICS3D collections into a common schema of 80
standardized attributes.

Reference: Section IV-A (Pre-Training Corpus) of the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Common schema: 80 harmonized features organized in five categories
FEATURE_SCHEMA = {
    "flow_identification": [
        "src_ip_encoded", "dst_ip_encoded", "src_port", "dst_port",
        "protocol", "flow_id", "timestamp", "duration",
    ],
    "packet_statistics": [
        "total_fwd_packets", "total_bwd_packets",
        "total_length_fwd", "total_length_bwd",
        "fwd_packet_length_max", "fwd_packet_length_min",
        "fwd_packet_length_mean", "fwd_packet_length_std",
        "bwd_packet_length_max", "bwd_packet_length_min",
        "bwd_packet_length_mean", "bwd_packet_length_std",
        "flow_bytes_per_sec", "flow_packets_per_sec",
        "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
        "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std",
        "fwd_iat_max", "fwd_iat_min",
        "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
        "bwd_iat_max", "bwd_iat_min",
    ],
    "temporal_dynamics": [
        "active_mean", "active_std", "active_max", "active_min",
        "idle_mean", "idle_std", "idle_max", "idle_min",
        "flow_start_delta", "flow_end_delta",
        "inter_arrival_time_mean", "inter_arrival_time_std",
        "burst_rate", "burst_duration",
    ],
    "protocol_specific": [
        "tcp_flags_fwd", "tcp_flags_bwd",
        "fwd_psh_flags", "bwd_psh_flags",
        "fwd_urg_flags", "bwd_urg_flags",
        "fin_flag_count", "syn_flag_count",
        "rst_flag_count", "psh_flag_count",
        "ack_flag_count", "urg_flag_count",
        "cwe_flag_count", "ece_flag_count",
        "down_up_ratio", "fwd_avg_bytes_bulk",
        "fwd_avg_packets_bulk", "bwd_avg_bytes_bulk",
        "bwd_avg_packets_bulk",
    ],
    "behavioral_derived": [
        "subflow_fwd_packets", "subflow_fwd_bytes",
        "subflow_bwd_packets", "subflow_bwd_bytes",
        "init_win_bytes_fwd", "init_win_bytes_bwd",
        "act_data_pkt_fwd", "min_seg_size_fwd",
        "fwd_header_length", "bwd_header_length",
        "fwd_seg_size_avg", "bwd_seg_size_avg",
        "entropy_src_bytes", "entropy_dst_bytes",
        "connection_density",
    ],
}

# Dataset-specific feature mappings
DATASET_MAPPINGS = {
    "CIC-IDS2018": {
        "Dst Port": "dst_port",
        "Protocol": "protocol",
        "Timestamp": "timestamp",
        "Flow Duration": "duration",
        "Tot Fwd Pkts": "total_fwd_packets",
        "Tot Bwd Pkts": "total_bwd_packets",
        "TotLen Fwd Pkts": "total_length_fwd",
        "TotLen Bwd Pkts": "total_length_bwd",
        "Label": "label",
    },
    "UNSW-NB15": {
        "srcip": "src_ip_encoded",
        "dstip": "dst_ip_encoded",
        "sport": "src_port",
        "dsport": "dst_port",
        "proto": "protocol",
        "dur": "duration",
        "sbytes": "total_length_fwd",
        "dbytes": "total_length_bwd",
        "attack_cat": "label",
    },
    "CIC-IoT-2023": {
        "src_ip": "src_ip_encoded",
        "dst_ip": "dst_ip_encoded",
        "src_port": "src_port",
        "dst_port": "dst_port",
        "protocol": "protocol",
        "flow_duration": "duration",
        "label": "label",
    },
    "Edge-IIoTset": {
        "ip.src": "src_ip_encoded",
        "ip.dst": "dst_ip_encoded",
        "ip.src_host": "src_port",
        "ip.dst_host": "dst_port",
        "frame.len": "total_length_fwd",
        "Attack_type": "label",
    },
    "Azure-Cloud": {
        "SourceIP": "src_ip_encoded",
        "DestinationIP": "dst_ip_encoded",
        "SourcePort": "src_port",
        "DestinationPort": "dst_port",
        "Protocol": "protocol",
        "Label": "label",
    },
    "Kubernetes-Docker": {
        "src_addr": "src_ip_encoded",
        "dst_addr": "dst_ip_encoded",
        "src_port": "src_port",
        "dst_port": "dst_port",
        "protocol": "protocol",
        "label": "label",
    },
}


class FeatureHarmonizer:
    """
    Three-stage feature harmonization pipeline.

    Stage 1: Map raw features to common schema (80 attributes).
    Stage 2: Impute missing features with domain-appropriate defaults.
    Stage 3: Normalize continuous features to zero mean and unit variance.
    """

    def __init__(self, n_harmonized_features=80):
        self.n_features = n_harmonized_features
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = self._build_feature_list()

    def _build_feature_list(self):
        features = []
        for category_features in FEATURE_SCHEMA.values():
            features.extend(category_features)
        return features[:self.n_features]

    def stage1_map_features(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """Map raw features from a specific dataset to the common schema."""
        mapping = DATASET_MAPPINGS.get(dataset_name, {})
        harmonized = pd.DataFrame(index=df.index)

        for raw_col, common_col in mapping.items():
            if raw_col in df.columns:
                harmonized[common_col] = df[raw_col]

        for feat in self.feature_names:
            if feat not in harmonized.columns:
                harmonized[feat] = np.nan

        if "label" in df.columns:
            harmonized["label"] = df["label"]
        elif dataset_name in mapping and mapping.get("Label", "") == "label":
            pass

        return harmonized

    def stage2_impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing features with domain-appropriate defaults."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0.0)

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if col != "label" and df[col].isna().any():
                df[col] = df[col].fillna("unknown")

        return df

    def stage3_normalize(
        self, df: pd.DataFrame, dataset_name: str, fit: bool = True
    ) -> pd.DataFrame:
        """Normalize continuous features to zero mean and unit variance."""
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "label"
        ]

        if fit:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers[dataset_name] = scaler
        else:
            if dataset_name in self.scalers:
                df[numeric_cols] = self.scalers[dataset_name].transform(
                    df[numeric_cols]
                )

        return df

    def encode_labels(
        self, labels: pd.Series, dataset_name: str, fit: bool = True
    ) -> np.ndarray:
        """Encode categorical labels to integer indices."""
        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(labels.astype(str))
            self.label_encoders[dataset_name] = le
        else:
            le = self.label_encoders[dataset_name]
            encoded = le.transform(labels.astype(str))
        return encoded

    def harmonize(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Complete three-stage harmonization pipeline."""
        harmonized = self.stage1_map_features(df, dataset_name)
        harmonized = self.stage2_impute_missing(harmonized)

        labels = None
        if "label" in harmonized.columns:
            labels = self.encode_labels(
                harmonized["label"], dataset_name, fit=fit
            )
            harmonized = harmonized.drop(columns=["label"])

        harmonized = self.stage3_normalize(harmonized, dataset_name, fit=fit)
        return harmonized, labels

    def split_temporal(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """Split data maintaining temporal ordering to prevent leakage."""
        n = len(df)
        train_end = int(n * ratios[0])
        val_end = int(n * (ratios[0] + ratios[1]))

        return {
            "train": (df.iloc[:train_end], labels[:train_end]),
            "val": (df.iloc[train_end:val_end], labels[train_end:val_end]),
            "test": (df.iloc[val_end:], labels[val_end:]),
        }
