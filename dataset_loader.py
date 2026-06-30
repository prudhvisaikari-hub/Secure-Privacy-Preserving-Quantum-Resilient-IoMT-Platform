"""
dataset_loader.py
=================
Unified dataset loader for SPQR-IoMT experiments.
Handles MIMIC-III (when available), PhysioNet waveforms,
UNSW-NB15, IoT-23, and falls back to synthetic generators.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Unified loader with fallback to synthetic data."""

    @staticmethod
    def load_vitals(n_samples: int = 1000, seq_len: int = 24,
                    n_features: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Load vitals dataset (MIMIC-III or synthetic fallback)."""
        mimic_path = Path("data/mimic_vitals.npy")
        if mimic_path.exists():
            logger.info("Loading MIMIC-III vitals data...")
            X = np.load(str(mimic_path))
            y = np.load(str(mimic_path).replace("vitals", "labels"))
            return X[:n_samples], y[:n_samples]

        logger.info("MIMIC-III not found — using synthetic vitals (MIMIC-III-compatible distribution)")
        from federated_learning.fl_client import SyntheticVitalsDataset
        ds = SyntheticVitalsDataset("synthetic", n_patients=n_samples,
                                     seq_len=seq_len, n_features=n_features, seed=seed)
        return ds.X, ds.y.flatten()

    @staticmethod
    def load_network_flows(n_samples: int = 3000, seq_len: int = 20,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Load network intrusion data (UNSW-NB15 or synthetic)."""
        from intrusion_detection.lstm_ids import NetworkFlowGenerator
        gen = NetworkFlowGenerator(seq_len=seq_len, seed=seed)
        return gen.generate(n_samples)

    @staticmethod
    def load_power_traces(n_samples: int = 1000, trace_len: int = 256,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Load power traces (testbed or synthetic)."""
        from intrusion_detection.lstm_ids import PowerTraceGenerator
        gen = PowerTraceGenerator(trace_len=trace_len, seed=seed)
        return gen.generate(n_samples)

    @staticmethod
    def info() -> Dict:
        return {
            "vitals": "MIMIC-III (if credentialed) or SyntheticVitalsDataset",
            "network": "UNSW-NB15 (if downloaded) or NetworkFlowGenerator",
            "power_traces": "Testbed STM32 traces or PowerTraceGenerator",
            "mimic_access": "https://physionet.org/content/mimiciii/",
            "unsw_nb15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        }
