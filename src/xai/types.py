from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ExplanationResult:

    error_matrix: np.ndarray
    feature_scores: np.ndarray
    recontruction: np.ndarray
    input_window: np.ndarray
    anomaly_score: float
    top_k_features: np.ndarray