from __future__ import annotations

import numpy as np

def compute_error_matrix(window: np.ndarray, reconstruction: np.ndarray,) -> np.ndarray:
        return np.abs(window - reconstruction)

def compute_features_scores(error_matrix: np.ndarray) -> np.ndarray:
        return error_matrix.mean(axis=0)
    
def get_top_k_features(feature_scores: np.ndarray, k:int) -> np.ndarray:
    k = min(k, len(feature_scores))
    return np.argsort(feature_scores)[::-1][:k]