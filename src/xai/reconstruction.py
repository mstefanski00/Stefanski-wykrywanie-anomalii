from __future__ import annotations
import numpy as np

def get_reference_reconstruction(
        window: np.ndarray,
        reconstruction: np.ndarray,
        feature_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = window.shape[1]
    if not (0 <= feature_idx < m):
        raise IndexError(f"feature_idx = {feature_idx} poza zakresem [0, {m-1}].")
    
    x_real = window [:, feature_idx]
    x_hat = reconstruction[:, feature_idx]

    return x_real, x_hat

def compute_residual(
        window: np.ndarray,
        reconstruction: np.ndarray,
        feature_idx: int,
) -> np.ndarray:
    m = window.shape[1]
    if not (0 <= feature_idx < m):
        raise IndexError(f"feature_idx = {feature_idx} poza zakresem [0, {m-1}].")
    
    return window[:, feature_idx] - reconstruction[:, feature_idx]