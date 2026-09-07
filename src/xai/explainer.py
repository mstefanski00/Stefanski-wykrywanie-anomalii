from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .types import ExplanationResult
from .attribution import compute_error_matrix, compute_features_scores, get_top_k_features
from .reconstruction import compute_residual, get_reference_reconstruction

class AnomalyExplainer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:

        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def explain(self, window: np.ndarray, top_k: int = 5) -> ExplanationResult:

        self._validate_input(window)
        reconstruction = self._reconstruct(window)
        error_matrix = compute_error_matrix(window, reconstruction)
        feature_scores = compute_features_scores(error_matrix)

        return ExplanationResult(
            error_matrix=error_matrix,
            feature_scores=feature_scores,
            reconstruction=reconstruction,
            input_window=window.copy(),
            anomaly_score=float(error_matrix.mean()),
            top_k_features=get_top_k_features(feature_scores, top_k),
        )
    
    @torch.no_grad()
    def explain_batch(self, windows: np.ndarray, top_k: int = 5,) -> list[ExplanationResult]:
        
        tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)
        reconstructions = self.model(tensor).cpu().numpy()

        results = []

        for i in range(len(windows)):
            error_matrix = compute_error_matrix(windows[i], reconstructions[i])
            feature_scores = compute_features_scores(error_matrix)
            results.append(ExplanationResult(
                error_matrix=error_matrix,
                feature_scores=feature_scores,
                reconstruction=reconstructions[i],
                input_window=windows[i].copy(),
                anomaly_score=float(error_matrix.mean()),
                top_k_features=get_top_k_features(feature_scores, top_k)
            ))
        return results
    
    def get_signal_comparison(self, window: np.ndarray, feature_idx: int,) -> tuple[np.ndarray, np.ndarray]:
        
        self._validate_input(window)
        reconstruction = self._reconstruct(window)
        return get_reference_reconstruction(window, reconstruction, feature_idx)
    
    def get_residual(self, window: np.ndarray, feature_idx: int,) -> np.ndarray:
        
        self._validate_input(window)
        reconstruction = self._reconstruct(window)
        return compute_residual(window, reconstruction, feature_idx)
    
    @torch.no_grad()
    def _reconstruct(self, window: np.ndarray) -> np.ndarray:
        tensor = (
            torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        return self.model(tensor).squeeze(0).cpu().numpy()
    
    def _validate_input(self, window: np.ndarray) -> None:
        if window.ndim != 2:
            raise ValueError(f"Oczekiwano tablicy 2D (W, m), otrzymano {window.ndim}D.")
        m = window.shape[1]
        expected_m = self.model.n_features
        if m != expected_m:
            raise ValueError(f"Liczba cech okna ({m}) różni się od spodziewanej liczby cech modelu: {expected_m}.")