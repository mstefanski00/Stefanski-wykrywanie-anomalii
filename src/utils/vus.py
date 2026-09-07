from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def volume_under_surface(
        scores: np.ndarray,
        labels: np.ndarray,
        max_buffer: int = 100,
        n_buffers: int = 10,
        ) -> tuple[float, float]:
    
    buffer_values = np.linspace(0, max_buffer, n_buffers, dtype=int)
    roc_aucs, avg_precs = [], []

    for buf in buffer_values:
        expanded = _expand_labels(labels, int(buf))
        try:
            roc_aucs.append(float(roc_auc_score(expanded, scores)))
            avg_precs.append(float(average_precision_score(expanded, scores)))
        except ValueError:
            continue

    if not roc_aucs:
        return float("nan"), float("nan")
    
    return float(np.mean(roc_aucs)), float(np.mean(avg_precs))

def _expand_labels(labels: np.ndarray, buffer: int) -> np.ndarray:
    if buffer == 0:
        return labels.copy()
    expanded = labels.copy()
    for idx in np.where(labels == 1)[0]:
        lo = max(0, idx - buffer)
        hi = min(len(labels) - 1, idx + buffer)
        expanded[lo : hi + 1] = 1
    return expanded