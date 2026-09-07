from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)

from .affiliation import affiliation_precision_recall
from .vus import volume_under_surface
from .metrics_types import AnomalyMetrics

def evaluate(
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None,
        threshold_percentile: float = 99.0,
        vus_max_buffer: int = 100,
        vus_n_buffers: int = 10,
) -> AnomalyMetrics:
    
    _validate_inputs(scores, labels)

    if threshold is None:
        threshold = float(np.percentile(scores, threshold_percentile))
    
    predictions = (scores >= threshold).astype(int)

    aff_p, aff_r = affiliation_precision_recall(predictions, labels)
    aff_f1 = _f1_from_pr(aff_p, aff_r)
    vus_roc, vus_pr = volume_under_surface(
        scores, labels,
        max_buffer=vus_max_buffer,
        n_buffers=vus_n_buffers,
    )

    p_prec = float(precision_score(labels, predictions, zero_division=0))
    p_rec = float(recall_score(labels, predictions, zero_division=0))
    p_f1 = float(f1_score(labels, predictions, zero_division=0))

    preds_pa = apply_point_adjustment(labels, predictions)
    pa_prec = float(precision_score(labels, preds_pa, zero_division=0))
    pa_rec = float(recall_score(labels, preds_pa, zero_division=0))
    pa_f1 = float(f1_score(labels, preds_pa, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc_auc = float("nan")

    return AnomalyMetrics(
        affiliation_precision=aff_p,
        affiliation_recall=aff_r,
        affiliation_f1=aff_f1,
        vus_roc=vus_roc,
        vus_pr=vus_pr,
        point_precision=p_prec,
        point_recall=p_rec,
        point_f1=p_f1,
        roc_auc=roc_auc,
        threshold=threshold,
        pa_precision=pa_prec,
        pa_recall=pa_rec,
        pa_f1=pa_f1,
    )

def _f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2*precision*recall/(precision + recall)

def _validate_inputs(scores: np.ndarray, labels: np.ndarray) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"Scores ({len(scores)}) i labels ({len(labels)}) muszą mieć tę samą długość.")
    
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Labels musi mieć obie klasy: 0 i 1."
                         f"Znaleziono tylko {np.unique(labels)}.")
    
def apply_point_adjustment(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:

    adjusted_preds = preds.copy()
    anomaly_state = False
    start_idx = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not anomaly_state:
            anomaly_state = True
            start_idx = i
        elif labels[i] == 0 and anomaly_state:
            anomaly_state = False
            if np.sum(preds[start_idx:i]) > 0:
                adjusted_preds[start_idx:i] = 1

    if anomaly_state:
        if np.sum(preds[start_idx:]) > 0:
            adjusted_preds[start_idx:] = 1

    return adjusted_preds