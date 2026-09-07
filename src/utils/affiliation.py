from __future__ import annotations
import numpy as np

def affiliation_precision_recall(predictions: np.ndarray, labels: np.ndarray,) -> tuple[float, float]:
    pred_ranges = _binary_to_ranges(predictions)
    true_ranges = _binary_to_ranges(labels)

    if len(true_ranges) == 0:
        return (1.0  if len(pred_ranges) == 0 else 0.0), 1.0
    if len(pred_ranges) == 0:
        return 0.0, 0.0
    
    precision = float(np.mean([
        _best_overlap_ratio(p_start, p_end, true_ranges)
        for p_start, p_end in pred_ranges
    ]))

    recall = float(np.mean([
        _coverage_ratio(t_start, t_end, pred_ranges)
        for t_start, t_end in true_ranges
    ]))

    return precision, recall

def _coverage_ratio(t_start: int, t_end: int, pred_ranges: list[tuple[int, int]],) -> float:
    t_len = t_end - t_start + 1
    covered = np.zeros(t_len, dtype=bool)

    for p_start, p_end in pred_ranges:
        lo = max(t_start, p_start) - t_start
        hi = min(t_end, p_end) - t_start

        if hi >= lo:
            covered[lo: hi + 1] = True
    return float(covered.sum()/ t_len)


def _binary_to_ranges(arr: np.ndarray) -> list[tuple[int, int]]:
    ranges = []
    i=0
    n  = len(arr)

    while i < n:
        if arr[i] == 1:
            start = i
            while i < n and arr[i] == 1:
                i+= 1
            ranges.append((start, i-1))
        else:
            i += 1
    return ranges

def _best_overlap_ratio(p_start: int, p_end: int, true_ranges: list[tuple[int, int]],) -> float:
    p_len = p_end - p_start + 1
    best = 0.0
    for t_start, t_end in true_ranges:
        overlap = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
        ratio = overlap / p_len
        if ratio > best:
            best = ratio
    return best