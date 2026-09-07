from __future__ import annotations

import json
from pathlib import Path

from .metrics_types import AnomalyMetrics

def save_metrics(
        metrics: AnomalyMetrics,
        file_name: str,
        output_dir: Path,
        threshold: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(file_name).stem}_metrics.json"

    payload = {
        "file": file_name,
        "threshold": threshold,
        "affiliation": {
            "precision": round(metrics.affiliation_precision, 6),
            "recall": round(metrics.affiliation_recall, 6),
            "f1": round(metrics.affiliation_f1, 6),
        },

        "vus": {
            "roc": round(metrics.vus_roc, 6),
            "pr": round(metrics.vus_pr, 6)
        },

        "point_metrics_reference_only": {
            "precision": round(metrics.point_precision, 6),
            "recall": round(metrics.point_recall, 6),
            "f1": round(metrics.point_f1, 6),
            "roc_auc": round(metrics.roc_auc, 6)
        },
    }

    with open (out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    return out_path