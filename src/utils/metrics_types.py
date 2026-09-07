from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AnomalyMetrics:
    affiliation_precision: float
    affiliation_recall: float
    affiliation_f1: float
    vus_roc: float
    vus_pr: float

    point_precision: float
    point_recall: float
    point_f1: float
    roc_auc: float
    threshold: float

    pa_precision: float
    pa_recall: float
    pa_f1: float

    def __str__(self) -> str:
        return (
            "Metryki oceny oparte na zakresach\n"
            f"  Affiliation Precision : {self.affiliation_precision:.4f}\n"
            f"  Affiliation Recall    : {self.affiliation_recall:.4f}\n"
            f"  Affiliation F1        : {self.affiliation_f1:.4f}\n"
            f"  VUS-ROC               : {self.vus_roc:.4f}\n"
            f"  VUS-PR                : {self.vus_pr:.4f}\n"
            "Metryki pomocnicze (punktowe, tylko do referencji)\n"
            f"  Point F1              : {self.point_f1:.4f}\n"
            f"  Point Precision       : {self.point_precision:.4f}\n"
            f"  Point Recall          : {self.point_recall:.4f}\n"
            f"  ROC-AUC               : {self.roc_auc:.4f}\n"
            "Metryki z protokołem Point-Adjusted\n"
            f"  PA Precision          : {self.pa_precision:.4f}\n"
            f"  PA Recall             : {self.pa_recall:.4f}\n"
            f"  PA F1                 : {self.pa_f1:.4f}\n"
            f"  Próg decyzyjny        : {self.threshold:.6f}"
        )