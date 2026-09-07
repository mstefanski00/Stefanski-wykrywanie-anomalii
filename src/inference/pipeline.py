from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data import prepare_dataloaders
from src.utils import evaluate
from src.utils.io import save_metrics
from .loader import build_model, load_or_fit_detector

def run_evaluation(
        config: dict,
        config_path: Path,
        file_name: str,
        detector_path: Path | None,
        output_dir: Path,
        num_workers: int = 4,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}.")

    print(f"Przygotowanie danych: {file_name}.")
    train_loader, test_loader = prepare_dataloaders(
        config_path=config_path,
        file_name=file_name,
        num_workers=num_workers
    )

    print("\nWczytywanie modelu i detektora.")
    model = build_model(config, device)
    detector = load_or_fit_detector(config, model, device, train_loader, detector_path)

    print("\nDetekcja anomalii.")
    scores, _ = detector.detect(test_loader)

    print("\nObliczanie metryk.")
    labels = _extract_labels(test_loader)

    if len(scores) != len(labels):
        raise RuntimeError(f"Niezgodność długości: scores={len(scores)}, labels={len(labels)}. "
            f"Sprawdź window_size i stride w configu."
        )
    
    metrics = evaluate(scores=scores, labels=labels, threshold=detector.threshold)
    print(f"\n{metrics}")

    out_path = save_metrics(metrics, file_name, output_dir, detector.threshold)
    print(f"\nWyniki zapisano w pliku: {out_path}")


def _extract_labels(test_loader) -> np.ndarray:
    all_labels = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            all_labels.append(batch[1].numpy())
        else:
            raise RuntimeError(
                "test_loader nie zawiera etykiet. "
                "Sprawdź, czy plik CSV zawiera kolumnę Label."
            )
    return np.concatenate(all_labels).astype(int)