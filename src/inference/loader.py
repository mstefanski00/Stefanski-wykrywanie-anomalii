from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.lstm_ae import LSTMAutoencoder
from .detector import AnomalyDetector

def build_model(config: dict, device: torch.device) -> LSTMAutoencoder:
    model_cfg = config["model"]
    model = LSTMAutoencoder(
        n_features=config["data"]["n_features"],
        hidden_dims = model_cfg["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],

    )

    model_path = Path(config["paths"]["model_save"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Wczytano model z {model_path}.")
    return model

def load_or_fit_detector(
        config: dict,
        model: nn.Module,
        device: torch.device,
        train_loader,
        detector_path: Path | None,
) -> AnomalyDetector:
    thr_cfg = config["thresholding"]

    if detector_path and detector_path.exists():
        return AnomalyDetector.load_threshold(
            path=detector_path,
            model=model,
            device=device,
        )
    
    print("Brak zapisanego stanu detektora - wyznaczam próg z danych treningowych.")
    detector = AnomalyDetector(
        model=model,
        device=device,
        sigma_multiplier=thr_cfg["sigma_multiplier"],
        aggregation=thr_cfg["aggregation"],
    )
    detector.fit_threshold(train_loader)
    return detector