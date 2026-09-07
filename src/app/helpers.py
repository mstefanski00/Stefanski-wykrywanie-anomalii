from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import create_sliding_windows, align_labels_to_windows
from src.models.lstm_ae import LSTMAutoencoder

CONFIG_PATH = Path("configs/config.yaml")
ROOT = Path(__file__).resolve().parent.parent.parent


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_machines(config: dict) -> list[str]:
    models_dir = ROOT / config["paths"]["model_save"]
    scalers_dir = ROOT / config["paths"]["scaler_save"]

    if not models_dir.exists():
        return []

    stems = []
    for model_file in sorted(models_dir.glob("*.pth")):
        stem = model_file.stem
        # Sprawdzamy tylko model i skaler. JSON-a poszukamy dynamicznie.
        if (scalers_dir / f"{stem}.pkl").exists():
            stems.append(stem)
    return stems


def load_pretrained(stem: str, config: dict):
    models_dir = ROOT / config["paths"]["model_save"]
    scalers_dir = ROOT / config["paths"]["scaler_save"]
    thresholds_dir = ROOT / config["paths"]["threshold_save"]

    model_path = models_dir / f"{stem}.pth"
    scaler_path = scalers_dir / f"{stem}.pkl"
    
    # 1. Szukamy progu POT z ostatecznego eksperymentu SOTA
    pot_path = thresholds_dir / f"{stem}_recalib_v12_v9_PA_MAX.json"
    if pot_path.exists():
        thr_path = pot_path
    else:
        # Fallback do starego pliku, jeśli SOTA nie zostało wygenerowane dla danej maszyny
        thr_path = thresholds_dir / f"{stem}.json"

    if not thr_path.exists():
        raise FileNotFoundError(f"Brak pliku progów: {thr_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMAutoencoder(
        n_features=config["data"]["n_features"],
        hidden_dims=config["model"]["hidden_dims"],
        latent_dim=config["model"]["latent_dim"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = joblib.load(scaler_path)

    with open(thr_path) as f:
        thr_data = json.load(f)

    return model, scaler, thr_data["threshold"], thr_data.get("train_stats", {})


# def machine_display_name(stem: str) -> str:
#     parts = stem.split("_")
#     for i, p in enumerate(parts):
#         if p == "id":
#             return f"Serwer {parts[i+1]} ({stem})"
#     return stem
def _extract_server_id(stem: str) -> str:
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if part == "id" and i + 1 < len(parts):
            return parts[i + 1]
    return "?"
    

def machine_display_name(stem: str) -> str:
    server_id = _extract_server_id(stem)
    return f"Serwer SMD-{server_id}"


def model_display_name(stem: str) -> str:
    server_id = _extract_server_id(stem)
    return f"Model LSTM-AE-V9-SMD-{server_id}"


def run_inference(
    _model: nn.Module,
    _device: torch.device,
    _scaler,
    file_bytes_test: bytes,
    filename: str,
    config: dict,
    threshold: float,
    train_stats: dict,
    top_k: int = 5,
):
    # =========================================================
    # OPTYMALIZACJA: Czysty Polars zamiast Pandas
    # =========================================================
    df_test = pl.read_csv(io.BytesIO(file_bytes_test), null_values=["", "NA", "NaN"])
    
    # Jeśli w pliku testowym brakuje kolumny 'label', wypełnij ją zerami
    if "label" not in df_test.columns:
        df_test = df_test.with_columns(pl.lit(0).alias("label"))

    # Zamiana do macierzy numpy, odseparowanie etykiet i rzutowanie typów
    raw_test = df_test.drop("label").cast(pl.Float32, strict=False).fill_null(0.0).to_numpy()
    labels_raw = df_test.get_column("label").cast(pl.Float32, strict=False).fill_null(0.0).to_numpy()

    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]
    batch_size = config["model"]["batch_size"]

    # Transformacja danych za pomocą załadowanego RobustScalera
    test_scaled = _scaler.transform(raw_test)

    X_test = create_sliding_windows(test_scaled, window_size, stride)
    y_test = align_labels_to_windows(labels_raw, window_size, stride)

    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(test_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    # =========================================================
    # SOTA V9: Przejście z MSE na L1Loss (MAE)
    # =========================================================
    criterion_none = nn.L1Loss(reduction="none")
    agg_errors, per_feature_errors, reconstructions = [], [], []

    _model.eval()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(_device)
            recon = _model(batch_x)
            
            # Macierz błędu: Kształt -> [Batch, Window_Size, Features]
            loss_matrix = criterion_none(recon, batch_x)  

            # Uśrednianie po długości okna -> [Batch, Features]
            feat_err = loss_matrix.mean(dim=1)  
            
            # =========================================================
            # SOTA V9: Agregacja Top-K (Filtracja Szumu)
            # =========================================================
            k_val = min(top_k, feat_err.size(-1))
            e = torch.topk(feat_err, k=k_val, dim=1).values.mean(dim=1).cpu().numpy()

            agg_errors.append(e)                                   # Błąd ostateczny (Top-K)
            per_feature_errors.append(feat_err.cpu().numpy())      # Błędy per czujnik (do Heatmapy)
            reconstructions.append(recon[:, -1, :].cpu().numpy())  # Rekonstrukcja (do widoku szczegółowego)

    raw_errors = np.concatenate(agg_errors)
    
    # =========================================================
    # SOTA V9: Wygładzanie za pomocą Polars
    # =========================================================

    reconstructions_concat = np.concatenate(reconstructions)
    reconstructed_unscaled = _scaler.inverse_transform(reconstructions_concat)

    smoothed_errors = (
        pl.Series("errors", raw_errors)
        .to_frame()
        .select(pl.col("errors").rolling_mean(window_size=5, min_samples=1))
        .get_column("errors")
        .to_numpy()
    )

    results = {
        "anomaly_score": smoothed_errors,
        # "error_matrix": np.concatenate(per_feature_errors),
        "error_matrix": np.concatenate(per_feature_errors),
        # "reconstructed": np.concatenate(reconstructions),
        "reconstructed": reconstructed_unscaled,
        "labels": y_test,
        "auto_threshold": threshold,
        "train_stats": train_stats,
        "window_size": window_size,
    }
    
    raw_data_dict = {"raw_test": raw_test, "labels_test": labels_raw}
    return results, raw_data_dict