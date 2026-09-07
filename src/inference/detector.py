import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Literal
import joblib

class AnomalyDetector:
    
    AGGREGATION_MODES = ("last", "mean", "maximum")

    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device, 
                 sigma_multiplier: float = 3.0, 
                 aggregation: Literal["last", "mean", "maximum"] = "mean"
                 ):
        
        if aggregation not in self.AGGREGATION_MODES:
            raise ValueError(f"Sposób agregacji musi być wybrany spośród następujących opcji: {self.AGGREGATION_MODES}")
        
        self.model = model
        self.device = device
        self.sigma_multiplier = sigma_multiplier
        self.threshold = None
        self.aggregation = aggregation
        self.train_error_stats = {}

        self.model.eval()
        self.criterion = nn.MSELoss(reduction='none')

    def _aggregate_window_errors(self, loss_matrix: torch.Tensor) -> torch.Tensor:
        per_step_errors = loss_matrix.mean(dim = 2)

        if self.aggregation == 'last':
            return per_step_errors[:, -1]
        elif self.aggregation == 'mean':
            return per_step_errors.mean(dim=1)
        elif self.aggregation == 'maximum':
            return per_step_errors.max(dim=1).values
            
    def get_reconstruction_errors(self, dataloader) -> np.ndarray:
        errors = []
        with torch.no_grad():
            for batch in dataloader:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.to(self.device)

                loss_matrix = self.criterion(self.model(batch_x), batch_x)
                window_errors = self._aggregate_window_errors(loss_matrix)
                errors.append(window_errors.cpu().numpy())

        return np.concatenate(errors)
    
    def fit_threshold(self, train_loader) -> float:
        print("Obliczanie błędu dla zbioru treningowego (norma).")
        train_errors = self.get_reconstruction_errors(train_loader)

        mean_err = float(np.mean(train_errors))
        std_err = float(np.std(train_errors))

        self.threshold = mean_err + self.sigma_multiplier * std_err
        self.train_error_stats = {"mean": mean_err, "std": std_err}

        print(f"Rozkład błędów normy - mean: {mean_err:.6f}, std: {std_err:.6f}")
        print(f"Próg {self.sigma_multiplier}σ: {self.threshold:.6f}")

        return self.threshold
    
    def detect(self, test_loader) -> tuple[np.ndarray, np.ndarray]:

        if self.threshold is None:
            raise RuntimeError("Próg nie został wyznaczony. Wywołaj fit_treshold przed detect.")
        
        print("Szukanie anomalii w zbiorze.")
        test_errors = self.get_reconstruction_errors(test_loader)
        predictions = (test_errors > self.threshold).astype(np.int8)

        n_anomalies = predictions.sum()
        percentage = 100 * n_anomalies / len(predictions)
        print(f"Wykryte anomalie: {n_anomalies}/{len(predictions)}, {percentage:.2f}%.")

        return test_errors, predictions
    
    def save(self, path: str | Path) -> None:

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "treshold": self.threshold,
            "sigma_multiplier": self.sigma_multiplier,
            "aggregation": self.aggregation,
            "train_error_stats": self.train_error_stats
            },
            path,
        )

        print(f"Dekoder zapisany w: {path}.")

    @classmethod
    def load_threshold(cls, path: str | Path, model, device) -> "AnomalyDetector":
        state = joblib.load(path)
        detector = cls(
            model=model,
            device=device,
            sigma_multiplier=state["sigma_multiplier"],
            aggregation = state["aggregation"],
        )

        detector.threshold = state["treshold"]
        detector.train_error_stats = state["train_error_stats"]
        print(f"Wczytano próg detekcji: {detector.threshold:.6f}")
        return detector