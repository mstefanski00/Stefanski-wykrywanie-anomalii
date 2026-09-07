import torch
import numpy as np
from torch.utils.data import Dataset


class SMDWindowDataset(Dataset):

    def __init__(self, windows: np.ndarray, labels: np.ndarray = None):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.windows[idx], self.labels[idx]
        return self.windows[idx]


def create_sliding_windows(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:

    windows = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))
    return windows[::stride, 0].copy()


def align_labels_to_windows(labels: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:

    label_windows = np.lib.stride_tricks.sliding_window_view(labels, window_size)
    return label_windows[::stride].max(axis=1)

