import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class CustomDataset(Dataset):
    SIGNAL_NAMES = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    def __init__(self, dataset_name: str, dir_name: str, window_size: int = 128, overlap: float = 0.5, norm_mean: Optional[float] = None, norm_std: Optional[float] = None):
        if not (0.0 <= overlap < 1.0):
             raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        
        self.dataset_name = dataset_name
        self.dir_name = dir_name
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        
        if self.stride < 1:
             raise ValueError(f"Invalid stride {self.stride}; choose a smaller overlap or a larger window_size")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        data_root = os.path.join(repo_root, "data")
        self.data_dir = os.path.join(data_root, dataset_name, dir_name)
        self.inertial_signals_dir = os.path.join(self.data_dir, "Inertial Signals")

        self.signals = self._load_signals()
        self.labels = self._load_labels()
        self.windows = self._create_windows()
        
        if norm_mean is not None and norm_std is not None:
            self.signals = (self.signals - norm_mean) / (norm_std + 1e-8)
        else:
            norm_mean = self.signals.mean()
            norm_std = self.signals.std()
            self.norm_mean = norm_mean
            self.norm_std = norm_std
            self.signals = (self.signals - norm_mean) / (norm_std + 1e-8)

        print(f"✓ Loaded {len(self)} windowed samples from {dir_name} set")

    def _load_signals(self) -> np.ndarray:
        signals = []
        for signal_name in self.SIGNAL_NAMES:
            file_path = os.path.join(
                self.inertial_signals_dir,
                f"{signal_name}_{self.dir_name}.txt"
            )
            signal = np.loadtxt(file_path)
            signals.append(signal)

        signals = np.stack(signals, axis=0)
        signals = signals.transpose(1, 2, 0)

        mean = signals.mean()
        std = signals.std()
        signals = (signals - mean) / (std + 1e-8)

        return signals

    def _load_labels(self) -> np.ndarray:
        label_file = os.path.join(self.data_dir, f"y_{self.dir_name}.txt")
        labels = np.loadtxt(label_file, dtype=int)
        labels = labels - 1
        return labels

    def _create_windows(self) -> list:
        windows = []
        num_samples = self.signals.shape[0]

        for sample_idx in range(num_samples):
            signal_length = self.signals.shape[1]
            start_idx = 0

            while start_idx + self.window_size <= signal_length:
                end_idx = start_idx + self.window_size
                windows.append((sample_idx, start_idx, end_idx))
                start_idx += self.stride

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start_idx, end_idx = self.windows[idx]
        window = self.signals[sample_idx, start_idx:end_idx, :]
        label = self.labels[sample_idx]

        x = torch.tensor(window, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y