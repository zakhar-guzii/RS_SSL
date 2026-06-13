import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    SIGNAL_NAMES = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    def __init__(self, dataset_name: str, dir_name: str, features_idxs: list[int],
                 window_size: int = 128, overlap: float = 0.5):
        self.dataset_name = dataset_name
        self.dir_name = dir_name
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))

        script_dir = os.getcwd()
        data_root = os.path.join(script_dir, "data")
        self.data_dir = os.path.join(data_root, dataset_name, dir_name)
        self.inertial_signals_dir = os.path.join(self.data_dir, "Inertial Signals")

        self.signals = self._load_signals()
        self.labels = self._load_labels()
        self.windows = self._create_windows()

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