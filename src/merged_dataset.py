import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple


class MergedHARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, norm_mean: np.ndarray = None, norm_std: np.ndarray = None):
        if norm_mean is None:
            norm_mean = X.mean(axis=(0, 1), keepdims=True)
            norm_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X = (X - norm_mean) / norm_std
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_and_prepare_data(config) -> Tuple[DataLoader, DataLoader, DataLoader, "MergedHARDataset", "MergedHARDataset"]:
    npz_path = config["dataset"]["paths"]["root"]
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    seed = config["training"].get("seed", 42)
    test_ratio = config["dataset"]["split"]["test_ratio"]
    val_ratio = config["dataset"]["split"]["val_ratio"]

    # Stratified hold-out: carve out test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(X, y))
    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Stratified hold-out: carve out val set from trainval
    val_share = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))
    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    train_dataset = MergedHARDataset(X_train, y_train)
    norm_mean = train_dataset.norm_mean
    norm_std = train_dataset.norm_std

    val_dataset = MergedHARDataset(X_val, y_val, norm_mean=norm_mean, norm_std=norm_std)
    test_dataset = MergedHARDataset(X_test, y_test, norm_mean=norm_mean, norm_std=norm_std)

    batch_size = config["training"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_dataset, test_dataset
