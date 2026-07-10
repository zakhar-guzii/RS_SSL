"""Reconstruct a model architecture from a Model Bundle's ``arch`` config.

Reuses the existing training-time architecture classes so inference and
training share one definition and cannot drift.
"""

from typing import Any, Dict

import torch.nn as nn


def build_model(arch: Dict[str, Any]) -> nn.Module:
    kind = arch["type"]

    if kind == "cnn":
        from baseline_cnn import CNNClassifier

        return CNNClassifier(
            input_channels=arch["input_channels"],
            conv_out_channels=arch["conv_out_channels"],
            kernel_size=arch["kernel_size"],
            padding=arch["padding"],
            pool_kernel=arch["pool_kernel"],
            num_classes=arch["num_classes"],
            hidden_dim=arch.get("hidden_dim", 64),
            dropout=arch.get("dropout", 0.4),
        )

    if kind == "transformer":
        from transformer.transformer import TransformerClassifier

        return TransformerClassifier(
            input_channels=arch["input_channels"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            num_layers=arch["num_layers"],
            dim_feedforward=arch["dim_feedforward"],
            num_classes=arch["num_classes"],
            window_size=arch["window_size"],
            dropout=arch.get("dropout", 0.2),
            pooling=arch.get("pooling", "cls"),
        )

    raise ValueError(f"Unknown architecture type: {kind!r}")
