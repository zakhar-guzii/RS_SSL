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

    if kind == "cnn_lstm":
        from encoder import ActivityClassifier

        return ActivityClassifier(
            input_channels=arch["input_channels"],
            conv_out_channels=arch["conv_out_channels"],
            kernel_size=arch["kernel_size"],
            padding=arch["padding"],
            pool_kernel=arch["pool_kernel"],
            feature_dim=arch["feature_dim"],
            hidden_dim=arch["hidden_dim"],
            num_classes=arch["num_classes"],
            num_layers=arch.get("num_layers", 1),
        )

    if kind == "ssl_encoder":
        from ssl_pretrain import SSLEncoder, SSLClassifier

        encoder = SSLEncoder(
            in_channels=arch.get("in_channels", 3),
            kernel_size=arch.get("kernel_size", 5),
            padding=arch.get("padding", 2),
            pool_kernel=arch.get("pool_kernel", 2),
            hidden_dim=arch.get("hidden_dim", 128),
        )
        head = nn.Linear(arch.get("hidden_dim", 128), arch["num_classes"])
        return SSLClassifier(encoder, head)

    raise ValueError(f"Unknown architecture type: {kind!r}")
