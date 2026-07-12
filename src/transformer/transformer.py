import os
import sys

# Make sibling modules in src/ importable (matches encoder.py's flat-import style)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import mlflow
import mlflow.pytorch
import mlflow.models
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from config_loader import ConfigLoader
from encoder import (
    LABEL_NAMES,
    set_seed,
    load_and_prepare_data,
    train_model,
    evaluate_and_log,
)


class TransformerClassifier(nn.Module):
    """Transformer encoder over raw inertial-signal windows (B, T, C)."""

    def __init__(
        self,
        input_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        window_size: int,
        dropout: float = 0.2,
        pooling: str = "cls",
    ):
        super(TransformerClassifier, self).__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling}")
        self.pooling = pooling

        self.input_proj = nn.Linear(input_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 position for the prepended CLS token
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.clf = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std
                              =0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        batch_size = x.size(0)
        x = self.input_proj(x)                                  # (B, T, d_model)

        cls = self.cls_token.expand(batch_size, -1, -1)         # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                          # (B, T+1, d_model)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.dropout(x)

        x = self.encoder(x)                                     # (B, T+1, d_model)

        if self.pooling == "cls":
            pooled = x[:, 0]                                    # CLS token output
        else:
            pooled = x[:, 1:].mean(dim=1)                       # mean over timesteps

        return self.clf(pooled)


def build_transformer(config, device):
    """Create the transformer model from config and move to device."""
    t_cfg = config["model"]["transformer"]
    model = TransformerClassifier(
        input_channels=t_cfg["input_channels"],
        d_model=t_cfg["d_model"],
        nhead=t_cfg["nhead"],
        num_layers=t_cfg["num_layers"],
        dim_feedforward=t_cfg["dim_feedforward"],
        num_classes=config["model"]["classifier"]["num_classes"],
        window_size=config["model"]["dataset"]["window_size"],
        dropout=config["model"]["dropout"],
        pooling=t_cfg.get("pooling", "cls"),
    )
    model.to(device)
    return model


def log_model_to_mlflow(model, test_dataloader, device, mlflow):
    """Create a signature and log the PyTorch transformer model."""
    print("\nLogging model to MLflow...")

    input_example_data = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            if i >= 1:
                break
            input_example_data.append(x.cpu().numpy())

    input_example = np.concatenate(input_example_data)[:5]

    model.eval()
    with torch.no_grad():
        example_input_tensor = torch.tensor(input_example, dtype=torch.float32).to(device)
        example_output = model(example_input_tensor).cpu().numpy()

    signature = mlflow.models.infer_signature(input_example, example_output)

    mlflow.pytorch.log_model(
        model,
        "transformer_model",
        signature=signature,
        input_example=input_example,
    )

    print("Model logged to MLflow\n")


def run_transformer_classifier():
    config = ConfigLoader().load_experiment("har_merged", "transformer")

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TRANSFORMER FOR HUMAN ACTIVITY RECOGNITION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Num epochs: {config['training']['num_epochs']}")
    print("=" * 60 + "\n")

    train_dataloader, val_dataloader, test_dataloader, train_dataset, test_dataset = load_and_prepare_data(config)

    mlflow.set_tracking_uri(config["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.set_tags(config["mlflow"]["tags"])

        mlflow.log_params({
            "dataset": "har_merged",
            "seed": seed,
            "device": device,
        })

        mlflow.log_params({
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs": config["training"]["num_epochs"],
            "patience": config["training"]["patience"],
            "weight_decay": config["training"]["weight_decay"],
        })

        t_cfg = config["model"]["transformer"]
        mlflow.log_params({
            "transformer_input_channels": t_cfg["input_channels"],
            "transformer_d_model": t_cfg["d_model"],
            "transformer_nhead": t_cfg["nhead"],
            "transformer_num_layers": t_cfg["num_layers"],
            "transformer_dim_feedforward": t_cfg["dim_feedforward"],
            "transformer_pooling": t_cfg.get("pooling", "cls"),
            "classifier_num_classes": config["model"]["classifier"]["num_classes"],
            "dropout": config["model"]["dropout"],
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
        })

        model = build_transformer(config, device)
        train_model(model, train_dataloader, val_dataloader, config, device, mlflow)

        preds, labels = evaluate_and_log(model, test_dataloader, device, mlflow)

        # Print F1 scores to stdout
        f1_weighted = f1_score(labels, preds, average="weighted")
        f1_macro = f1_score(labels, preds, average="macro")
        per_class_f1 = f1_score(labels, preds, average=None)

        print("\n" + "=" * 60)
        print("TRANSFORMER TEST F1 SCORES")
        print("=" * 60)
        print(f"Test F1 (weighted): {f1_weighted:.4f}")
        print(f"Test F1 (macro):    {f1_macro:.4f}")
        print("-" * 60)
        for name, score in zip(LABEL_NAMES, per_class_f1):
            print(f"  {name:<22} F1: {score:.4f}")
        print("=" * 60 + "\n")

        log_model_to_mlflow(model, test_dataloader, device, mlflow)

        # ── Export Model Bundle ──
        from server.bundle import save_bundle

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        save_bundle(
            model=model,
            arch={
                "type": "transformer",
                "input_channels": t_cfg["input_channels"],
                "d_model": t_cfg["d_model"],
                "nhead": t_cfg["nhead"],
                "num_layers": t_cfg["num_layers"],
                "dim_feedforward": t_cfg["dim_feedforward"],
                "num_classes": config["model"]["classifier"]["num_classes"],
                "window_size": config["model"]["dataset"]["window_size"],
                "dropout": config["model"]["dropout"],
                "pooling": t_cfg.get("pooling", "cls"),
            },
            norm_mean=train_dataset.norm_mean,
            norm_std=train_dataset.norm_std,
            label_order=LABEL_NAMES,
            models_dir=os.path.join(repo_root, "models"),
            id="transformer",
            display_name="Transformer (supervised)",
            description="Supervised Transformer encoder over raw signal windows, trained on 100% of the merged dataset's labels.",
            ssl_pretrained=False,
            is_default=False,
        )


if __name__ == "__main__":
    run_transformer_classifier()
