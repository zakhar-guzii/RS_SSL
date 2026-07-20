import os
import mlflow
import mlflow.pytorch
import mlflow.models
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from typing import Tuple

from config_loader import ConfigLoader
from merged_dataset import MergedHARDataset, load_and_prepare_data


LABEL_NAMES = [
    "downstairs",
    "sit",
    "stand",
    "upstairs",
    "walk",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(y_pred, y_true, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("CNN Baseline — Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


class CNNClassifier(nn.Module):
    def __init__(self, input_channels: int, conv_out_channels: list, kernel_size: int, padding: int, pool_kernel: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.4):
        super(CNNClassifier, self).__init__()
        # Normalize the input channels as the first layer, so the learned
        # mean/var live in the checkpoint and travel with the weights. This is
        # what stops the train/serve normalization seam from reopening: there is
        # no separate stats file for serving to apply inconsistently.
        layers = [nn.BatchNorm1d(input_channels)]
        in_channels = input_channels

        for out_channels in conv_out_channels:
            layers.extend([
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_kernel),
            ])
            in_channels = out_channels

        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        ])

        self.conv = nn.Sequential(*layers)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.conv(x)
        return self.clf(features)


def build_model(config, device):
    model = CNNClassifier(
        input_channels=config["model"]["cnn"]["input_channels"],
        conv_out_channels=config["model"]["cnn"]["conv_out_channels"],
        kernel_size=config["model"]["cnn"]["kernel_size"],
        padding=config["model"]["cnn"]["padding"],
        pool_kernel=config["model"]["cnn"]["pool_kernel"],
        num_classes=config["model"]["classifier"]["num_classes"],
        hidden_dim=config["model"]["classifier"]["hidden_dim"],
        dropout=config["model"]["dropout"],
    )
    model.to(device)
    return model


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, criterion, scheduler, device, mlflow, epoch=0):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            accuracy = (predictions == y).float().mean()
            epoch_acc += accuracy.item()
            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy.item():.4f}"
            })

        scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
    mlflow.log_metric("train_accuracy", avg_acc, step=epoch)

    print(f"✓ Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}\n")


def evaluate(model: nn.Module, test_loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds_list.append(preds.cpu().numpy())
            all_labels_list.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    return all_preds, all_labels


def train_model(model, train_loader, val_loader, config, device, mlflow):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])

    patience = config["training"]["patience"]
    best_val_f1 = 0.0
    no_improve_count = 0

    for epoch in range(config["training"]["num_epochs"]):
        train_epoch(model, train_loader, optimizer, criterion, scheduler, device, mlflow, epoch=epoch)

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = balanced_accuracy_score(val_labels, val_preds)
        val_f1_macro = f1_score(val_labels, val_preds, average="macro")
        per_class_f1 = f1_score(val_labels, val_preds, average=None)
        class_counts = np.bincount(val_labels, minlength=len(LABEL_NAMES))

        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_f1_macro", val_f1_macro, step=epoch)

        print(f"✓ Val Balanced Accuracy: {val_acc:.4f} | Val F1 Macro: {val_f1_macro:.4f}")
        total_val = len(val_labels)
        for name, count, f1 in zip(LABEL_NAMES, class_counts, per_class_f1):
            print(f"   {name:<12} {count:>5} / {total_val} samples   F1: {f1:.4f}")
        print()

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model


def evaluate_and_log(model, test_loader, device, mlflow):
    preds, labels = evaluate(model, test_loader, device)

    test_acc = balanced_accuracy_score(labels, preds)
    test_f1_weighted = f1_score(labels, preds, average="weighted")
    test_f1_macro = f1_score(labels, preds, average="macro")

    per_class_f1 = f1_score(labels, preds, average=None)
    per_class_metrics = {
        f"f1_{name}": float(score)
        for name, score in zip(LABEL_NAMES, per_class_f1)
    }

    mlflow.log_metrics({
        "test_balanced_accuracy": float(test_acc),
        "test_f1_weighted": float(test_f1_weighted),
        "test_f1_macro": float(test_f1_macro),
        **per_class_metrics,
    })

    report = classification_report(labels, preds, target_names=LABEL_NAMES)
    with open("/tmp/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("/tmp/classification_report.txt")

    cm_path = "/tmp/confusion_matrix.png"
    plot_confusion_matrix(preds, labels, cm_path)
    mlflow.log_artifact(cm_path)

    return preds, labels


def log_model_to_mlflow(model, test_dataloader, device, mlflow):
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
        "baseline_cnn_model",
        signature=signature,
        input_example=input_example
    )

    print("✓ Model logged to MLflow\n")


def run_cnn_baseline():
    config = ConfigLoader().load_experiment("har_merged", "baseline_cnn")

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("CNN BASELINE FOR MERGED HAR DATASET")
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
            "num_classes": config["model"]["classifier"]["num_classes"],
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

        mlflow.log_params({
            "cnn_input_channels": config["model"]["cnn"]["input_channels"],
            "cnn_conv_channels": str(config["model"]["cnn"]["conv_out_channels"]),
            "cnn_kernel_size": config["model"]["cnn"]["kernel_size"],
            "classifier_num_classes": config["model"]["classifier"]["num_classes"],
            "classifier_hidden_dim": config["model"]["classifier"]["hidden_dim"],
            "dropout": config["model"]["dropout"],
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
        })

        model = build_model(config, device)
        train_model(model, train_dataloader, val_dataloader, config, device, mlflow)

        evaluate_and_log(model, test_dataloader, device, mlflow)

        log_model_to_mlflow(model, test_dataloader, device, mlflow)

        # ── Export Model Bundle ──
        from server.bundle import save_bundle

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        save_bundle(
            model=model,
            arch={
                "type": "cnn",
                "input_channels": config["model"]["cnn"]["input_channels"],
                "conv_out_channels": config["model"]["cnn"]["conv_out_channels"],
                "kernel_size": config["model"]["cnn"]["kernel_size"],
                "padding": config["model"]["cnn"]["padding"],
                "pool_kernel": config["model"]["cnn"]["pool_kernel"],
                "num_classes": config["model"]["classifier"]["num_classes"],
                "hidden_dim": config["model"]["classifier"]["hidden_dim"],
                "dropout": config["model"]["dropout"],
            },
            norm_mean=train_dataset.norm_mean,
            norm_std=train_dataset.norm_std,
            label_order=LABEL_NAMES,
            models_dir=os.path.join(repo_root, "models"),
            id="baseline_cnn",
            display_name="CNN (supervised baseline)",
            description="Plain supervised CNN classifier, trained on 100% of the merged dataset's labels.",
            ssl_pretrained=False,
            is_default=True,
        )


if __name__ == "__main__":
    run_cnn_baseline()
