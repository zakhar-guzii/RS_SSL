import mlflow
import mlflow.pytorch
import mlflow.models
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple

from torch_dataset import CustomDataset
from config_loader import ConfigLoader


LABEL_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
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
    ax.set_title("Activity Classifier — Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    

class CNNFeatureEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int = 64):
        super(CNNFeatureEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.conv(x)
        features = self.projection(features)
        features = features.unsqueeze(1)
        return features
    
    
class LSTMFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 1):
        super(LSTMFeatureEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)
    

class ActivityClassifier(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1):
        super(ActivityClassifier, self).__init__()
        self.cnn = CNNFeatureEncoder(input_channels, feature_dim)
        self.lstm = LSTMFeatureEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        self.clf = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(x)
        lstm_output, (h_c, c_n) = self.lstm(cnn_features)
        result = self.clf(h_c[-1])
        return result


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
    mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)
    mlflow.log_metric("train_accuracy", epoch_acc / len(train_loader), step=epoch)
    
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


def load_and_prepare_data(config):
    window_size = config["model"]["dataset"]["window_size"]
    overlap = config["model"]["dataset"]["overlap"]

    train_dataset = CustomDataset("uci_har", "train", [], window_size=window_size, overlap=overlap)
    test_dataset = CustomDataset("uci_har", "test", [], window_size=window_size, overlap=overlap)
    
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.2,
        random_state=42
    )
    
    batch_size = config["training"]["batch_size"]
    train_dataloader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, train_dataset, test_dataset


def build_model(config, device):
    """Create and move model to device."""
    model = ActivityClassifier(
        config["model"]["cnn"]["input_channels"],
        config["model"]["lstm"]["feature_dim"],
        config["model"]["lstm"]["hidden_dim"],
        config["model"]["classifier"]["num_classes"],
        config["model"]["lstm"]["num_layers"]
    )
    model.to(device)
    return model


def train_model(model, train_loader, val_loader, config, device, mlflow):
    """Main training loop with validation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    
    patience = config["training"]["patience"]
    best_val_acc = 0
    no_improve_count = 0
    
    for epoch in range(config["training"]["num_epochs"]):
        train_epoch(model, train_loader, optimizer, criterion, scheduler, device, mlflow, epoch=epoch)
        
        # Validation
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = balanced_accuracy_score(val_labels, val_preds)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        print(f"✓ Val Accuracy: {val_acc:.4f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model


def evaluate_and_log(model, test_loader, device, mlflow):
    """Evaluate on test set and log metrics."""
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
    
    # Log artifacts
    report = classification_report(labels, preds, target_names=LABEL_NAMES)
    with open("/tmp/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("/tmp/classification_report.txt")
    
    cm_path = "/tmp/confusion_matrix.png"
    plot_confusion_matrix(labels, preds, cm_path)
    mlflow.log_artifact(cm_path)
    
    return preds, labels


def log_model_to_mlflow(model, test_dataloader, device, mlflow):
    """Create signature and log PyTorch model."""
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
        "cnn_lstm_model",
        signature=signature,
        input_example=input_example
    )
    
    print("✓ Model logged to MLflow\n")


def run_cnn_lstm_classifier():
    config = ConfigLoader().load_experiment("uci_har", "cnn_lstm")

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("CNN-LSTM FOR HUMAN ACTIVITY RECOGNITION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Window size: {config['model']['dataset']['window_size']}")
    print(f"Overlap: {config['model']['dataset']['overlap']}")
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
            "dataset": "uci_har",
            "window_size": config["model"]["dataset"]["window_size"],
            "overlap": config["model"]["dataset"]["overlap"],
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
            "cnn_output_dim": config["model"]["cnn"]["output_dim"],
            "lstm_feature_dim": config["model"]["lstm"]["feature_dim"],
            "lstm_hidden_dim": config["model"]["lstm"]["hidden_dim"],
            "lstm_num_layers": config["model"]["lstm"]["num_layers"],
            "classifier_num_classes": config["model"]["classifier"]["num_classes"],
            "dropout": config["model"]["dropout"],
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
        })

        model = build_model(config, device)
        train_model(model, train_dataloader, val_dataloader, config, device, mlflow)

        evaluate_and_log(model, test_dataloader, device, mlflow)

        log_model_to_mlflow(model, test_dataloader, device, mlflow)


if __name__ == "__main__":
    run_cnn_lstm_classifier()