"""
data2vec-inspired Self-Supervised Learning for HAR time-series.

Core idea (simplified from https://arxiv.org/abs/2202.03555):
  1. Teacher  = EMA copy of encoder; sees FULL signal → produces targets
  2. Student  = encoder; sees MASKED signal → predicts teacher targets
  3. Loss     = MSE between student predictions and teacher representations
  4. No labels used during pre-training.

After pre-training: label budget sweep (1%, 5%, 10%, 25%, 100%).
Each budget fine-tunes encoder + linear head jointly and reports F1 Macro.
This answers the core research question from CP1/CP2:
  "Can SSL + 1% labels match supervised with 10% labels (F1 ≥ 0.925)?"
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────
# Dataset mode — switch here when merged arrives
# ─────────────────────────────────────────────

DATASET_CONFIG = {
    "in_channels": 3,
    "num_classes": 5,
    "label_names": ["downstairs", "sit", "stand", "upstairs", "walk"],
    "npz_path": "./data/merged/har_merged.npz",
}

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

WINDOW_SIZE     = 128
EMA_DECAY       = 0.999
MASK_RATIO      = 0.40
HIDDEN_DIM      = 128
BATCH_SIZE      = 64
PRETRAIN_EPOCHS  = 10
FINETUNE_EPOCHS  = 10
PRETRAIN_LR      = 1e-3
HEAD_LR          = 1e-3
FINETUNE_LR      = 1e-4   # lower LR for encoder during fine-tuning
PATIENCE        = 7

LABEL_BUDGETS   = [0.01, 0.05, 0.10, 0.25, 1.00]  # 1%, 5%, 10%, 25%, 100%


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────
# Encoder backbone
# ─────────────────────────────────────────────

class SSLEncoder(nn.Module):
    """
    1D CNN encoder: (B, T, C) → (B, hidden_dim)

    Conv1d → BN → ReLU → MaxPool  (×3)
    AdaptiveAvgPool → Flatten → Linear projection
    """
    def __init__(self, in_channels=3,
                 kernel_size=5, padding=2, pool_kernel=2, hidden_dim=128):
        super().__init__()
        conv_channels = [64, 128, 256]

        layers, ch = [], in_channels
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(ch, out_ch, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_kernel),
            ]
            ch = out_ch

        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(ch, hidden_dim)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        return self.proj(self.conv(x.transpose(1, 2)))


class SSLClassifier(nn.Module):
    """Encoder + linear head combined into one module, so a fine-tuned SSL
    model has a single state_dict for Model Bundle export/inference —
    mirrors how the supervised architectures (CNNClassifier, etc.) already
    package encoder+head as one module."""

    def __init__(self, encoder: SSLEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        return self.head(self.encoder(x))


# ─────────────────────────────────────────────
# data2vec model
# ─────────────────────────────────────────────

class Data2Vec(nn.Module):
    """
    Student (masked input) predicts Teacher (full input) representations.
    Teacher = EMA copy of student. No labels needed.
    """
    def __init__(self, encoder: SSLEncoder, hidden_dim=128):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @torch.no_grad()
    def update_teacher(self, decay=EMA_DECAY):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = decay * pt.data + (1.0 - decay) * ps.data

    def mask_input(self, x):
        B, T, _ = x.shape
        n = int(T * MASK_RATIO)
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i in range(B):
            mask[i, torch.randperm(T, device=x.device)[:n]] = True
        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask

    def forward(self, x):
        with torch.no_grad():
            target = self.teacher(x)
        masked_x, _ = self.mask_input(x)
        pred = self.predictor(self.student(masked_x))
        return F.mse_loss(pred, target.detach())


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────


def load_merged(batch_size=BATCH_SIZE):
    from merged_dataset import load_and_prepare_data
    cfg = {
        "dataset": {"paths": {"root": DATASET_CONFIG["npz_path"]},
                    "split": {"test_ratio": 0.15, "val_ratio": 0.15}},
        "training": {"batch_size": batch_size, "seed": 42},
    }
    train_loader, val_loader, test_loader, train_ds, _ = load_and_prepare_data(cfg)
    return train_loader, val_loader, test_loader, train_ds




# ─────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────

def pretrain(model: Data2Vec, train_loader, device):
    optimizer = optim.Adam(
        list(model.student.parameters()) + list(model.predictor.parameters()),
        lr=PRETRAIN_LR, weight_decay=1e-5
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    best_loss = float("inf")
    no_improve = 0

    print("=" * 60)
    print("PHASE 1: SSL PRE-TRAINING (data2vec)")
    print(f"  Mask ratio : {MASK_RATIO:.0%}  |  EMA decay: {EMA_DECAY}")
    print(f"  Epochs     : {PRETRAIN_EPOCHS}  |  Patience: {PATIENCE}")
    print("=" * 60)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for x, _ in tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False):
            x = x.to(device)
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_teacher()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        mlflow.log_metric("pretrain_loss", avg, step=epoch)
        print(f"  Epoch {epoch:02d} | loss: {avg:.4f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            no_improve = 0
            torch.save(model.student.state_dict(), "/tmp/ssl_encoder_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.student.load_state_dict(torch.load("/tmp/ssl_encoder_best.pt", map_location=device, weights_only=True))
    print(f"\n  ✓ Pre-training done. Best loss: {best_loss:.4f}\n")
    return model.student


# ─────────────────────────────────────────────
# Linear probe for one label budget
# ─────────────────────────────────────────────

def train_probe_for_budget(encoder, full_train_ds, val_loader, test_loader,
                           budget: float, cfg: dict, device: str):
    """
    Subsample `budget` fraction of labeled train data,
    fine-tune encoder + linear head jointly, return test F1 macro.

    Full fine-tuning: encoder is unfrozen and updated with a lower LR
    (FINETUNE_LR) while the classification head uses HEAD_LR.
    This allows the pre-trained representations to adapt to the
    downstream classification task.
    """
    num_classes  = cfg["num_classes"]
    hidden_dim   = HIDDEN_DIM

    # ── Subsample labeled data ──
    all_indices = list(range(len(full_train_ds)))
    if budget < 1.0:
        # Stratified subsample
        all_labels = [int(full_train_ds[i][1]) for i in all_indices]
        sub_idx, _ = train_test_split(
            all_indices,
            train_size=budget,
            stratify=all_labels,
            random_state=42,
        )
    else:
        sub_idx = all_indices

    sub_loader = DataLoader(
        Subset(full_train_ds, sub_idx),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # ── Full fine-tuning: encoder + head jointly ──
    # Save a fresh copy of the pretrained encoder for each budget run
    ft_encoder = copy.deepcopy(encoder).to(device)

    head = nn.Linear(hidden_dim, num_classes).to(device)

    # Differential learning rates: lower for encoder, higher for head
    optimizer = optim.Adam([
        {"params": ft_encoder.parameters(), "lr": FINETUNE_LR},
        {"params": head.parameters(),       "lr": HEAD_LR},
    ], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        ft_encoder.train()
        head.train()
        for x, y in sub_loader:
            x, y = x.to(device), y.to(device)
            features = ft_encoder(x)          # encoder is trainable
            logits = head(features)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        ft_encoder.eval()
        head.eval()
        preds_v, labels_v = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                feat = ft_encoder(x)
                p = head(feat).argmax(1).cpu().numpy()
                preds_v.extend(p)
                labels_v.extend(y.numpy())

        val_f1 = f1_score(labels_v, preds_v, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "encoder": ft_encoder.state_dict(),
                "head":    head.state_dict(),
            }, f"/tmp/finetune_best_{int(budget*100)}.pt")

    # ── Test evaluation ──
    ckpt = torch.load(f"/tmp/finetune_best_{int(budget*100)}.pt", map_location=device, weights_only=True)
    ft_encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    ft_encoder.eval()
    head.eval()
    preds_t, labels_t = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            feat = ft_encoder(x)
            p = head(feat).argmax(1).cpu().numpy()
            preds_t.extend(p)
            labels_t.extend(y.numpy())

    test_f1  = f1_score(labels_t, preds_t, average="macro")
    test_acc = balanced_accuracy_score(labels_t, preds_t)

    return test_f1, test_acc, labels_t, preds_t, ft_encoder, head


# ─────────────────────────────────────────────
# Label budget sweep
# ─────────────────────────────────────────────

def label_budget_sweep(encoder, full_train_ds, val_loader, test_loader, cfg, device):
    """
    Train linear probe for each label budget and print comparison table.
    This is the core experiment of the project.

    Returns ``(results, full_budget_encoder, full_budget_head)`` — the fine-tuned
    encoder+head from the 100%-label-budget run, which is what gets exported as
    the deployable Model Bundle (see ADR 0002; the sweep itself stays MLflow-only).
    """
    cfg_mode = DATASET_CONFIG

    print("=" * 60)
    print("PHASE 2: LABEL BUDGET SWEEP (full fine-tuning)")
    print("=" * 60)
    print(f"  Budgets: {[f'{int(b*100)}%' for b in LABEL_BUDGETS]}")
    print()

    results = {}
    full_budget_encoder, full_budget_head = None, None

    for budget in LABEL_BUDGETS:
        pct = int(budget * 100)
        print(f"  ── {pct}% labels ──")
        f1, acc, labels, preds, ft_encoder, head = train_probe_for_budget(
            encoder, full_train_ds, val_loader, test_loader,
            budget=budget, cfg=cfg_mode, device=device
        )
        results[pct] = {"f1_macro": f1, "balanced_acc": acc}
        mlflow.log_metric(f"ssl_probe_f1_macro_{pct}pct", f1)
        mlflow.log_metric(f"ssl_probe_balanced_acc_{pct}pct", acc)
        print(f"     F1 Macro: {f1:.4f}  |  Balanced Acc: {acc:.4f}")

        if budget == 1.0:
            full_budget_encoder, full_budget_head = ft_encoder, head

        # Full classification report for 100% budget
        if budget == 1.0:
            label_names = cfg_mode["label_names"]
            report = classification_report(labels, preds, target_names=label_names)
            print(f"\n  Classification report (100% labels):\n{report}")
            with open("/tmp/ssl_report_100pct.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("/tmp/ssl_report_100pct.txt")

    # ── Summary table ──
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Budget':>8} | {'F1 Macro':>10} | {'Bal. Acc':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")

    # Supervised baselines from CP3 for comparison
    baselines = {
        "RF (CP2)":       0.955,
        "CNN-LSTM (CP3)": 0.927,
        "Transformer (CP3)": 0.920,
    }

    for pct, r in results.items():
        goal = " ✓ GOAL MET" if pct == 1 and r["f1_macro"] >= 0.925 else ""
        print(f"  SSL {pct:>3}%  | {r['f1_macro']:>10.4f} | {r['balanced_acc']:>10.4f}{goal}")

    print()
    print("  Supervised baselines (CP3):")
    for name, f1 in baselines.items():
        print(f"    {name}: F1 = {f1:.3f}")

    print()
    print("  CP2 goal: SSL + 1% labels ≥ 0.925")
    achieved = results.get(1, {}).get("f1_macro", 0) >= 0.925
    print(f"  Goal achieved: {'✓ YES' if achieved else '✗ NO (discuss why)'}")

    return results, full_budget_encoder, full_budget_head


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def run_ssl():
    set_seed(42)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    cfg_mode = DATASET_CONFIG

    print(f"\nDataset : merged HAR")
    print(f"Device  : {device}")
    print(f"Classes : {cfg_mode['num_classes']}\n")

    # ── Data ──
    train_loader, val_loader, test_loader, full_train_ds = load_merged(BATCH_SIZE)

    # ── MLflow ──
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(repo_root, 'mlflow.db')}")
    mlflow.set_experiment("HAR_SSL_data2vec")

    with mlflow.start_run(run_name="data2vec_merged"):
        mlflow.log_params({
            "dataset":          "merged",
            "mask_ratio":       MASK_RATIO,
            "ema_decay":        EMA_DECAY,
            "hidden_dim":       HIDDEN_DIM,
            "pretrain_epochs":  PRETRAIN_EPOCHS,
            "finetune_epochs":  FINETUNE_EPOCHS,
            "finetune_lr":      FINETUNE_LR,
            "batch_size":       BATCH_SIZE,
            "window_size":      WINDOW_SIZE,
            "label_budgets":    str([f"{int(b*100)}%" for b in LABEL_BUDGETS]),
        })

        # ── Build model ──
        encoder = SSLEncoder(
            in_channels=DATASET_CONFIG["in_channels"],
            hidden_dim=HIDDEN_DIM,
        )
        model = Data2Vec(encoder, hidden_dim=HIDDEN_DIM).to(device)

        # ── Phase 1: Pre-train ──
        trained_encoder = pretrain(model, train_loader, device)

        # ── Phase 2: Label budget sweep ──
        results, full_encoder, full_head = label_budget_sweep(
            trained_encoder, full_train_ds, val_loader, test_loader, cfg_mode, device
        )

        # ── Log encoder ──
        mlflow.pytorch.log_model(trained_encoder, "ssl_encoder")

        # ── Export Model Bundle (100%-label-budget fine-tune; ADR 0002) ──
        from server.bundle import save_bundle

        save_bundle(
            model=SSLClassifier(full_encoder, full_head),
            arch={
                "type": "ssl_encoder",
                "in_channels": DATASET_CONFIG["in_channels"],
                "hidden_dim": HIDDEN_DIM,
                "num_classes": DATASET_CONFIG["num_classes"],
            },
            norm_mean=full_train_ds.norm_mean,
            norm_std=full_train_ds.norm_std,
            label_order=DATASET_CONFIG["label_names"],
            models_dir=os.path.join(repo_root, "models"),
            id="ssl_data2vec",
            display_name="Data2Vec (SSL-pretrained)",
            description="CNN encoder pretrained with data2vec self-distillation, fine-tuned on 100% of labels.",
            ssl_pretrained=True,
            is_default=False,
        )

    print("\n✓ Done. Run `uv run mlflow ui` to inspect results.")


if __name__ == "__main__":
    run_ssl()
