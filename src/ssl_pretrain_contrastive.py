"""
SimCLR/TS-TCC-inspired contrastive Self-Supervised Learning for HAR time-series.

Core idea (simplified from SimCLR https://arxiv.org/abs/2002.05709 and
TS-TCC https://arxiv.org/abs/2106.14112):
  1. Two augmented "views" are generated per window (random crop, segment
     permutation, jitter, magnitude scaling) — time-series analogues of
     SimCLR's random-resized-crop/color-jitter for images.
  2. A single shared-weight encoder + projection head embeds both views.
  3. Loss = NT-Xent (normalized temperature-scaled cross-entropy / InfoNCE):
     the two views of the same window are pulled together, every other
     sample in the batch (both its views) is pushed apart as a negative.
  4. No labels used during pre-training.

Unlike data2vec and the DINOv2-style script, this method has NO teacher
network, EMA, or stop-gradient — it's a single network trained end-to-end
with negatives supplied by the rest of the batch. This is the core
structural difference between the InfoNCE family (this script) and the
self-distillation family (data2vec, DINOv2, BYOL).

Reuses the encoder, data pipeline, and label-budget evaluation harness from
ssl_pretrain.py (data2vec) so all SSL methods are directly comparable.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ssl_pretrain import (
    DATASET_CONFIG,
    SSLEncoder,
    load_merged,
    set_seed,
    label_budget_sweep,
    BATCH_SIZE,
    HIDDEN_DIM,
    PATIENCE,
)

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

WINDOW_SIZE      = 128

# Augmentation (two views per window)
CROP_SCALE       = (0.7, 1.0)   # fraction of window length kept after crop
JITTER_STD       = 0.05
SCALE_JITTER     = 0.1
NUM_SEGMENTS     = 8            # for segment-permutation augmentation
PERMUTE_PROB     = 0.5

# Projection head
PROJ_DIM         = 128
PROJ_HIDDEN_DIM  = 256

# NT-Xent (InfoNCE)
TEMPERATURE      = 0.2

PRETRAIN_EPOCHS  = 30
PRETRAIN_LR      = 1e-3


# ─────────────────────────────────────────────
# Time-series augmentations
# ─────────────────────────────────────────────

def _random_crop_batch(x, length):
    B, T, C = x.shape
    length = min(length, T)
    starts = torch.randint(0, T - length + 1, (B,), device=x.device)
    offsets = torch.arange(length, device=x.device)
    idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(x, 1, idx)


def _jitter_and_scale(x):
    x = x + torch.randn_like(x) * JITTER_STD
    scale = 1.0 + (torch.rand(x.size(0), 1, x.size(2), device=x.device) * 2 - 1) * SCALE_JITTER
    return x * scale


def _permute_segments(x, num_segments=NUM_SEGMENTS):
    """Split the window into equal segments and shuffle their order per-sample."""
    B, T, C = x.shape
    seg_len = T // num_segments
    trimmed = seg_len * num_segments
    segments = x[:, :trimmed, :].reshape(B, num_segments, seg_len, C)

    perms = torch.stack([torch.randperm(num_segments, device=x.device) for _ in range(B)])
    perms = perms.view(B, num_segments, 1, 1).expand(-1, -1, seg_len, C)
    segments = torch.gather(segments, 1, perms).reshape(B, trimmed, C)

    if trimmed < T:
        segments = torch.cat([segments, x[:, trimmed:, :]], dim=1)
    return segments


def augment_view(x, crop_scale=CROP_SCALE):
    if random.random() < PERMUTE_PROB:
        x = _permute_segments(x)
    T = x.shape[1]
    length = random.randint(int(crop_scale[0] * T), int(crop_scale[1] * T))
    x = _random_crop_batch(x, length)
    return _jitter_and_scale(x)


def two_views(x):
    return augment_view(x), augment_view(x)


# ─────────────────────────────────────────────
# Projection head + NT-Xent loss
# ─────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=PROJ_HIDDEN_DIM, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    """
    Normalized temperature-scaled cross-entropy (InfoNCE), SimCLR-style.
    For each view, its positive is the other view of the same window;
    every other sample in the batch (both views) is a negative.
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)   # (2B, D)

    sim = torch.matmul(z, z.T) / temperature              # (2B, 2B)
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))

    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# ─────────────────────────────────────────────
# SimCLR model
# ─────────────────────────────────────────────

class SimCLR(nn.Module):
    """Single shared-weight encoder + projector. No teacher, no EMA."""
    def __init__(self, encoder: SSLEncoder, hidden_dim=HIDDEN_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(hidden_dim, out_dim=proj_dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


# ─────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────

def pretrain(model: SimCLR, train_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    best_loss = float("inf")
    no_improve = 0

    print("=" * 60)
    print("PHASE 1: SSL PRE-TRAINING (SimCLR/TS-TCC-style contrastive)")
    print(f"  Crop scale : {CROP_SCALE}  |  Temperature: {TEMPERATURE}")
    print(f"  Epochs     : {PRETRAIN_EPOCHS}  |  Patience: {PATIENCE}")
    print("=" * 60)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for x, _ in tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False):
            x = x.to(device)
            x1, x2 = two_views(x)
            z1, z2 = model(x1, x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        mlflow.log_metric("pretrain_loss", avg, step=epoch)
        print(f"  Epoch {epoch:02d} | loss: {avg:.4f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            no_improve = 0
            torch.save(model.encoder.state_dict(), "/tmp/simclr_encoder_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.encoder.load_state_dict(
        torch.load("/tmp/simclr_encoder_best.pt", map_location=device, weights_only=True)
    )
    print(f"\n  ✓ Pre-training done. Best loss: {best_loss:.4f}\n")
    return model.encoder


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
    mlflow.set_experiment("HAR_SSL_contrastive")

    with mlflow.start_run(run_name="simclr_merged"):
        mlflow.log_params({
            "dataset":         "merged",
            "crop_scale":      str(CROP_SCALE),
            "num_segments":    NUM_SEGMENTS,
            "permute_prob":    PERMUTE_PROB,
            "temperature":     TEMPERATURE,
            "proj_dim":        PROJ_DIM,
            "hidden_dim":      HIDDEN_DIM,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "batch_size":      BATCH_SIZE,
            "window_size":     WINDOW_SIZE,
        })

        # ── Build model ──
        encoder = SSLEncoder(
            in_channels=DATASET_CONFIG["in_channels"],
            hidden_dim=HIDDEN_DIM,
        )
        model = SimCLR(encoder, hidden_dim=HIDDEN_DIM).to(device)

        # ── Phase 1: Pre-train ──
        trained_encoder = pretrain(model, train_loader, device)

        # ── Phase 2: Label budget sweep (shared with data2vec / DINOv2) ──
        results = label_budget_sweep(
            trained_encoder, full_train_ds, val_loader, test_loader, cfg_mode, device
        )

        # ── Log encoder ──
        mlflow.pytorch.log_model(trained_encoder, "ssl_encoder")

    print("\n✓ Done. Run `uv run mlflow ui` to inspect results.")


if __name__ == "__main__":
    run_ssl()
