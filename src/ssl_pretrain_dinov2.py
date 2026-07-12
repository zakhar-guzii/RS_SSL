"""
DINOv2-inspired Self-Supervised Learning for HAR time-series.

Core idea (simplified from https://arxiv.org/abs/2304.07193, building on
DINO https://arxiv.org/abs/2104.14294):
  1. Teacher  = EMA copy of student (encoder + projection head); sees a few
              "global" (long) crops of the signal
  2. Student  = encoder + projection head; sees the same global crops AND
              several "local" (short) crops
  3. Loss     = cross-entropy self-distillation between the softened teacher
              output (centered + sharpened) and the student output, summed
              over every student/teacher crop pair that comes from a
              different view
  4. No labels used during pre-training.

Adapted from images to 1D HAR windows: "multi-crop" becomes random temporal
sub-windows of varying length (global = large fraction of the window,
local = small fraction) instead of image regions at different resolutions.
This is a simplified DINOv2 — it keeps the multi-crop self-distillation +
centering core, but skips iBOT patch-level loss, KoLeo regularization and
Sinkhorn-Knopp centering used in the full paper.

Reuses the encoder, data pipeline, and label-budget evaluation harness from
ssl_pretrain.py (data2vec) so the two SSL methods are directly comparable.
"""

import copy
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
    SSLClassifier,
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

WINDOW_SIZE     = 128

# Multi-crop (temporal analogue of DINO's multi-crop image augmentation)
GLOBAL_CROPS    = 2
LOCAL_CROPS     = 4
GLOBAL_SCALE    = (0.6, 1.0)   # fraction of window length
LOCAL_SCALE     = (0.2, 0.5)
JITTER_STD      = 0.05
SCALE_JITTER    = 0.1

# DINO projection head
OUT_DIM         = 4096
BOTTLENECK_DIM  = 256
HEAD_HIDDEN_DIM = 1024

# Self-distillation loss
STUDENT_TEMP               = 0.1
WARMUP_TEACHER_TEMP        = 0.04
TEACHER_TEMP               = 0.07
WARMUP_TEACHER_TEMP_EPOCHS = 5
CENTER_MOMENTUM            = 0.9

# Teacher EMA momentum (cosine schedule, as in DINO)
MOMENTUM_BASE   = 0.996
MOMENTUM_FINAL  = 1.0

PRETRAIN_EPOCHS = 10
PRETRAIN_LR     = 1e-3


# ─────────────────────────────────────────────
# DINO projection head
# ─────────────────────────────────────────────

class DINOHead(nn.Module):
    """
    3-layer MLP → L2-normalize → weight-normalized linear "prototype" layer.
    Maps encoder features into a high-dimensional space where the
    self-distillation cross-entropy loss is computed.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=HEAD_HIDDEN_DIM,
                 bottleneck_dim=BOTTLENECK_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ─────────────────────────────────────────────
# DINO self-distillation loss (centering + sharpening)
# ─────────────────────────────────────────────

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, global_crops=GLOBAL_CROPS,
                 warmup_teacher_temp=WARMUP_TEACHER_TEMP, teacher_temp=TEACHER_TEMP,
                 warmup_teacher_temp_epochs=WARMUP_TEACHER_TEMP_EPOCHS, nepochs=PRETRAIN_EPOCHS,
                 student_temp=STUDENT_TEMP, center_momentum=CENTER_MOMENTUM):
        super().__init__()
        self.ncrops = ncrops
        self.global_crops = global_crops
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.global_crops)

        total_loss, n_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= n_terms

        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# ─────────────────────────────────────────────
# Temporal multi-crop augmentation
# ─────────────────────────────────────────────

def _random_crop_batch(x, length):
    B, T, C = x.shape
    length = min(length, T)
    starts = torch.randint(0, T - length + 1, (B,), device=x.device)
    offsets = torch.arange(length, device=x.device)
    idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(-1).expand(-1, -1, C)
    return torch.gather(x, 1, idx)


def _augment(x):
    x = x + torch.randn_like(x) * JITTER_STD
    scale = 1.0 + (torch.rand(x.size(0), 1, x.size(2), device=x.device) * 2 - 1) * SCALE_JITTER
    return x * scale


def make_multicrop(x, global_crops=GLOBAL_CROPS, local_crops=LOCAL_CROPS,
                    global_scale=GLOBAL_SCALE, local_scale=LOCAL_SCALE):
    """
    Returns a list of crops: the first `global_crops` are long temporal
    windows (seen by both teacher and student), the rest are short local
    windows (seen only by the student).
    """
    T = x.shape[1]
    crops = []
    for _ in range(global_crops):
        length = random.randint(int(global_scale[0] * T), int(global_scale[1] * T))
        crops.append(_augment(_random_crop_batch(x, length)))
    for _ in range(local_crops):
        length = random.randint(int(local_scale[0] * T), int(local_scale[1] * T))
        crops.append(_augment(_random_crop_batch(x, length)))
    return crops


# ─────────────────────────────────────────────
# DINOv2 model
# ─────────────────────────────────────────────

class DinoV2(nn.Module):
    """
    Student (encoder + head) sees global + local crops.
    Teacher (EMA of student) sees only global crops. No labels needed.
    """
    def __init__(self, encoder: SSLEncoder, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM):
        super().__init__()
        self.student_encoder = encoder
        self.student_head = DINOHead(hidden_dim, out_dim)

        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_head = DINOHead(hidden_dim, out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in list(self.teacher_encoder.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum):
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def forward(self, crops, n_global=GLOBAL_CROPS):
        student_out = torch.cat(
            [self.student_head(self.student_encoder(c)) for c in crops], dim=0
        )
        with torch.no_grad():
            teacher_out = torch.cat(
                [self.teacher_head(self.teacher_encoder(c)) for c in crops[:n_global]], dim=0
            )
        return student_out, teacher_out


def momentum_schedule(epoch, total_epochs, base=MOMENTUM_BASE, final=MOMENTUM_FINAL):
    return final - (final - base) * (np.cos(np.pi * epoch / total_epochs) + 1) / 2


# ─────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────

def pretrain(model: DinoV2, train_loader, device):
    ncrops = GLOBAL_CROPS + LOCAL_CROPS
    params = list(model.student_encoder.parameters()) + list(model.student_head.parameters())
    optimizer = optim.AdamW(params, lr=PRETRAIN_LR, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    dino_loss = DINOLoss(out_dim=OUT_DIM, ncrops=ncrops, nepochs=PRETRAIN_EPOCHS).to(device)

    best_loss = float("inf")
    no_improve = 0

    print("=" * 60)
    print("PHASE 1: SSL PRE-TRAINING (DINOv2-style)")
    print(f"  Global crops: {GLOBAL_CROPS} {GLOBAL_SCALE}  |  Local crops: {LOCAL_CROPS} {LOCAL_SCALE}")
    print(f"  Epochs: {PRETRAIN_EPOCHS}  |  Patience: {PATIENCE}")
    print("=" * 60)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        momentum = momentum_schedule(epoch, PRETRAIN_EPOCHS)

        for x, _ in tqdm(train_loader, desc=f"  Epoch {epoch:02d}", leave=False):
            x = x.to(device)
            crops = make_multicrop(x)
            student_out, teacher_out = model(crops)
            loss = dino_loss(student_out, teacher_out, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_teacher(momentum)
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        mlflow.log_metric("pretrain_loss", avg, step=epoch)
        print(f"  Epoch {epoch:02d} | loss: {avg:.4f} | teacher momentum: {momentum:.4f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            no_improve = 0
            torch.save(model.student_encoder.state_dict(), "/tmp/dinov2_encoder_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.student_encoder.load_state_dict(
        torch.load("/tmp/dinov2_encoder_best.pt", map_location=device, weights_only=True)
    )
    print(f"\n  ✓ Pre-training done. Best loss: {best_loss:.4f}\n")
    return model.student_encoder


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
    mlflow.set_experiment("HAR_SSL_dinov2")

    with mlflow.start_run(run_name="dinov2_merged"):
        mlflow.log_params({
            "dataset":         "merged",
            "global_crops":    GLOBAL_CROPS,
            "local_crops":     LOCAL_CROPS,
            "global_scale":    str(GLOBAL_SCALE),
            "local_scale":     str(LOCAL_SCALE),
            "out_dim":         OUT_DIM,
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
        model = DinoV2(encoder, hidden_dim=HIDDEN_DIM).to(device)

        # ── Phase 1: Pre-train ──
        trained_encoder = pretrain(model, train_loader, device)

        # ── Phase 2: Label budget sweep (shared with data2vec) ──
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
            id="ssl_dinov2",
            display_name="DINOv2 (SSL-pretrained)",
            description="CNN encoder pretrained with DINOv2-style self-distillation, fine-tuned on 100% of labels.",
            ssl_pretrained=True,
            is_default=False,
        )

    print("\n✓ Done. Run `uv run mlflow ui` to inspect results.")


if __name__ == "__main__":
    run_ssl()
