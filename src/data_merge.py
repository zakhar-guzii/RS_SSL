"""Merge UCI HAR + MotionSense + HHAR into one 6-channel HAR dataset.

Canonical Signal = ``[ax, ay, az, gx, gy, gz]`` — total acceleration in **g**
(gravity included) plus angular velocity in **rad/s** — resampled to 50 Hz and
cut into 128-sample / 50%-overlap Windows.

WISDM is intentionally excluded: it is accelerometer-only and cannot provide the
gyroscope channels (see ``docs/adr/0003-canonical-signal-is-6-channel-accel-gyro.md``).

This module is the single source of truth for the Canonical Signal transform.
``src/server/preprocessing.py`` mirrors the windowing here for inference, and
``tests/test_preprocessing.py`` imports these functions as its parity oracle, so
they must not drift.

Run as a script to (re)generate ``data/merged/har_merged.npz``::

    python src/data_merge.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Canonical Signal constants
# --------------------------------------------------------------------------
WINDOW = 128
STEP = 64
TARGET_HZ = 50
GAP_FACTOR = 5  # split segments where interval > GAP_FACTOR x median
N_CHANNELS = 6  # [ax, ay, az, gx, gy, gz]

COMMON_ACT = ["walk", "upstairs", "downstairs", "sit", "stand"]
LABEL_ENC = {a: i for i, a in enumerate(sorted(COMMON_ACT))}  # alphabetical 0..4

CANON = {
    "uci": {
        "WALKING": "walk", "WALKING_UPSTAIRS": "upstairs",
        "WALKING_DOWNSTAIRS": "downstairs", "SITTING": "sit", "STANDING": "stand",
    },
    "motionsense": {
        "wlk": "walk", "ups": "upstairs", "dws": "downstairs",
        "sit": "sit", "std": "stand",  # "jog" intentionally dropped
    },
    "hhar": {
        "walk": "walk", "stairsup": "upstairs", "stairsdown": "downstairs",
        "sit": "sit", "stand": "stand",
    },
}

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "merged"


# --------------------------------------------------------------------------
# Generic N-channel windowing helpers (mirrored by server.preprocessing)
# --------------------------------------------------------------------------
def sliding_windows(arr: np.ndarray, window: int = WINDOW, step: int = STEP) -> np.ndarray:
    """(N, C) -> (n_wins, window, C), step ``step`` (empty (0, window, C) if none)."""
    c = arr.shape[1] if arr.ndim == 2 else N_CHANNELS
    n_wins = max(0, (len(arr) - window) // step + 1)
    if n_wins == 0:
        return np.empty((0, window, c), dtype=np.float32)
    return np.stack([arr[i * step: i * step + window] for i in range(n_wins)])


def resample_interp(vals: np.ndarray, times_ns: np.ndarray,
                    target_hz: int = TARGET_HZ) -> np.ndarray:
    """Linearly interpolate (N, C) vals from irregular ns timestamps to uniform target_hz."""
    dt_ns = int(1e9 / target_hz)
    t_out = np.arange(times_ns[0], times_ns[-1], dt_ns)
    c = vals.shape[1]
    if len(t_out) < 2:
        return np.empty((0, c), dtype=np.float32)
    return np.stack(
        [np.interp(t_out, times_ns, vals[:, ch]) for ch in range(c)],
        axis=1,
    ).astype(np.float32)


def _segment_indices(times: np.ndarray, gap_factor: int = GAP_FACTOR) -> List[np.ndarray]:
    """Index groups of ``times`` split at gaps > gap_factor x median inter-sample interval."""
    diffs = np.diff(times)
    pos_mask = diffs > 0
    if pos_mask.sum() == 0:
        return []
    med_dt = float(np.median(diffs[pos_mask]))
    split_pts = np.where(diffs > med_dt * gap_factor)[0] + 1
    return np.split(np.arange(len(times)), split_pts)


def windows_from_ts(df: pd.DataFrame, time_col: str, val_cols: Sequence[str],
                    gap_factor: int = GAP_FACTOR) -> np.ndarray:
    """Single-stream: ns time column + C value columns -> (n_windows, WINDOW, C).

    Sort -> gap-split -> resample each segment to 50 Hz -> sliding window.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col].values.astype(np.int64)
    vals = df[list(val_cols)].values.astype(np.float32)
    c = vals.shape[1]

    all_wins = []
    for idx in _segment_indices(times, gap_factor):
        if len(idx) < WINDOW:
            continue
        seg = resample_interp(vals[idx], times[idx])
        wins = sliding_windows(seg)
        if len(wins):
            all_wins.append(wins)
    return np.concatenate(all_wins) if all_wins else np.empty((0, WINDOW, c), dtype=np.float32)


def windows_from_two_streams(
    acc_times: np.ndarray, acc_vals: np.ndarray,
    gyro_times: np.ndarray, gyro_vals: np.ndarray,
    gap_factor: int = GAP_FACTOR,
) -> np.ndarray:
    """Two independently-timestamped ns streams (accel + gyro) -> (n_win, WINDOW, 6).

    Used for HHAR, whose accelerometer and gyroscope live in separate files with
    their own clocks. The accelerometer defines the 50 Hz time grid (gap-split on
    its timestamps); gyroscope is linearly interpolated onto that same grid. Only
    the time span both sensors cover is used, so a Window is never built from
    extrapolated gyroscope.
    """
    order_a = np.argsort(acc_times, kind="stable")
    acc_times, acc_vals = acc_times[order_a], acc_vals[order_a]
    order_g = np.argsort(gyro_times, kind="stable")
    gyro_times, gyro_vals = gyro_times[order_g], gyro_vals[order_g]

    lo = max(acc_times[0], gyro_times[0])
    hi = min(acc_times[-1], gyro_times[-1])
    if hi - lo < int(1e9 / TARGET_HZ) * WINDOW:
        return np.empty((0, WINDOW, N_CHANNELS), dtype=np.float32)

    keep = (acc_times >= lo) & (acc_times <= hi)
    acc_times, acc_vals = acc_times[keep], acc_vals[keep]

    all_wins = []
    for idx in _segment_indices(acc_times, gap_factor):
        if len(idx) < WINDOW:
            continue
        acc_seg = resample_interp(acc_vals[idx], acc_times[idx])  # (M, 3)
        if len(acc_seg) < WINDOW:
            continue
        dt_ns = int(1e9 / TARGET_HZ)
        grid = np.arange(acc_times[idx][0], acc_times[idx][-1], dt_ns)[: len(acc_seg)]
        gyro_seg = np.stack(
            [np.interp(grid, gyro_times, gyro_vals[:, ch]) for ch in range(3)], axis=1
        ).astype(np.float32)
        six = np.concatenate([acc_seg, gyro_seg], axis=1)  # (M, 6)
        wins = sliding_windows(six)
        if len(wins):
            all_wins.append(wins)
    return np.concatenate(all_wins) if all_wins else np.empty((0, WINDOW, N_CHANNELS), dtype=np.float32)


# --------------------------------------------------------------------------
# 1. UCI HAR — already windowed at 128 / 50 Hz; total_acc (g) + body_gyro (rad/s)
# --------------------------------------------------------------------------
def load_uci() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    act_labels = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+",
                             header=None, names=["id", "activity"])
    id_to_name = dict(zip(act_labels["id"], act_labels["activity"]))
    canon = CANON["uci"]

    def load_split(split: str):
        sig = DATA_DIR / split / "Inertial Signals"

        def stack3(prefix: str) -> np.ndarray:
            return np.stack([
                np.loadtxt(sig / f"{prefix}_{ax}_{split}.txt", dtype=np.float32)
                for ax in ("x", "y", "z")
            ], axis=2)  # (N, 128, 3)

        acc = stack3("total_acc")
        gyro = stack3("body_gyro")
        X = np.concatenate([acc, gyro], axis=2)  # (N, 128, 6)
        y_ids = pd.read_csv(DATA_DIR / split / f"y_{split}.txt", header=None)[0].values
        subs = pd.read_csv(DATA_DIR / split / f"subject_{split}.txt", header=None)[0].values
        acts = np.array([id_to_name[i] for i in y_ids])
        return X, acts, subs.astype(str)

    Xs, acts, subs = zip(*(load_split(s) for s in ("train", "test")))
    X = np.concatenate(Xs)
    act = np.concatenate(acts)
    sub = np.concatenate(subs)

    keep = np.isin(act, list(canon.keys()))
    X, act, sub = X[keep], np.array([canon[a] for a in act[keep]]), sub[keep]
    print(f"UCI HAR   : {X.shape}  |  {len(np.unique(act))} activities  |  {len(np.unique(sub))} subjects")
    return X, act, sub


# --------------------------------------------------------------------------
# 2. MotionSense — 50 Hz DeviceMotion CSV; accel = userAcceleration + gravity,
#    gyro = rotationRate (rad/s). Accel & gyro share every row.
# --------------------------------------------------------------------------
def load_motionsense() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = DATA_DIR / "A_DeviceMotion_data" / "A_DeviceMotion_data"
    canon = CANON["motionsense"]
    all_X, all_act, all_sub = [], [], []

    for act_dir in sorted(root.iterdir()):
        if not act_dir.is_dir() or act_dir.name.startswith("__"):
            continue
        code = act_dir.name.rsplit("_", 1)[0]  # "dws_11" -> "dws"
        if code not in canon:
            continue
        label = canon[code]
        for csv_file in sorted(act_dir.glob("sub_*.csv")):
            subj = "".join(ch for ch in csv_file.stem if ch.isdigit())
            df = pd.read_csv(csv_file)
            ax = (df["userAcceleration.x"] + df["gravity.x"]).values
            ay = (df["userAcceleration.y"] + df["gravity.y"]).values
            az = (df["userAcceleration.z"] + df["gravity.z"]).values
            gx, gy, gz = df["rotationRate.x"].values, df["rotationRate.y"].values, df["rotationRate.z"].values
            arr = np.stack([ax, ay, az, gx, gy, gz], axis=1).astype(np.float32)
            wins = sliding_windows(arr)  # already 50 Hz, contiguous
            if len(wins):
                all_X.append(wins)
                all_act.extend([label] * len(wins))
                all_sub.extend([subj] * len(wins))

    X = np.concatenate(all_X)
    act, sub = np.array(all_act), np.array(all_sub)
    print(f"MotionSense: {X.shape}  |  {len(np.unique(act))} activities  |  {len(np.unique(sub))} subjects")
    return X, act, sub


# --------------------------------------------------------------------------
# 3. HHAR — Phones accel & gyro in separate CSVs (Arrival_Time in ms), 100-200 Hz.
#    Aligned per (User, Device, activity) onto a shared 50 Hz grid.
# --------------------------------------------------------------------------
def _read_hhar_csv(zip_path: Path, inner: str) -> pd.DataFrame:
    canon = CANON["hhar"]
    cols = ["Arrival_Time", "x", "y", "z", "User", "Device", "gt"]
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as fh:
        df = pd.read_csv(fh, usecols=cols)
    df = df[df["gt"].isin(canon.keys())].copy()
    df["time_ns"] = df["Arrival_Time"].astype(np.int64) * 1_000_000
    return df


def load_hhar() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zip_path = DATA_DIR / "heterogeneity+activity+recognition" / "Activity recognition exp.zip"
    canon = CANON["hhar"]
    print("HHAR      : reading accelerometer + gyroscope CSVs (large, please wait)...")
    acc = _read_hhar_csv(zip_path, "Activity recognition exp/Phones_accelerometer.csv")
    gyro = _read_hhar_csv(zip_path, "Activity recognition exp/Phones_gyroscope.csv")

    gyro_groups = {k: g for k, g in gyro.groupby(["User", "Device", "gt"], sort=False)}
    all_X, all_act, all_sub = [], [], []
    for key, ga in acc.groupby(["User", "Device", "gt"], sort=False):
        gg = gyro_groups.get(key)
        if gg is None:
            continue
        wins = windows_from_two_streams(
            ga["time_ns"].values.astype(np.int64), ga[["x", "y", "z"]].values.astype(np.float32),
            gg["time_ns"].values.astype(np.int64), gg[["x", "y", "z"]].values.astype(np.float32),
        )
        if len(wins):
            all_X.append(wins)
            all_act.extend([canon[key[2]]] * len(wins))
            all_sub.extend([str(key[0])] * len(wins))

    X = np.concatenate(all_X)
    act, sub = np.array(all_act), np.array(all_sub)
    print(f"HHAR      : {X.shape}  |  {len(np.unique(act))} activities  |  {len(np.unique(sub))} subjects")
    return X, act, sub


# --------------------------------------------------------------------------
# 4. Merge & save
# --------------------------------------------------------------------------
def merge_and_save(out_path: Path = OUT_DIR / "har_merged.npz") -> None:
    sources = {
        "uci": load_uci(),
        "motionsense": load_motionsense(),
        "hhar": load_hhar(),
    }

    X_parts, act_parts, sub_parts, src_parts = [], [], [], []
    for tag, (Xs, acts, subs) in sources.items():
        # Keep RAW canonical windows (g, gravity included). Normalization is
        # computed once globally below and persisted with the data, so the exact
        # same transform can be reproduced at inference time. See
        # ``src/server/predictor.py`` (applies norm_mean/norm_std to live
        # Recordings) — storing per-source stats and discarding them here was the
        # train/serve seam that made static poses (sit vs stand) unrecoverable.
        X_parts.append(Xs)
        act_parts.append(acts)
        sub_parts.append(np.array([f"{tag}_{s}" for s in subs]))
        src_parts.append(np.full(len(Xs), tag))

    X = np.concatenate(X_parts).astype(np.float32)
    activity = np.concatenate(act_parts)
    subject = np.concatenate(sub_parts)
    source = np.concatenate(src_parts)
    y = np.array([LABEL_ENC[a] for a in activity], dtype=np.int32)

    # Single global z-normalization over the pooled raw windows. One transform
    # (no per-source ambiguity) that a single-device deployment can reproduce.
    # X is saved RAW; MergedHARDataset applies these stats at load time, and the
    # same stats flow into each Model Bundle's meta.json via save_bundle(...).
    norm_mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # (1,1,6)
    norm_std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

    print(f"\nTotal windows : {X.shape}   (channels: [ax,ay,az,gx,gy,gz])")
    print(f"Label encoding: {LABEL_ENC}")
    print(f"norm_mean     : {np.ravel(norm_mean)}")
    print(f"norm_std      : {np.ravel(norm_std)}")
    print(pd.DataFrame({"activity": activity, "source": source})
          .groupby(["source", "activity"]).size().unstack(fill_value=0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), X=X, y=y, activity=activity,
                        subject=subject, source=source,
                        norm_mean=norm_mean, norm_std=norm_std)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    merge_and_save()
