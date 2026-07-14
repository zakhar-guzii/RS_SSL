"""No-retrain fix for the sit/stand serving collapse.

The bundles were trained on per-source z-normalized data but ship *identity*
norm stats (mean≈0, std≈1), so the serving path applied no normalization and
static poses (sit/stand) collapsed. The models themselves are fine — feeding
correctly-normalized windows recovers sit/stand at ~0.95 accuracy.

This script computes the **raw-domain** per-channel mean/std of the Canonical
Signal directly from the local raw UCI HAR data (total_acc in g + body_gyro in
rad/s — waist-mounted, a fixed device position) and writes them into every
bundle's meta.json. Serving (src/server/predictor.py) then applies a real
normalization. The previous stats are backed up to meta.json.bak.

Run::

    python scripts/set_bundle_norm_from_uci.py            # write stats
    python scripts/set_bundle_norm_from_uci.py --dry-run  # just print them

For the fully-correct multi-source normalization, restore the raw MotionSense +
HHAR datasets and re-run ``python src/data_merge.py`` (which now persists global
raw-domain stats) followed by a retrain.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UCI = ROOT / "data" / "uci_har"
MODELS = ROOT / "models"
UCI_CANON = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING"}


def _stack3(split: str, prefix: str) -> np.ndarray:
    sig = UCI / split / "Inertial Signals"
    return np.stack([np.loadtxt(sig / f"{prefix}_{a}_{split}.txt", dtype=np.float32)
                     for a in ("x", "y", "z")], axis=2)


def uci_raw_stats() -> tuple[list[float], list[float]]:
    """Per-channel mean/std of raw Canonical Signal windows from UCI HAR."""
    labels = {int(i): n for i, n in
              (line.split() for line in (UCI / "activity_labels.txt").read_text().splitlines())}
    parts = []
    for split in ("train", "test"):
        X = np.concatenate([_stack3(split, "total_acc"), _stack3(split, "body_gyro")], axis=2)
        yid = np.loadtxt(UCI / split / f"y_{split}.txt", dtype=int)
        keep = np.array([labels[i] in UCI_CANON for i in yid])
        parts.append(X[keep])
    X = np.concatenate(parts)  # (N, 128, 6)
    flat = X.reshape(-1, 6)
    mean = flat.mean(0).round(6)
    std = (flat.std(0) + 1e-8).round(6)
    return mean.tolist(), std.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mean, std = uci_raw_stats()
    print(f"UCI raw-domain norm_mean [ax,ay,az,gx,gy,gz]:\n  {mean}")
    print(f"UCI raw-domain norm_std  [ax,ay,az,gx,gy,gz]:\n  {std}")
    if args.dry_run:
        return 0

    metas = sorted(MODELS.glob("*/meta.json"))
    if not metas:
        print(f"No bundles under {MODELS}", file=__import__("sys").stderr)
        return 2
    for mp in metas:
        if not (mp.parent / "meta.json.bak").exists():
            shutil.copy2(mp, mp.parent / "meta.json.bak")
        meta = json.loads(mp.read_text())
        old0 = float(np.ravel(meta.get("norm_mean", [0]))[0])
        meta["norm_mean"] = mean
        meta["norm_std"] = std
        mp.write_text(json.dumps(meta, indent=2))
        print(f"  updated {mp.parent.name:14s} (ax mean {old0:+.4f} -> {mean[0]:+.4f})")
    print("\nDone. Verify with:  python scripts/verify_sit_stand.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
