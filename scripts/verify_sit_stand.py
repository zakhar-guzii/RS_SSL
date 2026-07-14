"""End-to-end check that the normalization-seam fix restored the static poses.

Builds real *sitting*, *standing* and *walking* Recordings from the UCI HAR test
inertial signals (total acceleration in g + gyroscope in rad/s — the Canonical
Signal form, gravity included), pushes each through the live serving path
(``Predictor.predict``, the exact code ``POST /predict`` calls), and asserts the
predicted Activity matches.

This is the regression guard for the bug where every ``sit`` Recording was
classified ``stand`` because training-time normalization stats were discarded and
never reproduced at inference. Run it after re-merging + re-exporting bundles::

    python scripts/verify_sit_stand.py                # default bundle
    python scripts/verify_sit_stand.py --model transformer

Requires only ``data/uci_har/`` (no MotionSense/HHAR) and at least one bundle
under ``models/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from server.bundle import load_bundle, load_bundles  # noqa: E402

UCI = ROOT / "data" / "uci_har" / "test"
SIGNALS = UCI / "Inertial Signals"
DT_NS = int(1e9 / 50)  # 50 Hz sample spacing in nanoseconds

# UCI activity id -> canonical label we expect back from the Backend.
UCI_TO_CANONICAL = {1: "walk", 2: "upstairs", 3: "downstairs", 4: "sit", 5: "stand"}
MAX_PER_CLASS = 150  # cap windows per class for a quick but meaningful check


def _load_windows(prefix: str) -> np.ndarray:
    """Load a UCI inertial-signal file as (n_windows, 128) float array."""
    return np.loadtxt(SIGNALS / f"{prefix}_test.txt", dtype=np.float32)


def build_recording(win_idx: int, channels: dict[str, np.ndarray]) -> list[list[float]]:
    """One genuine UCI 128-sample window -> a raw ``/predict`` samples list.

    Rows are ``[t_ns, ax, ay, az, gx, gy, gz]`` — total acceleration in g
    (gravity included) + angular velocity in rad/s, spaced at 50 Hz. A 129th row
    (last sample duplicated) is appended so the endpoint-exclusive 50 Hz resample
    grid yields exactly one 128-sample Window from this single real window — no
    concatenation of discontinuous windows (which would inject resampling
    artifacts and misrepresent the serving path).
    """
    ax = channels["total_acc_x"][win_idx]
    ay = channels["total_acc_y"][win_idx]
    az = channels["total_acc_z"][win_idx]
    gx = channels["body_gyro_x"][win_idx]
    gy = channels["body_gyro_y"][win_idx]
    gz = channels["body_gyro_z"][win_idx]
    rows = [
        [i * DT_NS, float(ax[i]), float(ay[i]), float(az[i]),
         float(gx[i]), float(gy[i]), float(gz[i])]
        for i in range(len(ax))
    ]
    rows.append([len(ax) * DT_NS] + rows[-1][1:])  # duplicate last sample
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="bundle id under models/ (default: the is_default bundle)")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if args.model:
        bundle = load_bundle(models_dir / args.model)
    else:
        bundles = load_bundles(models_dir)
        if not bundles:
            print(f"No bundles found under {models_dir}", file=sys.stderr)
            return 2
        bundle = next(b for b in bundles.values() if b.is_default)
    print(f"Model bundle : {bundle.id} ({bundle.display_name})")
    print(f"norm_mean    : {np.ravel(bundle.predictor.norm_mean)}")
    if float(np.abs(bundle.predictor.norm_mean).max()) < 0.05:
        print("  WARNING: norm_mean is ~0 — bundle still ships the identity stats; "
              "re-export bundles after re-merging (the seam is not fixed).")

    y = np.loadtxt(UCI / "y_test.txt", dtype=int)
    channels = {p: _load_windows(p) for p in (
        "total_acc_x", "total_acc_y", "total_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
    )}

    overall_correct = overall_total = 0
    worst = 1.0
    for uci_id, expected in UCI_TO_CANONICAL.items():
        idxs = np.where(y == uci_id)[0][:MAX_PER_CLASS]
        if len(idxs) == 0:
            print(f"  (no UCI windows for id {uci_id}) — skipped")
            continue
        preds = [bundle.predictor.predict(build_recording(int(i), channels), units="g")["activity"]
                 for i in idxs]
        correct = sum(p == expected for p in preds)
        acc = correct / len(idxs)
        worst = min(worst, acc)
        overall_correct += correct
        overall_total += len(idxs)
        # top confusions
        conf = {}
        for p in preds:
            if p != expected:
                conf[p] = conf.get(p, 0) + 1
        conf_str = "  ".join(f"{k}:{v}" for k, v in sorted(conf.items(), key=lambda t: -t[1]))
        mark = "OK " if acc >= 0.70 else "LOW"
        print(f"  [{mark}] {expected:<10} acc={acc:.3f} ({correct}/{len(idxs)})"
              + (f"   confused-> {conf_str}" if conf_str else ""))

    overall = overall_correct / max(1, overall_total)
    print(f"\noverall acc = {overall:.3f}   worst class = {worst:.3f}")
    if worst < 0.70:
        print("A class is still failing — normalization is not correct yet.")
        return 1
    print("All classes classified correctly through the serving path — seam is fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
