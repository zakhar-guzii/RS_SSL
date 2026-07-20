"""Seam 1 — Recording -> Canonical Signal Windows.

Transforms a raw Recording (irregularly-sampled accelerometer + gyroscope from
a phone) into the Canonical Signal Windows the models were trained on. The
transform must mirror the merge pipeline in ``src/data_merge.py`` exactly, or
Windows will not match training and Predictions will be garbage.

The Canonical Signal is 6-channel ``[ax, ay, az, gx, gy, gz]`` — total
acceleration in g plus angular velocity in rad/s. Only the accelerometer
channels are unit-converted (``g`` / ``m/s2``); gyroscope is always rad/s.

This function is bundle-agnostic: it produces pre-normalization Windows.
Per-Model-Bundle normalization is applied downstream.
"""

from typing import Sequence

import numpy as np

from canonical import gravity_project

WINDOW = 128
STEP = 64
TARGET_HZ = 50
GAP_FACTOR = 5  # split segments where interval > GAP_FACTOR x median
N_CHANNELS = 6  # [ax, ay, az, gx, gy, gz]
G = 9.80665  # m/s^2 per g


def _resample_interp(vals: np.ndarray, times_ns: np.ndarray) -> np.ndarray:
    """Linearly interpolate (N, C) vals from irregular timestamps to 50 Hz."""
    dt_ns = int(1e9 / TARGET_HZ)
    t_out = np.arange(times_ns[0], times_ns[-1], dt_ns)
    c = vals.shape[1]
    if len(t_out) < 2:
        return np.empty((0, c), dtype=np.float32)
    return np.stack(
        [np.interp(t_out, times_ns, vals[:, ch]) for ch in range(c)],
        axis=1,
    ).astype(np.float32)


def _sliding_windows(arr: np.ndarray) -> np.ndarray:
    """(N, C) -> (n_wins, WINDOW, C), step STEP."""
    c = arr.shape[1]
    n_wins = max(0, (len(arr) - WINDOW) // STEP + 1)
    if n_wins == 0:
        return np.empty((0, WINDOW, c), dtype=np.float32)
    return np.stack([arr[i * STEP: i * STEP + WINDOW] for i in range(n_wins)])


def to_canonical_windows(samples: Sequence[Sequence[float]], units: str) -> np.ndarray:
    """Raw Recording -> Canonical Signal Windows (n_windows, WINDOW, 6) float32.

    ``samples`` is a list of ``[t_ns, ax, ay, az, gx, gy, gz]`` rows. ``units``
    is the accelerometer unit (``"g"`` or ``"m/s2"``); gyroscope is always
    rad/s and never converted. Output is 6-channel Canonical Signal resampled to
    50 Hz, cut into 128-sample Windows (pre-normalization).
    """
    arr = np.asarray(samples, dtype=np.float64)
    times = arr[:, 0].astype(np.int64)
    vals = arr[:, 1:1 + N_CHANNELS].astype(np.float32)

    if units == "m/s2":
        vals[:, 0:3] = vals[:, 0:3] / G  # accelerometer channels only

    order = np.argsort(times, kind="stable")
    times = times[order]
    vals = vals[order]

    diffs = np.diff(times)
    pos_mask = diffs > 0
    if pos_mask.sum() == 0:
        return np.empty((0, WINDOW, N_CHANNELS), dtype=np.float32)

    # Split into continuous segments at gaps larger than 5x the median interval,
    # so a discontinuous capture is never interpolated across (matches merge).
    med_dt = float(np.median(diffs[pos_mask]))
    split_pts = np.where(diffs > med_dt * GAP_FACTOR)[0] + 1
    segments = np.split(np.arange(len(times)), split_pts)

    all_wins = []
    for idx in segments:
        if len(idx) < WINDOW:
            continue
        resampled = _resample_interp(vals[idx], times[idx])
        wins = _sliding_windows(resampled)
        if len(wins):
            all_wins.append(wins)

    if not all_wins:
        return np.empty((0, WINDOW, N_CHANNELS), dtype=np.float32)
    # Project onto the per-window gravity direction — the exact transform
    # data_merge.py applies to training data. 6 raw device-frame channels ->
    # 4 orientation-invariant ones, so the Prediction no longer depends on how
    # the phone was held.
    return gravity_project(np.concatenate(all_wins))
