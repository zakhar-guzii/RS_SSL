"""Tests for Seam 1: Recording -> Canonical Signal Windows.

The source of truth for parity is the merge pipeline in
``notebooks/har_merge.ipynb``. Its three functions are reproduced verbatim
below as an independent oracle: the code under test
(``server.preprocessing.to_canonical_windows``) is written fresh and must
reproduce the oracle's Windows, so a match is a real agreement between two
independent implementations, not a tautology.
"""

import numpy as np
import pandas as pd

from server.preprocessing import to_canonical_windows

# Canonical Signal constants (mirror the merge notebook).
WINDOW = 128
STEP = 64
TARGET_HZ = 50
G = 9.80665  # m/s^2 per g


# --------------------------------------------------------------------------
# Oracle: reference implementation copied verbatim from har_merge.ipynb.
# Do not "improve" these — they define the training-time Canonical Signal.
# --------------------------------------------------------------------------
def _oracle_sliding_windows(arr, window=WINDOW, step=STEP):
    n_wins = max(0, (len(arr) - window) // step + 1)
    if n_wins == 0:
        return np.empty((0, window, 3), dtype=np.float32)
    return np.stack([arr[i * step: i * step + window] for i in range(n_wins)])


def _oracle_resample_interp(vals, times_ns, target_hz=TARGET_HZ):
    dt_ns = int(1e9 / target_hz)
    t_out = np.arange(times_ns[0], times_ns[-1], dt_ns)
    if len(t_out) < 2:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(
        [np.interp(t_out, times_ns, vals[:, c]) for c in range(vals.shape[1])],
        axis=1,
    ).astype(np.float32)


def _oracle_windows_from_ts(df, time_col, acc_cols, gap_factor=5):
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col].values.astype(np.int64)
    vals = df[acc_cols].values.astype(np.float32)

    diffs = np.diff(times)
    pos_mask = diffs > 0
    if pos_mask.sum() == 0:
        return np.empty((0, WINDOW, 3), dtype=np.float32)

    med_dt = float(np.median(diffs[pos_mask]))
    gap_mask = diffs > med_dt * gap_factor
    split_pts = np.where(gap_mask)[0] + 1
    segments = np.split(np.arange(len(df)), split_pts)

    all_wins = []
    for idx in segments:
        if len(idx) < WINDOW:
            continue
        seg = _oracle_resample_interp(vals[idx], times[idx])
        wins = _oracle_sliding_windows(seg)
        if len(wins):
            all_wins.append(wins)

    return np.concatenate(all_wins) if all_wins else np.empty((0, WINDOW, 3), dtype=np.float32)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _uniform_recording_g(n_samples, hz=TARGET_HZ, seed=0):
    """A clean, uniformly sampled Recording in g: list of [t_ns, x, y, z]."""
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t = np.arange(n_samples, dtype=np.int64) * dt_ns
    # Gravity-included total acceleration, ~1 g magnitude, small motion.
    xyz = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(n_samples, 3))
    return [[int(t[i]), float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])]
            for i in range(n_samples)]


def _to_df(samples):
    arr = np.array(samples, dtype=np.float64)
    return pd.DataFrame({"t": arr[:, 0].astype(np.int64),
                         "x": arr[:, 1], "y": arr[:, 2], "z": arr[:, 3]})


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_uniform_g_recording_matches_notebook_windows():
    """A clean 50 Hz Recording yields the same Window(s) as the merge pipeline."""
    samples = _uniform_recording_g(n_samples=130)

    oracle = _oracle_windows_from_ts(_to_df(samples), "t", ["x", "y", "z"])
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape[0] == 1  # 130 samples @ 50 Hz -> exactly one Window
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def _two_segment_recording_g(seg_len=130, gap_multiple=100, hz=TARGET_HZ, seed=1):
    """Two clean segments separated by one big timestamp gap."""
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t1 = np.arange(seg_len, dtype=np.int64) * dt_ns
    t2 = t1[-1] + gap_multiple * dt_ns + np.arange(seg_len, dtype=np.int64) * dt_ns
    times = np.concatenate([t1, t2])
    xyz = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(2 * seg_len, 3))
    return [[int(times[i]), float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])]
            for i in range(2 * seg_len)]


def test_timestamp_gap_splits_into_segments_like_notebook():
    """A Recording with a gap > 5x median interval is split, not interpolated across."""
    samples = _two_segment_recording_g()

    oracle = _oracle_windows_from_ts(_to_df(samples), "t", ["x", "y", "z"])
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape[0] == 2  # one Window per segment, none straddling the gap
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def _jittery_multisegment_recording_g(seed=7):
    """Realistic phone-like Recording: ~50 Hz with jitter, unsorted, two gaps."""
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / TARGET_HZ)
    rows = []
    t = 0
    for seg in range(3):
        n = rng.integers(200, 320)
        for _ in range(n):
            jitter = int(rng.normal(0, dt_ns * 0.15))
            t += max(1, dt_ns + jitter)
            xyz = rng.normal(loc=[0.02, -0.05, 1.0], scale=0.2, size=3)
            rows.append([t, float(xyz[0]), float(xyz[1]), float(xyz[2])])
        t += dt_ns * 60  # a gap between segments (> 5x median)
    rng.shuffle(rows)  # arrive out of order; the pipeline must sort
    return [[int(r[0]), r[1], r[2], r[3]] for r in rows]


def test_realistic_recording_matches_notebook_parity():
    """End-to-end parity against the merge pipeline on jittery, out-of-order data."""
    samples = _jittery_multisegment_recording_g()

    oracle = _oracle_windows_from_ts(_to_df(samples), "t", ["x", "y", "z"])
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape[0] >= 3  # multiple Windows across the segments
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def test_window_count_and_step_overlap():
    """Windows are length 128, step 64, and consecutive Windows overlap by 64."""
    # 257 uniform samples @ 50 Hz -> 256 resampled -> (256-128)//64 + 1 = 3 Windows.
    samples = _uniform_recording_g(n_samples=257)
    result = to_canonical_windows(samples, units="g")

    assert result.shape == (3, WINDOW, 3)
    # Step is 64: the second half of Window i equals the first half of Window i+1.
    np.testing.assert_allclose(result[0, STEP:], result[1, :WINDOW - STEP],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result[1, STEP:], result[2, :WINDOW - STEP],
                               rtol=1e-5, atol=1e-6)


def test_recording_too_short_yields_zero_windows():
    """A Recording shorter than one Window produces no Windows (caller -> 422)."""
    samples = _uniform_recording_g(n_samples=50)  # < 128 samples
    result = to_canonical_windows(samples, units="g")

    assert result.shape == (0, WINDOW, 3)


def test_ms2_recording_equals_g_recording():
    """The same physical motion sent as m/s^2 yields the same Windows as g."""
    samples_g = _uniform_recording_g(n_samples=200)
    # Same rows, but acceleration expressed in m/s^2 (Android-style).
    samples_ms2 = [[t, x * G, y * G, z * G] for t, x, y, z in samples_g]

    from_g = to_canonical_windows(samples_g, units="g")
    from_ms2 = to_canonical_windows(samples_ms2, units="m/s2")

    assert from_g.shape[0] >= 1
    np.testing.assert_allclose(from_ms2, from_g, rtol=1e-5, atol=1e-6)
