"""Tests for Seam 1: Recording -> Canonical Signal Windows (6-channel).

The source of truth for parity is the merge pipeline in ``src/data_merge.py``.
Its windowing functions are imported directly here as an independent oracle: the
code under test (``server.preprocessing.to_canonical_windows``) is a separate
implementation and must reproduce the merge's Windows, so a match is a real
agreement between the two, not a tautology.

The Canonical Signal is 6-channel ``[ax, ay, az, gx, gy, gz]`` — accelerometer
in g, gyroscope in rad/s. Only the accelerometer channels are unit-converted.
"""

import numpy as np
import pandas as pd

from server.preprocessing import to_canonical_windows

# Merge pipeline = the parity oracle.
from data_merge import windows_from_ts as _oracle_windows_from_ts

WINDOW = 128
STEP = 64
TARGET_HZ = 50
G = 9.80665  # m/s^2 per g
CH = ["ax", "ay", "az", "gx", "gy", "gz"]


# --------------------------------------------------------------------------
# Fixtures — 6-channel rows [t_ns, ax, ay, az, gx, gy, gz]
# --------------------------------------------------------------------------
def _uniform_recording(n_samples, hz=TARGET_HZ, seed=0):
    """A clean, uniformly sampled 6-channel Recording (accel g + gyro rad/s)."""
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t = np.arange(n_samples, dtype=np.int64) * dt_ns
    acc = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(n_samples, 3))   # ~1 g
    gyro = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.5, size=(n_samples, 3))  # rad/s
    return [[int(t[i]), *acc[i].tolist(), *gyro[i].tolist()] for i in range(n_samples)]


def _to_df(samples):
    arr = np.array(samples, dtype=np.float64)
    return pd.DataFrame({"t": arr[:, 0].astype(np.int64),
                         **{c: arr[:, i + 1] for i, c in enumerate(CH)}})


def _oracle(samples):
    return _oracle_windows_from_ts(_to_df(samples), "t", CH)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_uniform_recording_matches_merge_windows():
    """A clean 50 Hz Recording yields the same 6-channel Window(s) as the merge."""
    samples = _uniform_recording(n_samples=130)

    oracle = _oracle(samples)
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape == (1, WINDOW, 6)  # 130 samples @ 50 Hz -> one Window
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def _two_segment_recording(seg_len=130, gap_multiple=100, hz=TARGET_HZ, seed=1):
    """Two clean segments separated by one big timestamp gap."""
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t1 = np.arange(seg_len, dtype=np.int64) * dt_ns
    t2 = t1[-1] + gap_multiple * dt_ns + np.arange(seg_len, dtype=np.int64) * dt_ns
    times = np.concatenate([t1, t2])
    acc = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(2 * seg_len, 3))
    gyro = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.5, size=(2 * seg_len, 3))
    return [[int(times[i]), *acc[i].tolist(), *gyro[i].tolist()] for i in range(2 * seg_len)]


def test_timestamp_gap_splits_into_segments_like_merge():
    """A Recording with a gap > 5x median interval is split, not interpolated across."""
    samples = _two_segment_recording()

    oracle = _oracle(samples)
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape[0] == 2  # one Window per segment, none straddling the gap
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def _jittery_multisegment_recording(seed=7):
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
            acc = rng.normal(loc=[0.02, -0.05, 1.0], scale=0.2, size=3)
            gyro = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.5, size=3)
            rows.append([t, *acc.tolist(), *gyro.tolist()])
        t += dt_ns * 60  # a gap between segments (> 5x median)
    rng.shuffle(rows)  # arrive out of order; the pipeline must sort
    return [[int(r[0]), *r[1:]] for r in rows]


def test_realistic_recording_matches_merge_parity():
    """End-to-end parity against the merge pipeline on jittery, out-of-order data."""
    samples = _jittery_multisegment_recording()

    oracle = _oracle(samples)
    result = to_canonical_windows(samples, units="g")

    assert oracle.shape[0] >= 3 and oracle.shape[2] == 6
    assert result.shape == oracle.shape
    np.testing.assert_allclose(result, oracle, rtol=1e-5, atol=1e-6)


def test_window_count_and_step_overlap():
    """Windows are length 128, step 64, 6 channels, and consecutive Windows overlap."""
    # 257 uniform samples @ 50 Hz -> 256 resampled -> (256-128)//64 + 1 = 3 Windows.
    samples = _uniform_recording(n_samples=257)
    result = to_canonical_windows(samples, units="g")

    assert result.shape == (3, WINDOW, 6)
    np.testing.assert_allclose(result[0, STEP:], result[1, :WINDOW - STEP], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result[1, STEP:], result[2, :WINDOW - STEP], rtol=1e-5, atol=1e-6)


def test_recording_too_short_yields_zero_windows():
    """A Recording shorter than one Window produces no Windows (caller -> 422)."""
    samples = _uniform_recording(n_samples=50)  # < 128 samples
    result = to_canonical_windows(samples, units="g")

    assert result.shape == (0, WINDOW, 6)


def test_ms2_converts_accel_only_gyro_unchanged():
    """m/s^2 request: accel is divided by g; gyroscope channels are left as rad/s."""
    samples_g = _uniform_recording(n_samples=200)
    # Same physical motion, but accelerometer expressed in m/s^2 (Android-style);
    # gyroscope stays in rad/s (never scaled).
    samples_ms2 = [[t, ax * G, ay * G, az * G, gx, gy, gz]
                   for t, ax, ay, az, gx, gy, gz in samples_g]

    from_g = to_canonical_windows(samples_g, units="g")
    from_ms2 = to_canonical_windows(samples_ms2, units="m/s2")

    assert from_g.shape[0] >= 1 and from_g.shape[2] == 6
    np.testing.assert_allclose(from_ms2, from_g, rtol=1e-5, atol=1e-6)
