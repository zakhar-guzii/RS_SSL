"""Tests for Seam 2: Predictor turns a Recording into a Prediction.

Aggregation (mean-then-argmax) is verified against an independent numpy
computation over known logits, so the assertion cannot recompute the expected
value the same way the Predictor does.
"""

import numpy as np
import torch
import torch.nn as nn

import pytest

from server.predictor import Predictor, TooShortError

LABEL_ORDER = ["downstairs", "sit", "stand", "upstairs", "walk"]


def _uniform_recording_g(n_samples, hz=50, seed=0):
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t = np.arange(n_samples, dtype=np.int64) * dt_ns
    acc = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(n_samples, 3))
    gyro = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.5, size=(n_samples, 3))
    return [[int(t[i]), *acc[i].tolist(), *gyro[i].tolist()] for i in range(n_samples)]


class _FixedLogitsModel(nn.Module):
    """Ignores input, returns preset per-Window logits (one row per Window)."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        assert x.shape[0] == self.logits.shape[0], "one logit row per Window"
        return self.logits


def _numpy_softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_too_short_recording_raises():
    predictor = Predictor(
        model=_FixedLogitsModel(np.zeros((1, 5), np.float32)),
        norm_mean=np.zeros((1, 1, 6), np.float32),
        norm_std=np.ones((1, 1, 6), np.float32),
        label_order=LABEL_ORDER,
    )
    with pytest.raises(TooShortError):
        predictor.predict(_uniform_recording_g(50), units="g")


class _EchoChannelMeanModel(nn.Module):
    """logits[:, :3] = per-Window means of the first 3 (accel) input channels."""

    def forward(self, x):  # x: (N, 128, 6)
        n = x.shape[0]
        out = torch.full((n, 5), -1e9, dtype=torch.float32)
        out[:, :3] = x.mean(dim=1)[:, :3]
        return out


def test_normalization_is_applied_per_bundle():
    from server.preprocessing import to_canonical_windows

    samples = _uniform_recording_g(400, seed=3)
    norm_mean = np.array([[[0.01, -0.02, 0.98, 0.0, 0.0, 0.0]]], dtype=np.float32)
    norm_std = np.array([[[0.30, 0.30, 0.35, 0.50, 0.50, 0.50]]], dtype=np.float32)

    predictor = Predictor(
        model=_EchoChannelMeanModel(),
        norm_mean=norm_mean, norm_std=norm_std, label_order=LABEL_ORDER,
    )
    pred = predictor.predict(samples, units="g")

    # Independently reproduce: normalize, echo accel channel means, softmax, mean.
    windows = to_canonical_windows(samples, units="g")
    normed = (windows - norm_mean) / norm_std
    logits = np.full((windows.shape[0], 5), -1e9, dtype=np.float32)
    logits[:, :3] = normed.mean(axis=1)[:, :3]
    mean_probs = _numpy_softmax(logits).mean(axis=0)

    assert pred["activity"] == LABEL_ORDER[int(mean_probs.argmax())]
    np.testing.assert_allclose(
        [pred["probabilities"][l] for l in LABEL_ORDER], mean_probs,
        rtol=1e-4, atol=1e-6,
    )


def test_prediction_is_mean_softmax_then_argmax():
    # 257 uniform samples -> 3 Windows (matches Seam 1 geometry).
    samples = _uniform_recording_g(257)
    logits = np.array([
        [8.0, 0.0, 0.0, 0.0, 0.0],   # Window 0 -> downstairs
        [0.0, 8.0, 0.0, 0.0, 0.0],   # Window 1 -> sit
        [0.0, 0.0, 2.5, 2.0, 0.0],   # Window 2 -> stand (mild)
    ], dtype=np.float32)

    predictor = Predictor(
        model=_FixedLogitsModel(logits),
        norm_mean=np.zeros((1, 1, 6), np.float32),
        norm_std=np.ones((1, 1, 6), np.float32),
        label_order=LABEL_ORDER,
    )
    pred = predictor.predict(samples, units="g")

    probs = _numpy_softmax(logits)
    mean_probs = probs.mean(axis=0)
    expected_idx = int(mean_probs.argmax())

    assert pred["activity"] == LABEL_ORDER[expected_idx]
    assert pred["confidence"] == float(mean_probs[expected_idx])
    # Per-class probabilities: one entry per label, summing to ~1.
    assert set(pred["probabilities"].keys()) == set(LABEL_ORDER)
    np.testing.assert_allclose(
        [pred["probabilities"][l] for l in LABEL_ORDER], mean_probs,
        rtol=1e-5, atol=1e-6,
    )
    # Per-Window label sequence = per-Window argmax.
    expected_window_labels = [LABEL_ORDER[i] for i in probs.argmax(axis=1)]
    assert pred["window_labels"] == expected_window_labels
