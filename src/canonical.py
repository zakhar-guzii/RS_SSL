"""The Canonical Signal transform — the one definition shared by train and serve.

This module exists because of a specific bug. The merge pipeline and the serving
pipeline each had their own copy of "how a Recording becomes model input", they
drifted apart, and every static-pose Prediction silently collapsed (`sit` always
came back `stand`). Nothing crashed; the two halves simply disagreed.

So: ``src/data_merge.py`` (training) and ``src/server/preprocessing.py``
(serving) both import ``gravity_project`` from here. Neither defines its own.
If this transform changes, both sides change together or neither does.

**Input** — 6 raw device-frame channels ``[ax, ay, az, gx, gy, gz]``: total
acceleration in g (gravity included) and angular velocity in rad/s.

**Output** — 4 orientation-invariant channels
``[a_vert, a_horiz, g_vert, g_horiz]``: each triad split into the component
along the window's own gravity direction and the magnitude perpendicular to it.

Why project instead of feeding raw axes: the device frame rotates with the
phone, so a raw channel means something different in every Recording. Held
vertically, gravity sits on ay; in a pocket it's on ax; flat on a table it's az.
Models trained on one mounting produce garbage on another. Projecting onto the
measured gravity vector removes the device frame entirely while keeping the
vertical/horizontal split that separates walking from stairs.

Measured on 5,945 UCI HAR windows (logistic regression on per-window mean/std,
4-fold CV), under random per-window 3D rotation simulating arbitrary phone
orientation:

    transform      frame          5-class   sit-vs-stand
    raw axes       as-recorded      0.786          0.871
    raw axes       rotated          0.596          0.605
    gravity-proj   as-recorded      0.812          0.892
    gravity-proj   rotated          0.812          0.892

Exactly invariant, and better than raw axes even in the as-recorded frame.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS_RAW = 6  # [ax, ay, az, gx, gy, gz] as recorded
N_CHANNELS = 4      # [a_vert, a_horiz, g_vert, g_horiz] after projection
GRAVITY_WIN = 51    # ~1 s at 50 Hz — low-pass span used to estimate gravity


def _moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average over axis 1 of (N, T, C), edge-padded.

    numpy-only on purpose: this runs inside the serving image, and a rolling
    mean is not worth a scipy dependency there.
    """
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(xp, axis=1, dtype=np.float64)
    c = np.concatenate([np.zeros((x.shape[0], 1, x.shape[2])), c], axis=1)
    return ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)


def gravity_project(X: np.ndarray) -> np.ndarray:
    """(N, T, 6) raw device-frame Windows -> (N, T, 4) orientation-invariant.

    Gravity is estimated per window by low-passing the accelerometer (it is the
    only sustained acceleration a phone sees), then both triads are split into
    the component along that direction and the magnitude perpendicular to it.
    Both parts are rotation-invariant, so the result no longer depends on how
    the phone was held.
    """
    if X.shape[-1] != N_CHANNELS_RAW:
        raise ValueError(f"expected {N_CHANNELS_RAW} channels, got {X.shape[-1]}")

    a, g = X[:, :, 0:3], X[:, :, 3:6]
    grav = _moving_mean(a, GRAVITY_WIN)
    ghat = grav / (np.linalg.norm(grav, axis=2, keepdims=True) + 1e-8)

    a_vert = (a * ghat).sum(2)
    a_horiz = np.linalg.norm(a - a_vert[:, :, None] * ghat, axis=2)
    g_vert = (g * ghat).sum(2)
    g_horiz = np.linalg.norm(g - g_vert[:, :, None] * ghat, axis=2)

    return np.stack([a_vert, a_horiz, g_vert, g_horiz], axis=2).astype(np.float32)


def _self_check() -> None:
    """Rotating the input must not change the output. That is the whole point."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 128, 6)).astype(np.float32)
    X[:, :, 0:3] += np.array([0.2, -0.9, 0.3])  # a plausible gravity offset

    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))  # uniform random rotation
    Xr = X.copy()
    Xr[:, :, 0:3] = X[:, :, 0:3] @ q.T
    Xr[:, :, 3:6] = X[:, :, 3:6] @ q.T

    out, out_rot = gravity_project(X), gravity_project(Xr)
    assert out.shape == (8, 128, N_CHANNELS), out.shape
    err = np.abs(out - out_rot).max()
    assert err < 1e-4, f"not rotation invariant: max deviation {err}"

    # a_horiz and g_horiz are magnitudes, so they can never be negative.
    assert (out[:, :, 1] >= 0).all() and (out[:, :, 3] >= 0).all()
    print(f"canonical self-check OK (max rotation deviation {err:.2e})")


if __name__ == "__main__":
    _self_check()
