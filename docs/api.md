# HAR Backend API

The Backend classifies a phone Recording — **accelerometer + gyroscope** — into one of five Activities: `downstairs`, `sit`, `stand`, `upstairs`, `walk`. It does all preprocessing itself — see [ADR 0001](adr/0001-backend-owns-preprocessing.md) — so a client only needs to capture raw samples and declare their accelerometer units.

Domain terms (App, Activity, Recording, Canonical Signal, Window, Prediction, Model Bundle) are defined in [`CONTEXT.md`](../CONTEXT.md).

Base URL: `http://<laptop-ip>:8000` (phone and laptop must be on the same Wi-Fi). No authentication; CORS is open. A live interactive reference (try-it-out) is served by the Backend itself at `GET /docs`.

A full runnable client is at [`examples/simulate_phone.py`](../examples/simulate_phone.py) — read it alongside this page.

## `GET /health`

Liveness check.

```
curl http://127.0.0.1:8000/health
→ 200 {"status": "ok"}
```

## `GET /models`

Lists every Model Bundle the Backend loaded at startup.

```
curl http://127.0.0.1:8000/models
→ 200 [
  {"id": "baseline_cnn", "display_name": "CNN (supervised baseline)", "ssl_pretrained": false, "is_default": true},
  {"id": "transformer",  "display_name": "Transformer (supervised)",  "ssl_pretrained": false, "is_default": false},
  ...
]
```

Exactly one bundle has `is_default: true` — use it as the app's pre-selection when the user hasn't chosen a model. The set of ids is fixed for the lifetime of a Backend process (bundles are scanned once at startup), so a `model_id` picked before recording is still valid at predict time.

## `POST /predict`

Classifies one Recording.

**Request body:**

```json
{
  "model_id": "baseline_cnn",
  "units": "g",
  "samples": [[0, 0.01, -0.02, 0.99, 0.05, -0.03, 0.01], [20000000, 0.02, -0.01, 1.0, 0.04, -0.02, 0.0], ...]
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `model_id` | string | Must be an `id` from `GET /models`. |
| `units` | `"g"` \| `"m/s2"` | Unit of the **accelerometer** channels (`ax, ay, az`) only. **Android's `SensorEvent` reports m/s²; iOS's `CMAccelerometerData` reports g.** Get this wrong and every prediction is silently wrong — nothing else will error. Gyroscope is **always rad/s** (native on both iOS CoreMotion and Android) and is never converted — there is no unit field for it. |
| `samples` | array of `[t, ax, ay, az, gx, gy, gz]` | Seven values per row: nanosecond timestamp + 3 accelerometer + 3 gyroscope channels, **in that exact order**. `ax,ay,az` = total acceleration (gravity included) in `units`; `gx,gy,gz` = angular velocity in rad/s. **`t` is a timestamp in nanoseconds** (an integer, e.g. from `System.nanoTime()` / a monotonic clock) — not seconds, not milliseconds. Samples need not be evenly spaced or sorted; the Backend resamples to 50 Hz and sorts internally. Any length is valid as long as it yields at least one 128-sample Window at 50 Hz (~2.56 s of continuous signal after resampling); shorter is rejected with `422`. |

**Response (200):**

```json
{
  "activity": "walk",
  "confidence": 0.92,
  "probabilities": {"downstairs": 0.01, "sit": 0.0, "stand": 0.02, "upstairs": 0.05, "walk": 0.92},
  "window_labels": ["walk", "walk", "walk", "stand", "walk"]
}
```

- `activity` / `confidence` — the top Activity and its probability, from mean-then-argmax over all Windows in the Recording.
- `probabilities` — one entry per label, summing to ~1.0. Use it to show runner-up activities.
- `window_labels` — the per-Window argmax label sequence, in order. Length equals the number of 128-sample Windows the Recording produced (step 64, so roughly `duration_s * 50 / 64`).

**Errors:**

| Status | Cause |
| --- | --- |
| `404` | `model_id` is not one of the ids from `GET /models`. |
| `422` | The Recording is too short to form even one Window (< ~2.56 s of continuous signal after resampling), or the request body doesn't match the schema above. |

## Known limitation

Activities outside the five labels (e.g. lying down, jogging) are not detected as "unknown" — they are confidently misclassified into one of the five. This is an accepted limitation of the MVP, not a bug.

## Recordings are never stored

The Backend classifies and discards every Recording. Nothing sent to `/predict` is persisted, logged to disk, or retained after the response is returned.
