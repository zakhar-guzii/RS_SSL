# PRD 0001 — HAR Activity-Recognition Backend

_Labels: `ready-for-agent`_
_Branch: `backend` → PRs target `baseline`_
_Domain vocabulary: see [`CONTEXT.md`](../../CONTEXT.md) (App, Activity, Recording, Canonical Signal, Window, Prediction, Model Bundle, Backend)._

## Problem Statement

A user has a phone in their pocket and wants to know which of five everyday **Activities** they're doing (`downstairs`, `sit`, `stand`, `upstairs`, `walk`). The trained models that can answer this live only inside Jupyter notebooks and MLflow runs — there is no way for a phone to ask them anything. There is also no standardized way to turn a phone's raw, irregularly-sampled accelerometer stream into the exact **Canonical Signal** the models were trained on, so even if a phone could reach the models, the answer would be wrong.

On top of that, the training code computes normalization statistics on the fly and **never persists them** (`src/merged_dataset.py:49-51`). Without the exact training-time mean/std, any inference — however well-wired — produces garbage. There is currently no artifact that packages a trained model together with everything needed to reproduce its training-time preprocessing.

## Solution

Ship a standalone **Backend**: a FastAPI + PyTorch service, runnable on a team laptop, that

- exposes the trained models as selectable **Model Bundles**,
- accepts a raw **Recording** from a phone over HTTP,
- transforms it into the **Canonical Signal**, cuts **Windows**, runs inference, and
- returns a **Prediction** (top Activity + confidence + per-class probabilities + per-Window label sequence),

and never persists the user's motion data.

To make that possible we also add a **Model Bundle export** step to the training scripts, so every trained model is saved as `models/<id>/weights.pt` + `meta.json` (norm stats, label order, architecture config, display name, SSL flag, default flag). The Backend scans `models/` at startup and serves whatever bundles it finds.

From the user's perspective: they open the (future) App, pick "SSL-pretrained" or "plain" model, press a button, and a couple seconds later see "**walk** — 92%".

Governed by [ADR 0001 — Backend owns all preprocessing](../adr/0001-backend-owns-preprocessing.md) (to be written alongside this work): the App sends raw data and declares only `units`; the Backend owns the entire pipeline.

## User Stories

### App / end-to-end (the phone is simulated by a Python/`curl` client for now)

1. As an App user, I want to press one button and get back the Activity I'm doing, so that I don't have to interpret raw sensor numbers myself.
2. As an App user, I want a confidence score with the Activity, so that I know how much to trust the answer.
3. As an App user, I want to choose which model classifies my Recording before I record, so that I can compare the SSL-pretrained model against the plain one.
4. As an App user, I want to see the full list of available models with human-readable names, so that I can make an informed choice.
5. As an App user, I want to know which model is the default, so that I don't have to choose if I don't care.
6. As an App user, I want my Recording to be classified and discarded, so that my motion data is never stored anywhere.
7. As an App user, I want a Recording of any reasonable length (not just exactly 15 s) to work, so that I'm not forced into a rigid capture window.
8. As an App user, I want a clear error if my Recording is too short to classify, so that I know to record for longer rather than getting a meaningless answer.
9. As an App user on Android (m/s²) or iOS (g), I want my Recording interpreted in the correct units, so that the Activity is correct regardless of my phone's OS.
10. As an App user, I want per-class probabilities returned, so that the App can show me the runner-up Activities.
11. As an App user, I want the per-Window label sequence returned, so that the App could (later) show how the predicted Activity changed over the Recording.

### Client integration (App developer, who is not us)

12. As an App developer, I want a documented, stable API contract, so that I can build the phone client against it without reading the server source.
13. As an App developer, I want a runnable Python/`curl` example that simulates a phone Recording, so that I can see exactly what to send and what to expect.
14. As an App developer, I want the Backend to own all preprocessing, so that my client only has to capture raw samples and declare their units.
15. As an App developer, I want CORS open and no auth (MVP), so that I can call the Backend from any client on the shared Wi-Fi without configuring credentials.
16. As an App developer, I want to discover models at runtime via an endpoint, so that new models added to the Backend appear in my App without a client release.
17. As an App developer, I want well-defined HTTP status codes (404 unknown model, 422 too short), so that I can map failures to user-facing messages.
18. As an App developer, I want to know the model list is stable within a Backend run (ids don't change), so that a `model_id` chosen before recording is still valid at predict time.

### Backend operator (team member running the laptop)

19. As a Backend operator, I want to start the service with a single command, so that I can host it during a demo without ceremony.
20. As a Backend operator, I want the service to bind `0.0.0.0:8000`, so that phones on the same Wi-Fi can reach it and it doesn't collide with MLflow on 5000.
21. As a Backend operator, I want a `GET /health` endpoint, so that I can confirm the service is up before a demo.
22. As a Backend operator, I want the Backend to scan `models/` at startup and log which bundles it loaded, so that I know what's being served.
23. As a Backend operator, I want a clear startup error if a bundle is malformed or missing files, so that I find out at boot rather than on the first prediction.
24. As a Backend operator, I want the Backend to run on CPU (no GPU on the laptop), so that I can host it on ordinary hardware.
25. As a Backend operator, I want inference on a ~15 s Recording to return in well under a second, so that the demo feels responsive.

### Model author (us, retraining on the merged dataset)

26. As a model author, I want each training run to export a Model Bundle, so that the exact weights and preprocessing are reproducible at inference time.
27. As a model author, I want the training-time normalization mean/std persisted into the bundle, so that inference normalizes inputs identically to training.
28. As a model author, I want the label order (`downstairs, sit, stand, upstairs, walk`) recorded in the bundle, so that class indices are never misinterpreted.
29. As a model author, I want the architecture config stored in the bundle, so that the Backend can reconstruct the model without guessing hyperparameters.
30. As a model author, I want to mark a bundle as SSL-pretrained (or not) and set a display name, so that the App can present a meaningful choice.
31. As a model author, I want to designate exactly one bundle as the default, so that the App has a sensible pre-selection.
32. As a model author, I want to add a new model by dropping a bundle folder into `models/` and restarting, so that I don't have to touch Backend code to serve a new architecture.
33. As a model author, I want the same export format to work for the baseline CNN, CNN-LSTM, transformer, and SSL variants, so that all approaches are served uniformly.

### Correctness / parity (the crux)

34. As a model author, I want the Backend's preprocessing to reproduce the merge pipeline exactly (unit conversion → gap-splitting → 50 Hz resample → 128-sample windows, step 64), so that Windows fed to the model match training Windows.
35. As a model author, I want a test that feeds known merged-dataset Windows through the Backend pipeline and asserts near-identity, so that parity is guaranteed and stays guaranteed.
36. As a model author, I want Recordings with timestamp gaps split into segments (gap > 5× median interval), so that discontinuous captures don't produce corrupted Windows.
37. As a model author, I want per-Window softmax aggregated by mean-then-argmax across the whole Recording, so that the Prediction reflects the entire capture, not a single Window.
38. As a model author, I want out-of-vocabulary activities (lying, jogging…) to still return one of the five labels (known, accepted limitation), so that behavior is predictable even if wrong.

## Implementation Decisions

### New module: `src/server/` (standalone FastAPI + PyTorch service)

- Lives in this repo, reuses existing model code in `src/`. Not a notebook. Entry point runnable via a single command; binds `0.0.0.0:8000`.
- Runs on **CPU**. No auth, CORS open (MVP, shared Wi-Fi).
- Loads all Model Bundles from a `models/` directory **at startup**; the loaded set is fixed for the lifetime of the process. Startup fails loudly on a malformed bundle.

### Seam 1 — Preprocessing (pure function, highest-value test seam)

A pure function transforming a raw Recording into model-ready Windows, mirroring the merge notebook (`resample_interp`, `sliding_windows`, `windows_from_ts` in `notebooks/har_merge.ipynb`):

- **Signature (shape contract):** `samples: list[[t, x, y, z]]`, `units: "g" | "m/s2"` → `ndarray (n_windows, 128, 3)` float32 of Canonical Signal Windows (pre-normalization).
- **Steps, in order:** convert to g (divide by 9.80665 if `m/s2`) → sort by timestamp → split into continuous segments where inter-sample interval > `5 × median interval` → linear-interp resample each segment to 50 Hz → sliding windows of length 128, step 64 → drop segments shorter than one Window.
- Normalization (subtract `norm_mean`, divide by `norm_std`) is applied **per-bundle** downstream (Seam 2), because stats differ per model. Keep this function bundle-agnostic so parity can be tested against the notebook without a model.
- Returns zero Windows when the Recording yields none; the caller maps that to HTTP 422.

### Seam 2 — Prediction service + route

- A `Predictor` composes: preprocessing (Seam 1) → per-bundle normalization → model forward → per-Window softmax → **mean across Windows → argmax**.
- Produces a **Prediction**: `{ activity, confidence, probabilities: {label: p}, window_labels: [label, ...] }`.
- Reconstructs the model from the bundle's architecture config and loads `weights.pt`; model runs in `eval()`/`no_grad`.

### Model Bundle format and export (training-side change)

- A bundle is a directory `models/<id>/` containing:
  - `weights.pt` — model `state_dict`.
  - `meta.json` — `{ id, display_name, description, ssl_pretrained: bool, is_default: bool, arch: {...full architecture config...}, norm_mean, norm_std, label_order: ["downstairs","sit","stand","upstairs","walk"], input: {window: 128, channels: 3, hz: 50, units: "g"} }`.
- **`norm_mean` / `norm_std` must be the exact training-split stats.** Persist them from `MergedHARDataset` at train time — this fixes the trap at `src/merged_dataset.py:49-51` where stats are computed and discarded.
- Add an export step to the training entry points (baseline CNN, CNN-LSTM, transformer, `ssl_pretrain*.py`) that writes the bundle after training. Exactly one bundle across `models/` has `is_default: true`.
- Architecture reconstruction reuses existing `build_model`-style constructors (`CNNClassifier`, `TransformerClassifier`, etc.) driven by `meta.arch`.

### API contract (agreed, do not drift)

- `GET /health` → liveness.
- `GET /models` → list of `{ id, display_name, ssl_pretrained, is_default }`.
- `POST /predict` → body `{ model_id, units, samples: [[t, x, y, z], ...] }` → Prediction.
  - `404` unknown `model_id`.
  - `422` Recording too short (yields zero Windows) or malformed payload.

### Client example

- A Python/`curl` example under `src/server/` (or `examples/`) that simulates a phone: builds a `samples` payload, POSTs to `/predict`, prints the Prediction. Doubles as living API documentation.

## Testing Decisions

Good tests here assert **external behavior** — the shape and correctness of Canonical Signal Windows and of Predictions — not internal wiring. Two seams, matching the two above:

### Seam 1 — preprocessing parity (the high-value tests; use `/tdd`)

- Feed known merged-dataset content through the pure preprocessing function and **assert near-identity** against the notebook's `windows_from_ts` / `resample_interp` / `sliding_windows` output (allclose within float tolerance). This is the parity guarantee against training.
- Unit conversion: identical Windows whether the same physical signal is sent as `g` or as `m/s2`.
- Gap-splitting: a Recording with an injected timestamp gap (> 5× median) produces the same segment boundaries as the notebook; no Window straddles the gap.
- Window geometry: correct `n_windows` for a given length (length 128 → 1, step 64); a sub-Window-length Recording → zero Windows.
- Prior art: the merge notebook functions are the reference oracle; mirror their fixtures.

### Seam 2 — service / route (contract + aggregation)

- Drive `POST /predict` via FastAPI `TestClient` against a small real Model Bundle: assert Prediction shape (`activity` in the five labels, `probabilities` sum ≈ 1 and one entry per label, `window_labels` length == `n_windows`).
- Aggregation: with a stubbed/known model producing controlled per-Window logits, assert the returned `activity` equals mean-then-argmax (not per-Window majority).
- Error paths: unknown `model_id` → 404; too-short Recording → 422.
- `GET /models` returns exactly the bundles present in the `models/` dir, with exactly one `is_default: true`.

Out of test scope: model accuracy itself (that's an MLflow/training concern), and load/perf testing beyond a sanity latency check.

## Out of Scope

- The cross-platform mobile App (built later, by others). We deliver only the API contract + simulated-phone client.
- Authentication, TLS, tunnels, rate limiting, and multi-user concerns — MVP is shared Wi-Fi, open CORS.
- Persisting Recordings or any user motion data — explicitly forbidden (classify and discard).
- GPU inference — CPU only.
- Retraining quality/accuracy targets — this PRD covers serving and export plumbing, not model performance.
- Handling out-of-vocabulary activities gracefully — accepted limitation; they map to one of the five labels.
- Hot-reloading bundles without restart — bundles are scanned once at startup.

## Further Notes

- **Blocking prerequisite:** the norm-stats trap at `src/merged_dataset.py:49-51`. The Model Bundle export (which persists training-split `norm_mean`/`norm_std`) must land before any bundle is trustworthy; the user is retraining all models on the merged dataset as part of this work.
- Reference implementation for resample/window/gap-split lives in `notebooks/har_merge.ipynb` (`resample_interp`, `sliding_windows`, `windows_from_ts`). Constants: `WINDOW=128`, `STEP=64`, `TARGET_HZ=50`, `gap_factor=5`.
- Verified dataset facts (don't re-derive): merged data is accelerometer-only, total acceleration in **g**, 50 Hz, labels 0–4 alphabetical. Median |a| ≈ 1.25–1.34 across sources.
- Config system is hierarchical YAML in `configs/` via `src/config_loader.py`; the Backend should read paths/ports from config where practical rather than hardcoding.
- Suggested skills for the build: `/tdd` (Seam 1 parity), `/verify` (exercise the running server with the simulated-phone client before commit), `/domain-modeling` (write ADR 0001), `/code-review` (before PR to `baseline`).
- Port 8000 deliberately avoids MLflow's 5000.
- Publish note: no issue tracker / `gh` CLI is configured, so this PRD ships as a repo Markdown file carrying the `ready-for-agent` label in its header instead of a tracker ticket.
