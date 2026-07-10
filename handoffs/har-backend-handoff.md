# Handoff: HAR Activity-Recognition Backend (RS_SSL)

**Repo:** `/Users/zaharguzij/Documents/GitHub/RS_SSL` (work on branch `backend`; PRs target `baseline`)
**Date:** 2026-07-10
**State:** Design grilled and converged; **implementation not started**. The user was asked to confirm the final design + a proposed ADR at the end of the session but has not yet replied — get that confirmation (or treat this doc as the spec if they say "go").

## What is being built

A **backend only** (the cross-platform mobile app comes later, by others). Flow: user picks a model in the app → presses a button → app records ~15 s of phone accelerometer data → POSTs raw data to the backend → backend classifies and returns the activity.

Domain vocabulary (App, Activity, Recording, Canonical Signal, Window, Prediction, Model Bundle) is already written down in the repo — **read `CONTEXT.md` first**; do not duplicate it here or drift from it.

## Decisions made (with the user, one by one)

1. **Sensor/UI** = a cross-platform mobile app using the phone's own IMU. Out of scope for now; deliver a documented API contract + a Python/`curl` client example simulating a phone.
2. **Server** = a standalone FastAPI + PyTorch service on a team laptop — explicitly *not* a Jupyter notebook. Lives in this repo (suggested `src/server/`), reuses existing model code in `src/`.
3. **Network (MVP)** = phone and laptop on shared Wi-Fi. Bind `0.0.0.0`, **port 8000** (MLflow uses 5000). No auth, no tunnel, CORS open.
4. **App sends raw data; backend owns ALL preprocessing** (proposed as ADR 0001, not yet written — offer to write `docs/adr/0001-backend-owns-preprocessing.md` after confirmation). One exception: the payload must declare `units` (`"g"` or `"m/s2"`) because Android reports m/s² and iOS reports g.
5. **Prediction semantics**: recording duration is a client choice (15 s is just the default example); anything yielding ≥ 1 window (≥ ~2.6 s) is valid. Aggregate per-window softmax by **mean, then argmax**; return top label + confidence + per-class probabilities + per-window label sequence.
6. **Multiple selectable models**: user picks SSL-pretrained vs non-SSL in the app *before* recording. Backend scans a `models/` directory of Model Bundles at startup; `POST /predict` takes `model_id`.
7. **Privacy**: classify and discard — never persist user recordings (user was explicit, against my recommendation; do not re-litigate).

## API contract (agreed)

- `GET /health`
- `GET /models` → list: id, display name, SSL flag, default flag
- `POST /predict` → `{model_id, units, samples: [[t, x, y, z], ...]}` → Prediction. Errors: 404 unknown model, 422 too short.

## Inference pipeline (must mirror training exactly)

Convert to g → split segments at timestamp gaps (> 5× median interval, matching the merge notebook) → linear-interp resample to 50 Hz → 128-sample windows, step 64 → normalize with the bundle's training stats → model → mean softmax → argmax. Reference implementation of resampling/windowing: `notebooks/har_merge.ipynb` (`resample_interp`, `sliding_windows`, `windows_from_ts`).

## Verified dataset facts (checked in-session, don't re-derive)

- `data/merged/har_merged.npz`: `X (61891, 128, 3)` float32, **accelerometer-only** (no gyro), total acceleration (gravity included), **units = g, consistent across all 4 sources** (median |a| ≈ 1.25–1.34 per source: uci, wisdm, motionsense, hhar), 50 Hz.
- Labels 0–4 alphabetical: `downstairs, sit, stand, upstairs, walk`.
- Known limitation (accepted): out-of-vocabulary activities (lying, jogging…) are confidently misclassified into these 5.

## Critical prerequisite / trap

`src/merged_dataset.py:49-51` computes normalization mean/std from the training split at load time and **never persists them**. The user will retrain all models (baseline CNN, CNN-LSTM, transformer, SSL variants: `ssl_pretrain*.py`) on the merged dataset. Part of this work: add an export step to training scripts producing a **Model Bundle** — `models/<id>/weights.pt` + `meta.json` (norm_mean, norm_std, label order, architecture config, display name, SSL flag, default flag). Without the exact training norm stats, predictions are garbage.

## Suggested skills for the next session

- `/domain-modeling` — keep `CONTEXT.md` sharp as terms evolve; write the pending ADR via its ADR format.
- `/tdd` — build the preprocessing pipeline test-first (resample/window/normalize parity against the merge-notebook functions is very testable; e.g. feed UCI windows through the server pipeline and assert near-identity).
- `/verify` — before committing, exercise the running server end-to-end with the simulated-phone client.
- `/code-review` — review the branch before PR to `baseline`.

## Environment notes

- Python via `uv` (`uv sync`, `.venv/bin/python` works). Config system: hierarchical YAML in `configs/` via `src/config_loader.py`.
- `notebooks/har_merge.ipynb` has uncommitted modifications (pre-existing; not from this session).
