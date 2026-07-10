# Handoff: HAR Backend — core built & tested (session 2)

**Repo:** `/Users/zaharguzij/Documents/GitHub/RS_SSL` (branch `backend`; PRs → `baseline`)
**Date:** 2026-07-10
**State:** Backend core implemented **test-first, 16 tests green**. Nothing committed. Real Model Bundles do not exist yet (training doesn't export them) — the suite proves the machinery with random-weight bundles.

## Read these first (do not duplicate)

- Domain vocabulary: [`CONTEXT.md`](../../../../../Users/zaharguzij/Documents/GitHub/RS_SSL/CONTEXT.md) — App, Activity, Recording, Canonical Signal, Window, Prediction, Model Bundle, Backend.
- PRD (38 user stories, seams, API contract): `docs/prd/0001-backend.md`.
- Prior handoff (design decisions, dataset facts, the norm-stats trap): `handoffs/har-backend-handoff.md`.
- Proposed issue breakdown (5 vertical slices) was drafted in-conversation and **not yet published/approved** — see below.

## What was built this session (all under `src/server/`, all TDD red→green)

Don't re-describe the modules — read them. Summary of coverage:

- `preprocessing.py` — **Seam 1**, `to_canonical_windows(samples, units) -> (n_win,128,3)`. Proven `allclose` to the merge notebook's `windows_from_ts`/`resample_interp`/`sliding_windows` on jittery, out-of-order, multi-segment data. This was the crux; it's done.
- `predictor.py` — **Seam 2**, `Predictor.predict` (normalize → infer → mean-softmax → argmax → Prediction dict). Raises `TooShortError`.
- `architectures.py` — `build_model(arch)` rebuilds `CNNClassifier` / `TransformerClassifier` from a bundle's `arch` config (lazy imports the existing training classes).
- `bundle.py` — `load_bundle` / `load_bundles`; enforces unique ids and exactly one default.
- `app.py` — FastAPI: `GET /health`, `GET /models`, `POST /predict` (404 unknown, 422 too short), CORS open. `create_app(bundles)` for tests; `build_default_app()` reads `HAR_MODELS_DIR` (default `models/`) for uvicorn.

Tests in `tests/` (`test_preprocessing.py`, `test_predictor.py`, `test_bundle.py`, `test_app.py`). Oracle/parity approach: notebook functions copied verbatim into the test as an independent source of truth.

Run: `.venv/bin/pytest -q` (pytest config added to `pyproject.toml`; `pythonpath=["src"]`).
Deps added this session: `fastapi`, `uvicorn` (main); `pytest`, `httpx` (dev).

## The one blocking gap (do this next)

**Training does not export Model Bundles**, and never persists the training-split norm stats — [`src/merged_dataset.py:49-51`](../../../../../Users/zaharguzij/Documents/GitHub/RS_SSL/src/merged_dataset.py). Until this is done, `models/` is empty and the server has nothing real to serve. Bundle format expected by `bundle.py` (mirror it exactly):

```
models/<id>/
  weights.pt        # state_dict
  meta.json         # {id, display_name, description, ssl_pretrained, is_default,
                    #  arch{type,...}, norm_mean[[[..]]], norm_std[[[..]]], label_order}
```

Add an export step to the training entry points (`baseline_cnn.py`/`encoder.py`, `transformer/transformer.py`, `ssl_pretrain*.py`) that writes this after training, pulling `norm_mean`/`norm_std` from the `MergedHARDataset`. The user will retrain — this is their domain; confirm before touching training loops.

## Remaining work (from the 5-slice plan, unpublished)

1. Bundle export + retrain (baseline CNN) — **blocking, above**.
2. ~~Tracer bullet single-model `/predict`~~ — **machinery done**; needs a real bundle to be truly end-to-end.
3. Multiple selectable bundles — **done** (`/models`, `model_id`, 404, one-default enforcement).
4. Simulated-phone client (`curl`/Python example) + API docs — **not started**.
5. Bundle export for CNN-LSTM / transformer / SSL — depends on (1).
- ADR 0001 (backend owns preprocessing) — **not written**; PRD references it.

## Watch-outs

- `PredictRequest.model_id` starts with `model_` (pydantic protected namespace). Works and tested, but if pydantic emits a warning later, set `model_config = ConfigDict(protected_namespaces=())`.
- Server is CPU-only, binds `0.0.0.0:8000` (avoid MLflow's 5000) — not yet exercised as a live process; only via `TestClient`.
- `notebooks/har_merge.ipynb` has pre-existing uncommitted edits (not from these sessions).

## Suggested skills for next session

- `/tdd` — for the bundle-export step (assert a trained run writes a loadable bundle whose `Predictor` round-trips).
- `/verify` — boot the real uvicorn server against a real `models/` dir and drive a live Prediction with the simulated-phone client before committing.
- `/domain-modeling` — write ADR 0001 in its ADR format.
- `/code-review` — before the PR to `baseline`.
- `/to-issues` — if you still want the 5 slices published (no issue tracker / `gh` configured; prior fallback was Markdown files).
