# Human Activity Recognition with CNN-LSTM

CNN-LSTM models for human activity recognition using raw inertial sensor signals from UCI HAR dataset.

## Setup

```bash
uv sync
```

## Running Experiments

```bash
uv run src/encoder.py          # Train CNN-LSTM model
uv run src/baseline.py         # Train Random Forest baseline
```

## Project Structure

### Source Code

| File | Purpose |
| --- | --- |
| `src/encoder.py` | CNN-LSTM training pipeline with config-driven setup |
| `src/baseline.py` | Random Forest baseline model |
| `src/torch_dataset.py` | Windowed dataset loader (128-sample windows with 50% overlap) |
| `src/dataset.py` | Utility functions for loading UCI HAR dataset |
| `src/config_loader.py` | YAML configuration loader with hierarchical merging |

### Configuration Hierarchy

All hyperparameters defined in `configs/` with inheritance:

```txt
configs/
├── shared.yaml                    # Base config (defaults for all experiments)
├── models/
│   ├── cnn_lstm.yaml             # CNN-LSTM architecture (overrides shared.yaml)
│   └── baseline_rf.yaml          # Random Forest config (overrides shared.yaml)
└── datasets/
    └── uci_har.yaml              # UCI HAR dataset paths
```

**How configs merge:**
1. Load `shared.yaml` as base
2. Merge `datasets/uci_har.yaml` (dataset-specific settings)
3. Merge `models/cnn_lstm.yaml` (experiment-specific settings, highest priority)

To create new experiment: create new YAML in `configs/models/` and run training.

### Data Directory

UCI HAR dataset structure required in `data/uci_har/`:

```txt
data/uci_har/
├── features.txt               # Feature metadata
├── train/
│   ├── X_train.txt
│   ├── y_train.txt
│   └── Inertial Signals/      # 9 raw signal files (body_acc_*, body_gyro_*, total_acc_*)
└── test/
    ├── X_test.txt
    ├── y_test.txt
    └── Inertial Signals/      # 9 raw signal files
```

## Backend / API

A FastAPI service in `src/server/` classifies phone Recordings — accelerometer + gyroscope, 6 channels — into one of five Activities (`downstairs`, `sit`, `stand`, `upstairs`, `walk`), serving whichever trained Model Bundles are in `models/`.

```bash
HAR_MODELS_DIR=models PYTHONPATH=src uv run uvicorn server.app:build_default_app --factory --host 0.0.0.0 --port 8000
```

Full API contract, error codes, and unit-conversion gotchas: [`docs/api.md`](docs/api.md). Interactive reference once running: `http://<host>:8000/docs`. Runnable client example simulating a phone: [`examples/simulate_phone.py`](examples/simulate_phone.py).

## MLflow Tracking

View experiment results:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens dashboard at `http://localhost:5000`