# RS-SSL — Self-Supervised Learning for Human Activity Recognition

## Setup

```bash
uv sync
```

## Run the baseline

```bash
uv run python train_baseline.py
```

## View results in MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.