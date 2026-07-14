# HAR Backend — CPU-only inference image for Render (or any container host).
#
# The model classes live in the training modules (baseline_cnn, encoder,
# ssl_pretrain, transformer), which import mlflow / pandas / sklearn / matplotlib
# at import time. All six bundles load at startup, so the full stack below is
# genuinely required until those nn.Module definitions are split into a
# serving-only module.
FROM python:3.13-slim

WORKDIR /app

# libgomp1 is needed by torch/scipy OpenMP at runtime on debian-slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST, from the PyTorch CPU index, so the ~2.5GB CUDA wheel is
# never pulled. Keep this as its own layer/command — the CPU index does not host
# the other packages.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Remaining runtime deps (versions float; pin here if you need reproducibility).
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    mlflow \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml

# Only what the server needs at runtime (see .dockerignore for the rest).
COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src \
    HAR_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

# Shell form so $PORT (injected by Render) expands; falls back to 8000 locally.
CMD uvicorn server.app:build_default_app --factory --host 0.0.0.0 --port ${PORT:-8000}
