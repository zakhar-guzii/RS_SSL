"""Live end-to-end test: boots the real ``uvicorn`` process against the real
``models/`` directory (the actual trained bundles), not synthetic ones.

The rest of the suite proves the FastAPI wiring is correct with random-weight
bundles via ``TestClient``. This test is the only one that exercises the real
entry point (``build_default_app`` + ``HAR_MODELS_DIR``) and the real trained
weights over an actual HTTP socket, so it is what would catch a bundle that
loads in-process but breaks the live server (bad env var, startup crash,
serialization issue, etc.).
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
LABEL_ORDER = ["downstairs", "sit", "stand", "upstairs", "walk"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _recording(n_samples, hz=50):
    dt_ns = int(1e9 / hz)
    # 6-channel: accel (g) + gyro (rad/s)
    return [[i * dt_ns, 0.02, 0.01, 1.0, 0.05, -0.03, 0.01] for i in range(n_samples)]


@pytest.fixture(scope="module")
def live_server():
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "server.app:build_default_app",
            "--factory", "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "HAR_MODELS_DIR": str(MODELS_DIR), "PYTHONPATH": "src"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"uvicorn exited early:\n{proc.stdout.read()}")
            try:
                if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.3)
        else:
            raise TimeoutError("live server did not become healthy in time")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_live_models_lists_all_real_bundles(live_server):
    resp = httpx.get(f"{live_server}/models", timeout=5)
    assert resp.status_code == 200
    models = resp.json()
    ids = {m["id"] for m in models}
    assert ids == {p.name for p in MODELS_DIR.iterdir() if (p / "meta.json").exists()}
    assert sum(m["is_default"] for m in models) == 1


def test_live_predict_returns_prediction_for_every_bundle(live_server):
    models = httpx.get(f"{live_server}/models", timeout=5).json()
    for model in models:
        resp = httpx.post(
            f"{live_server}/predict",
            json={"model_id": model["id"], "units": "g", "samples": _recording(400)},
            timeout=10,
        )
        assert resp.status_code == 200, (model["id"], resp.text)
        body = resp.json()
        assert body["activity"] in LABEL_ORDER
        assert set(body["probabilities"]) == set(LABEL_ORDER)
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
        assert len(body["window_labels"]) > 0


def test_live_predict_unknown_model_is_404(live_server):
    resp = httpx.post(
        f"{live_server}/predict",
        json={"model_id": "does_not_exist", "units": "g", "samples": _recording(400)},
        timeout=5,
    )
    assert resp.status_code == 404


def test_live_predict_too_short_is_422(live_server):
    resp = httpx.post(
        f"{live_server}/predict",
        json={"model_id": "baseline_cnn", "units": "g", "samples": _recording(50)},
        timeout=5,
    )
    assert resp.status_code == 422
