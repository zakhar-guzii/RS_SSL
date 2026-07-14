"""Tests for the Backend HTTP contract (GET /health, GET /models, POST /predict)."""

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from baseline_cnn import CNNClassifier
from server.app import create_app
from server.bundle import load_bundles

LABEL_ORDER = ["downstairs", "sit", "stand", "upstairs", "walk"]
ARCH = {
    "type": "cnn", "input_channels": 6, "conv_out_channels": [16, 32],
    "kernel_size": 5, "padding": 2, "pool_kernel": 2,
    "num_classes": 5, "hidden_dim": 32, "dropout": 0.4,
}


def _write_bundle(root, bundle_id, is_default, ssl):
    d = root / bundle_id
    d.mkdir(parents=True)
    model = CNNClassifier(
        input_channels=6, conv_out_channels=[16, 32], kernel_size=5,
        padding=2, pool_kernel=2, num_classes=5, hidden_dim=32, dropout=0.4,
    )
    torch.save(model.state_dict(), d / "weights.pt")
    (d / "meta.json").write_text(json.dumps({
        "id": bundle_id, "display_name": f"Model {bundle_id}",
        "description": "test", "ssl_pretrained": ssl, "is_default": is_default,
        "arch": ARCH, "norm_mean": [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
        "norm_std": [[[0.3, 0.3, 0.3, 0.5, 0.5, 0.5]]], "label_order": LABEL_ORDER,
    }))


def _recording(n_samples, hz=50, seed=0):
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t = np.arange(n_samples, dtype=np.int64) * dt_ns
    acc = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(n_samples, 3))
    gyro = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.5, size=(n_samples, 3))
    return [[int(t[i]), *acc[i].tolist(), *gyro[i].tolist()] for i in range(n_samples)]


@pytest.fixture
def client(tmp_path):
    _write_bundle(tmp_path, "cnn_baseline", is_default=True, ssl=False)
    _write_bundle(tmp_path, "cnn_ssl", is_default=False, ssl=True)
    app = create_app(load_bundles(tmp_path))
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    models = {m["id"]: m for m in r.json()}
    assert set(models) == {"cnn_baseline", "cnn_ssl"}
    assert models["cnn_baseline"]["is_default"] is True
    assert models["cnn_ssl"]["ssl_pretrained"] is True
    assert set(models["cnn_baseline"]) >= {"id", "display_name", "ssl_pretrained", "is_default"}


def test_predict_returns_prediction(client):
    r = client.post("/predict", json={
        "model_id": "cnn_baseline", "units": "g", "samples": _recording(300),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["activity"] in LABEL_ORDER
    assert set(body["probabilities"].keys()) == set(LABEL_ORDER)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert all(lbl in LABEL_ORDER for lbl in body["window_labels"])


def test_predict_unknown_model_is_404(client):
    r = client.post("/predict", json={
        "model_id": "does_not_exist", "units": "g", "samples": _recording(300),
    })
    assert r.status_code == 404


def test_predict_too_short_is_422(client):
    r = client.post("/predict", json={
        "model_id": "cnn_baseline", "units": "g", "samples": _recording(50),
    })
    assert r.status_code == 422
