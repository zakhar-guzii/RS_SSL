"""Tests for Model Bundle loading (the deployable unit the Backend serves)."""

import json

import numpy as np
import torch

from baseline_cnn import CNNClassifier
from server.bundle import load_bundle, load_bundles

LABEL_ORDER = ["downstairs", "sit", "stand", "upstairs", "walk"]

ARCH = {
    "type": "cnn",
    "input_channels": 3,
    "conv_out_channels": [16, 32],
    "kernel_size": 5,
    "padding": 2,
    "pool_kernel": 2,
    "num_classes": 5,
    "hidden_dim": 32,
    "dropout": 0.4,
}


def _write_bundle(root, bundle_id, is_default=True, ssl=False):
    d = root / bundle_id
    d.mkdir(parents=True)
    model = CNNClassifier(
        input_channels=ARCH["input_channels"],
        conv_out_channels=ARCH["conv_out_channels"],
        kernel_size=ARCH["kernel_size"],
        padding=ARCH["padding"],
        pool_kernel=ARCH["pool_kernel"],
        num_classes=ARCH["num_classes"],
        hidden_dim=ARCH["hidden_dim"],
        dropout=ARCH["dropout"],
    )
    torch.save(model.state_dict(), d / "weights.pt")
    meta = {
        "id": bundle_id,
        "display_name": f"Model {bundle_id}",
        "description": "test bundle",
        "ssl_pretrained": ssl,
        "is_default": is_default,
        "arch": ARCH,
        "norm_mean": [[[0.0, 0.0, 1.0]]],
        "norm_std": [[[0.3, 0.3, 0.3]]],
        "label_order": LABEL_ORDER,
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _uniform_recording_g(n_samples, hz=50, seed=0):
    rng = np.random.default_rng(seed)
    dt_ns = int(1e9 / hz)
    t = np.arange(n_samples, dtype=np.int64) * dt_ns
    xyz = rng.normal(loc=[0.0, 0.0, 1.0], scale=0.1, size=(n_samples, 3))
    return [[int(t[i]), float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])]
            for i in range(n_samples)]


def test_load_bundle_reconstructs_model_and_metadata(tmp_path):
    d = _write_bundle(tmp_path, "cnn_baseline", is_default=True, ssl=False)

    bundle = load_bundle(d)

    assert bundle.id == "cnn_baseline"
    assert bundle.display_name == "Model cnn_baseline"
    assert bundle.ssl_pretrained is False
    assert bundle.is_default is True

    pred = bundle.predictor.predict(_uniform_recording_g(300), units="g")
    assert pred["activity"] in LABEL_ORDER
    assert set(pred["probabilities"].keys()) == set(LABEL_ORDER)


def test_load_bundles_scans_directory(tmp_path):
    _write_bundle(tmp_path, "cnn_baseline", is_default=True)
    _write_bundle(tmp_path, "cnn_ssl", is_default=False, ssl=True)

    bundles = load_bundles(tmp_path)

    assert set(bundles.keys()) == {"cnn_baseline", "cnn_ssl"}
    defaults = [b for b in bundles.values() if b.is_default]
    assert len(defaults) == 1
