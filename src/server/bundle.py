"""Model Bundle — one deployable trained model the Backend serves.

A bundle is a directory ``<id>/`` with ``weights.pt`` (state_dict) and
``meta.json`` (norm stats, label order, architecture config, display metadata).
The Backend scans a ``models/`` directory of these at startup.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .architectures import build_model
from .predictor import Predictor


@dataclass
class ModelBundle:
    id: str
    display_name: str
    description: str
    ssl_pretrained: bool
    is_default: bool
    predictor: Predictor


def load_bundle(path: Path) -> ModelBundle:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())

    model = build_model(meta["arch"])
    state = torch.load(path / "weights.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    predictor = Predictor(
        model=model,
        norm_mean=np.asarray(meta["norm_mean"], dtype=np.float32),
        norm_std=np.asarray(meta["norm_std"], dtype=np.float32),
        label_order=meta["label_order"],
    )

    return ModelBundle(
        id=meta["id"],
        display_name=meta["display_name"],
        description=meta.get("description", ""),
        ssl_pretrained=bool(meta["ssl_pretrained"]),
        is_default=bool(meta["is_default"]),
        predictor=predictor,
    )


def load_bundles(models_dir: Path) -> Dict[str, ModelBundle]:
    """Scan ``models_dir`` for bundle subdirectories. Fail loudly if the set is
    malformed (duplicate ids, or not exactly one default)."""
    models_dir = Path(models_dir)
    bundles: Dict[str, ModelBundle] = {}

    for child in sorted(models_dir.iterdir()):
        if not (child / "meta.json").exists():
            continue
        bundle = load_bundle(child)
        if bundle.id in bundles:
            raise ValueError(f"Duplicate Model Bundle id: {bundle.id!r}")
        bundles[bundle.id] = bundle

    if bundles:
        defaults = [b for b in bundles.values() if b.is_default]
        if len(defaults) != 1:
            raise ValueError(
                f"Exactly one Model Bundle must be default, found {len(defaults)}"
            )

    return bundles
