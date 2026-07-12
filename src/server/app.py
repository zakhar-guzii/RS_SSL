"""The Backend HTTP surface.

A standalone FastAPI service that serves Model Bundles and classifies
Recordings. MVP: no auth, CORS open, phone and laptop on a shared Wi-Fi.
Recordings are classified and discarded — never persisted.
"""

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from .bundle import ModelBundle, load_bundles
from .predictor import TooShortError


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    units: str  # "g" or "m/s2"
    samples: List[List[float]]  # rows of [t_ns, x, y, z]


def create_app(bundles: Dict[str, ModelBundle]) -> FastAPI:
    app = FastAPI(title="HAR Activity-Recognition Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/models")
    def list_models():
        return [
            {
                "id": b.id,
                "display_name": b.display_name,
                "ssl_pretrained": b.ssl_pretrained,
                "is_default": b.is_default,
            }
            for b in bundles.values()
        ]

    @app.post("/predict")
    def predict(req: PredictRequest):
        bundle = bundles.get(req.model_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {req.model_id}")
        try:
            return bundle.predictor.predict(req.samples, units=req.units)
        except TooShortError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app


def build_default_app() -> FastAPI:
    """Entry point for ``uvicorn``: load bundles from the models directory."""
    models_dir = Path(os.environ.get("HAR_MODELS_DIR", "models"))
    bundles = load_bundles(models_dir)
    return create_app(bundles)
