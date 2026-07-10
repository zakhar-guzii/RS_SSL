"""Seam 2 — Predictor: Recording -> Prediction.

Composes the Canonical Signal transform (Seam 1), per-Model-Bundle
normalization, model inference, and aggregation. A Recording yields many
Windows; the Prediction is the mean of per-Window softmax, argmax'd over the
whole Recording, plus per-class probabilities and the per-Window label
sequence.
"""

from typing import List, Sequence

import numpy as np
import torch

from .preprocessing import to_canonical_windows


class TooShortError(ValueError):
    """Recording yielded no Windows (shorter than ~2.56 s of signal)."""


class Predictor:
    def __init__(self, model, norm_mean, norm_std, label_order: Sequence[str]):
        self.model = model
        self.model.eval()
        self.norm_mean = np.asarray(norm_mean, dtype=np.float32)
        self.norm_std = np.asarray(norm_std, dtype=np.float32)
        self.label_order = list(label_order)

    def predict(self, samples: Sequence[Sequence[float]], units: str) -> dict:
        windows = to_canonical_windows(samples, units=units)
        if windows.shape[0] == 0:
            raise TooShortError("Recording is too short to form a Window")

        normed = (windows - self.norm_mean) / self.norm_std
        x = torch.from_numpy(normed.astype(np.float32))

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        mean_probs = probs.mean(axis=0)
        top_idx = int(mean_probs.argmax())
        window_labels = [self.label_order[i] for i in probs.argmax(axis=1)]

        return {
            "activity": self.label_order[top_idx],
            "confidence": float(mean_probs[top_idx]),
            "probabilities": {
                label: float(p) for label, p in zip(self.label_order, mean_probs)
            },
            "window_labels": window_labels,
        }
