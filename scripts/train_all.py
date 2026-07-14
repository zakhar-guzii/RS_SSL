#!/usr/bin/env python3
"""Train every HAR Model Bundle and export it to ``models/``.

Runs each training entry point as an isolated subprocess (so one failure or a
leaked MLflow run can't take the others down), streams its output live, tees a
per-model log to ``logs/train_all/``, and prints a summary at the end verifying
each exported bundle is 6-channel.

Usage:
    python scripts/train_all.py                 # all six models
    python scripts/train_all.py baseline_cnn     # just one (or several)
    python scripts/train_all.py --supervised     # CNN, CNN-LSTM, Transformer
    python scripts/train_all.py --ssl            # the three SSL variants
    python scripts/train_all.py --list           # show the models and exit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
LOG_DIR = REPO_ROOT / "logs" / "train_all"
EXPECTED_CHANNELS = 6

# bundle_id -> (python -m module, group). Order matters: supervised first (fast,
# proves the pipeline), SSL last (slow: pretrain + fine-tune).
MODELS: dict[str, tuple[str, str]] = {
    "baseline_cnn": ("baseline_cnn", "supervised"),
    "cnn_lstm": ("encoder", "supervised"),
    "transformer": ("transformer.transformer", "supervised"),
    "ssl_data2vec": ("ssl_pretrain", "ssl"),
    "ssl_simclr": ("ssl_pretrain_contrastive", "ssl"),
    "ssl_dinov2": ("ssl_pretrain_dinov2", "ssl"),
}


def _run_one(bundle_id: str, module: str) -> tuple[int, float]:
    """Run one training entry point, streaming + logging output. Returns (rc, secs)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{bundle_id}.log"
    env = {**os.environ, "PYTHONPATH": "src"}

    print(f"\n{'=' * 70}\n▶ Training {bundle_id}  (python -m {module})\n  log: {log_path}\n{'=' * 70}", flush=True)
    start = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
    return proc.returncode, time.monotonic() - start


def _verify_bundle(bundle_id: str) -> str:
    """Human-readable check that the exported bundle exists and is 6-channel."""
    meta_path = MODELS_DIR / bundle_id / "meta.json"
    if not meta_path.exists():
        return "NO BUNDLE (meta.json missing)"
    meta = json.loads(meta_path.read_text())
    ch = meta.get("arch", {}).get("input_channels") or meta.get("arch", {}).get("in_channels")
    default = " [default]" if meta.get("is_default") else ""
    if ch != EXPECTED_CHANNELS:
        return f"WRONG CHANNELS: input_channels={ch} (expected {EXPECTED_CHANNELS})"
    return f"ok · {ch}ch{default}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="bundle ids to train (default: all)")
    parser.add_argument("--supervised", action="store_true", help="train only the supervised models")
    parser.add_argument("--ssl", action="store_true", help="train only the SSL models")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()

    if args.list:
        for bid, (mod, group) in MODELS.items():
            print(f"  {bid:15s} {group:11s} python -m {mod}")
        return 0

    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            parser.error(f"unknown model id(s): {', '.join(unknown)}. Choose from: {', '.join(MODELS)}")
        selected = list(args.models)
    elif args.supervised:
        selected = [b for b, (_, g) in MODELS.items() if g == "supervised"]
    elif args.ssl:
        selected = [b for b, (_, g) in MODELS.items() if g == "ssl"]
    else:
        selected = list(MODELS)

    print(f"Training {len(selected)} model(s): {', '.join(selected)}")
    results = []
    for bundle_id in selected:
        module, _ = MODELS[bundle_id]
        rc, secs = _run_one(bundle_id, module)
        results.append((bundle_id, rc, secs, _verify_bundle(bundle_id)))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'model':16s} {'exit':>4s} {'time':>8s}  bundle")
    all_ok = True
    for bundle_id, rc, secs, bundle in results:
        ok = rc == 0 and bundle.startswith("ok")
        all_ok &= ok
        mins = f"{secs / 60:.1f}m"
        print(f"{bundle_id:16s} {rc:>4d} {mins:>8s}  {bundle}")

    # Exactly one default across all bundles present in models/ (server startup invariant).
    defaults = [p.name for p in MODELS_DIR.iterdir()
                if (p / "meta.json").exists()
                and json.loads((p / "meta.json").read_text()).get("is_default")]
    print(f"\ndefault bundle(s): {defaults or '— none —'}"
          + ("" if len(defaults) == 1 else "   ⚠ server requires exactly one"))

    return 0 if (all_ok and len(defaults) == 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
