#!/usr/bin/env python3
"""Simulates a phone Recording against the HAR Backend.

Builds a synthetic ~15s walking-like 6-channel stream (accelerometer +
gyroscope), POSTs it to /predict exactly as the real App would, and prints the
Prediction. Doubles as a runnable reference for the API contract described in
docs/api.md.

Stdlib only (urllib) so it runs with no extra installs beyond Python 3.

Usage:
    python examples/simulate_phone.py
    python examples/simulate_phone.py --url http://192.168.1.23:8000 --model-id transformer
"""

import argparse
import json
import math
import urllib.error
import urllib.request

HZ = 50
NS_PER_SAMPLE = int(1e9 / HZ)
G = 9.80665


def build_recording(duration_s: float, units: str) -> list:
    """A synthetic walking-like signal: ~1g vertical + a stride oscillation,
    plus a matching gyroscope (angular velocity, rad/s) stride wobble.

    Each row is 6-channel ``[t_ns, ax, ay, az, gx, gy, gz]``: accelerometer in
    ``units`` (g or m/s2), gyroscope always in rad/s.
    """
    n_samples = int(duration_s * HZ)
    samples = []
    for i in range(n_samples):
        t_ns = i * NS_PER_SAMPLE
        stride = math.sin(2 * math.pi * 1.8 * i / HZ)
        # Accelerometer (g)
        ax = 0.15 * stride
        ay = 0.1 * math.sin(2 * math.pi * 0.9 * i / HZ)
        az = 1.0 + 0.25 * stride
        if units == "m/s2":
            ax, ay, az = ax * G, ay * G, az * G
        # Gyroscope (rad/s) — never unit-converted
        gx = 0.6 * math.sin(2 * math.pi * 1.8 * i / HZ + 0.5)
        gy = 0.4 * math.sin(2 * math.pi * 0.9 * i / HZ)
        gz = 0.2 * stride
        samples.append([t_ns, ax, ay, az, gx, gy, gz])
    return samples


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post(url: str, payload: dict) -> tuple:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--model-id", default=None, help="Model Bundle id (default: the bundle marked is_default)")
    parser.add_argument("--units", default="g", choices=["g", "m/s2"], help="Units the simulated phone reports in")
    parser.add_argument("--duration", type=float, default=15.0, help="Recording length in seconds")
    args = parser.parse_args()

    models = get(f"{args.url}/models")
    print(f"Available models: {[m['id'] for m in models]}")

    model_id = args.model_id or next(m["id"] for m in models if m["is_default"])
    print(f"Using model_id={model_id!r}, units={args.units!r}, duration={args.duration}s")

    samples = build_recording(args.duration, args.units)
    status, body = post(
        f"{args.url}/predict",
        {"model_id": model_id, "units": args.units, "samples": samples},
    )
    if status != 200:
        print(f"Error {status}: {body}")
        raise SystemExit(1)

    print(f"\nActivity: {body['activity']} ({body['confidence']:.1%})")
    print("Probabilities:")
    for label, p in sorted(body["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:>10}: {p:.1%}")
    print(f"Windows classified: {len(body['window_labels'])}")


if __name__ == "__main__":
    main()
