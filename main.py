"""GTA VI launch-day Twitch viewer forecaster entry point.

Reads case files from MAKEFUN_INPUT_DIR (or /input), writes one forecast
file per case to MAKEFUN_OUTPUT_DIR (or /output).

Usage:
    python3 main.py
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Architecture:
    src/features.py  - feature engineering (growth slope, peak-to-end, etc.)
    src/model.py     - multi-quantile model on the log scale
    main.py          - entry point: read /input, write /output

Standard library only. No network. Deterministic (seeded, no randomness).
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import MultiQuantileForecaster  # noqa: E402


def load_case(path: str) -> dict:
    """Load a case JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_forecast(output_dir: str, case_id: str, forecast: dict) -> None:
    """Write a forecast to /output/<id>.out."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}.out")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forecast, f)


def main() -> None:
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    in_files = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    if not in_files:
        print(f"No .in files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect comparables from the first case that has them.
    # Comparables are identical across cases within a phase.
    all_comparables: list = []
    first_case = None
    for in_path in in_files:
        case = load_case(in_path)
        first_case = case
        comps = case.get("comparables") or []
        if comps:
            all_comparables = comps
            break

    if first_case is None:
        print("No cases to process", file=sys.stderr)
        sys.exit(1)

    # Fit the model once on all available comparables.
    model = MultiQuantileForecaster()
    if all_comparables:
        model.fit(all_comparables)
    else:
        print(
            "Warning: no comparables found; forecasts will use "
            "the median decay curve only.",
            file=sys.stderr,
        )

    for in_path in in_files:
        case = load_case(in_path)
        case_id = case.get("id") or os.path.splitext(os.path.basename(in_path))[0]
        known = case.get("known") or {}

        forecast_rows = model.predict(known, all_comparables)
        forecast = {"hours": forecast_rows}

        write_forecast(output_dir, case_id, forecast)
        print(f"Wrote forecast for case '{case_id}' -> {case_id}.out")


if __name__ == "__main__":
    main()
