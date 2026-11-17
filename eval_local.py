"""Local evaluation: leave-one-out cross-validation on comparable launches.

For each comparable launch c:
  1. Train the model on all *other* comparables.
  2. Predict hours 2-24 for c using c's own hour-1 viewers as ``p``.
  3. Compute mean pinball loss on the log scale against c's true viewers.

Do the same for the house baseline. Report both losses so we can verify
our model beats the house forecast before submitting.

This is a valid local proxy because the practice cases exclude the four
practice launches from their comparables, and the final case excludes the
final itself -- so LOO within comparables mimics the real scoring setup.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import MultiQuantileForecaster, house_baseline_forecast  # noqa: E402

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]


def pinball_loss(y: float, q: float, tau: float) -> float:
    """Pinball loss on the log scale: 2 * max(tau*d, (tau-1)*d)."""
    y = max(y, 1.0)
    q = max(q, 1.0e-9)
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def load_case(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hour_1_viewers(comparable: dict) -> float:
    """Extract hour-1 viewers from a comparable's hourly data."""
    for pt in comparable.get("hourly") or []:
        try:
            h = int(pt["from"][11:13])
        except (ValueError, KeyError, IndexError):
            continue
        if h == 1:
            v = pt.get("viewers")
            return float(v) if v and v > 0 else 0.0
    return 0.0


def true_viewers_by_hour(comparable: dict) -> dict:
    """Extract {hour: viewers} from a comparable's hourly data."""
    out: dict = {}
    for pt in comparable.get("hourly") or []:
        try:
            h = int(pt["from"][11:13])
        except (ValueError, KeyError, IndexError):
            continue
        v = pt.get("viewers")
        if v is not None and v > 0:
            out[h] = float(v)
    return out


def evaluate_comparable(
    target: dict,
    train_comparables: list,
) -> tuple:
    """Evaluate one held-out comparable. Returns (model_loss, baseline_loss, n_scored)."""
    p = hour_1_viewers(target)
    if p <= 0:
        return None

    true = true_viewers_by_hour(target)
    scored_hours = [h for h in range(2, 25) if h in true]
    if not scored_hours:
        return None

    # ---- Our model ----
    model = MultiQuantileForecaster()
    model.fit(train_comparables)
    # Build a fake "known" block from the target's hour-1 readers
    known = {"viewers": p, "channels": 0, "reads": 0}
    model_forecast = model.predict(known)

    model_loss = 0.0
    n = 0
    for row in model_forecast:
        h = row["hour"]
        if h not in true:
            continue
        y = true[h]
        for tau, q in zip(TAUS, row["q"]):
            model_loss += pinball_loss(y, q, tau)
            n += 1

    # ---- House baseline ----
    baseline_forecast = house_baseline_forecast(p)
    baseline_loss = 0.0
    for row in baseline_forecast:
        h = row["hour"]
        if h not in true:
            continue
        y = true[h]
        for tau, q in zip(TAUS, row["q"]):
            baseline_loss += pinball_loss(y, q, tau)
            n += 1

    model_loss /= n
    baseline_loss /= n
    return model_loss, baseline_loss, n


def main() -> None:
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "input")
    in_files = sorted(glob.glob(os.path.join(input_dir, "*.in")))

    if not in_files:
        print(f"No practice cases found in {input_dir}")
        sys.exit(1)

    # Load comparables from the first case
    all_comparables: list = []
    for in_path in in_files:
        case = load_case(in_path)
        comps = case.get("comparables") or []
        if comps:
            all_comparables = comps
            break

    if not all_comparables:
        print("No comparables found in any case; cannot evaluate.")
        sys.exit(1)

    print(f"Found {len(all_comparables)} comparables. Running LOO evaluation...\n")

    model_losses: list = []
    baseline_losses: list = []
    skipped = 0

    for i, target in enumerate(all_comparables):
        train = all_comparables[:i] + all_comparables[i + 1 :]
        result = evaluate_comparable(target, train)
        if result is None:
            skipped += 1
            continue
        m_loss, b_loss, n = result
        model_losses.append(m_loss)
        baseline_losses.append(b_loss)

    if not model_losses:
        print("No comparables could be scored (missing hour-1 or hours 2-24).")
        sys.exit(1)

    avg_model = sum(model_losses) / len(model_losses)
    avg_baseline = sum(baseline_losses) / len(baseline_losses)
    wins = sum(1 for m, b in zip(model_losses, baseline_losses) if m < b)

    print(f"Scored {len(model_losses)} comparables ({skipped} skipped).")
    print(f"Our model mean pinball loss:   {avg_model:.6f}")
    print(f"House baseline mean loss:      {avg_baseline:.6f}")
    print(f"Improvement:                   {avg_baseline - avg_model:+.6f}")
    print(f"Our model won {wins}/{len(model_losses)} head-to-head.")

    if avg_model < avg_baseline:
        print("\n✅ Our model beats the house baseline on LOO evaluation.")
    else:
        print("\n⚠️  Our model does NOT beat the house baseline on LOO evaluation.")


if __name__ == "__main__":
    main()
