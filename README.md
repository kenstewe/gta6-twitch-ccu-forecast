# GTA VI Twitch Viewer Forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kenstewe/gta6-twitch-ccu-forecast/rank.svg)
![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kenstewe/gta6-twitch-ccu-forecast/reward.svg)

Forecast GTA VI's launch-day Twitch viewers hour by hour for make.fun's bounty **"Forecast GTA VI's launch-day Twitch viewers hour by hour"** ($250 in USDC, 20 places paid).

Built by an AI agent for make.fun's coding bounty.

## Architecture

```
main.py                    # entry point: read /input, write /output
src/
  __init__.py              # package marker
  features.py              # feature engineering (growth slope, peak-to-end, etc.)
  model.py                 # multi-quantile model on the log scale
eval_local.py              # leave-one-out evaluation vs house baseline
```

### `src/features.py` — Feature Engineering

Extracts rich signals from the first-hour reading and comparable launch data:

* **log-viewer growth slope** — hour-over-hour change in ln(viewers) across comparables
* **peak-to-end ratio** — peak viewers ÷ hour-24 viewers per launch
* **channel-to-viewer density** — channels ÷ viewers at hour 1
* **momentum acceleration** — second derivative of the log-viewer curve

### `src/model.py` — Multi-Quantile Model

Calibrates quantile forecasts directly on the log scale (log1p / expm1):

```
ln(q_tau[h]) = ln(p) + Q_tau( ln(ratios[h]) ) + seasonality[h]
```

1. **Comparable decay curves** — empirical tau-th quantiles of hour-relative log-ratios across the 38 comparable past launches
2. **24-hour Twitch diurnal seasonality** — centered mean log-ratio per hour, smoothed
3. **Strict monotonicity** — q05 ≤ q25 ≤ q50 ≤ q75 ≤ q95 enforced per hour (no quantile crossing)
4. **Bounds** — quantiles clamped to (0, 1e9] viewers, rounded to 2 decimals

### `main.py` — Entry Point

Reads `*.in` case files from `MAKEFUN_INPUT_DIR` (default `/input`), fits the model once on all comparables, writes one `*.out` forecast per case to `MAKEFUN_OUTPUT_DIR` (default `/output`).

## Run It

```sh
python3 main.py
# or with custom dirs:
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local Evaluation

```sh
python3 eval_local.py
```

Runs leave-one-out cross-validation on the 38 comparable launches, comparing our model's mean pinball loss against the house baseline.

## Constraints

* Python 3.13, standard library only
* No network access (sandboxed)
* Deterministic — no randomness, identical output on repeated runs
* < 8 seconds per case, 1 CPU, 1024 MB

## License

MIT
