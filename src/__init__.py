"""GTA VI Twitch viewer forecasting package.

A clean, modular, production-grade forecaster for make.fun's bounty
"Forecast GTA VI's launch-day Twitch viewers hour by hour".

The pipeline:
    1. features.py   - extract signals from hour-1 and comparable launches
    2. model.py      - multi-quantile model on the log scale
    3. main.py       - entry point: read /input, write /output
"""

__version__ = "2.0.0"
