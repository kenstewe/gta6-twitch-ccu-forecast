"""Feature engineering for GTA VI Twitch viewer forecasting.

Extracts rich signals from the first-hour reading and comparable
launch-day data, including:

* log-viewer growth slope
* peak-to-end ratio
* channel-to-viewer density
* momentum acceleration

All functions are pure, deterministic, and standard-library only.
"""

from __future__ import annotations

from typing import Dict, List

import math


def extract_first_hour_signals(known: Dict) -> Dict[str, float]:
    """Extract signals from the hour-1 reading.

    Parameters
    ----------
    known : dict
        The ``known`` block from a case file. Expected keys: ``viewers``,
        ``channels``, ``reads`` (hour 1 only).

    Returns
    -------
    dict
        ``p`` (hour-1 viewers, floored at 1), ``log_p`` (log1p of p),
        ``channels``, ``reads``, and ``viewers_per_channel``.
    """
    p = float(known.get("viewers", 0.0) or 0.0)
    channels = float(known.get("channels", 0.0) or 0.0)
    reads = float(known.get("reads", 0.0) or 0.0)

    p = max(p, 1.0)  # avoid log(0); house baseline does the same
    viewers_per_channel = p / channels if channels > 0 else p

    return {
        "p": p,
        "log_p": math.log1p(p),
        "channels": channels,
        "reads": reads,
        "viewers_per_channel": viewers_per_channel,
    }


def _hour_map(hourly: List[Dict]) -> Dict[int, float]:
    """Convert an ``hourly`` list into {hour: viewers}, skipping bad rows."""
    by_hour: Dict[int, float] = {}
    for pt in hourly:
        try:
            h = int(pt["from"][11:13])  # "YYYY-MM-DDTHH:MM:SSZ" -> HH
        except (ValueError, KeyError, IndexError):
            continue
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        by_hour[h] = float(v)
    return by_hour


def compute_hourly_log_ratios(comparables: List[Dict]) -> Dict[int, List[float]]:
    """Compute ln(viewers[h] / viewers[1]) for each comparable, per hour.

    Parameters
    ----------
    comparables : list of dict
        Each comparable has an ``hourly`` list of {from, viewers, channels}.

    Returns
    -------
    dict
        Mapping hour (2-24) -> list of log-ratios across comparables that
        have data for both hour 1 and that hour.
    """
    log_ratios: Dict[int, List[float]] = {h: [] for h in range(2, 25)}

    for comp in comparables:
        by_hour = _hour_map(comp.get("hourly") or [])
        v1 = by_hour.get(1)
        if v1 is None or v1 <= 0:
            continue
        for h in range(2, 25):
            vh = by_hour.get(h)
            if vh is not None and vh > 0:
                log_ratios[h].append(math.log(vh / v1))

    return log_ratios


def compute_growth_slope(log_ratios: Dict[int, List[float]]) -> Dict[int, float]:
    """Average hour-over-hour log-viewer slope across comparables.

    slope[h] = mean( log_ratio[h] - log_ratio[h-1] ).
    Positive means viewership is growing into hour h; negative means decay.
    """
    slopes: Dict[int, float] = {}
    for h in range(2, 25):
        vals = log_ratios.get(h, [])
        prev = log_ratios.get(h - 1, [])
        n = min(len(vals), len(prev))
        if n == 0:
            slopes[h] = 0.0
            continue
        slopes[h] = sum(vals[i] - prev[i] for i in range(n)) / n
    return slopes


def compute_peak_to_end_ratio(comparables: List[Dict]) -> List[float]:
    """peak_viewers / viewers[24] for each comparable that has both."""
    ratios: List[float] = []
    for comp in comparables:
        by_hour = _hour_map(comp.get("hourly") or [])
        v1 = by_hour.get(1)
        v24 = by_hour.get(24)
        if v1 is None or v1 <= 0 or v24 is None or v24 <= 0:
            continue
        peak = max(by_hour.values())
        ratios.append(peak / v24)
    return ratios


def compute_channel_density(comparables: List[Dict]) -> List[float]:
    """channels / viewers at hour 1 for each comparable."""
    densities: List[float] = []
    for comp in comparables:
        c1 = None
        for pt in comp.get("hourly") or []:
            try:
                h = int(pt["from"][11:13])
            except (ValueError, KeyError, IndexError):
                continue
            if h != 1:
                continue
            v = pt.get("viewers")
            c = pt.get("channels")
            if v and c and v > 0:
                c1 = float(c)
            break
        by_hour = _hour_map(comp.get("hourly") or [])
        v1 = by_hour.get(1)
        if v1 is None or v1 <= 0 or c1 is None:
            continue
        densities.append(c1 / v1)
    return densities


def compute_momentum_acceleration(
    log_ratios: Dict[int, List[float]],
) -> Dict[int, float]:
    """Second derivative of the log-viewer curve per hour.

    accel[h] = mean( log_ratio[h] - 2*log_ratio[h-1] + log_ratio[h-2] ).
    Positive means the decay is slowing (curve convex); negative means
    accelerating decay.
    """
    accel: Dict[int, float] = {}
    for h in range(3, 25):
        v0 = log_ratios.get(h, [])
        v1 = log_ratios.get(h - 1, [])
        v2 = log_ratios.get(h - 2, [])
        n = min(len(v0), len(v1), len(v2))
        if n == 0:
            accel[h] = 0.0
            continue
        accel[h] = sum(v0[i] - 2.0 * v1[i] + v2[i] for i in range(n)) / n
    return accel


def summarize_features(
    known: Dict,
    comparables: List[Dict],
) -> Dict:
    """Convenience: compute all feature groups at once.

    Returns a nested dict with keys: first_hour, growth_slope,
    peak_to_end, channel_density, momentum_acceleration, hourly_log_ratios.
    """
    log_ratios = compute_hourly_log_ratios(comparables)
    return {
        "first_hour": extract_first_hour_signals(known),
        "growth_slope": compute_growth_slope(log_ratios),
        "peak_to_end": compute_peak_to_end_ratio(comparables),
        "channel_density": compute_channel_density(comparables),
        "momentum_acceleration": compute_momentum_acceleration(log_ratios),
        "hourly_log_ratios": log_ratios,
    }
