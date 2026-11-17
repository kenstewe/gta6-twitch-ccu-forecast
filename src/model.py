from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TAU = [-1.645, -0.6745, 0.0, 0.6745, 1.645]

BASE_SIGMA = 0.15
MIN_VIEWERS = 0.01
MAX_VIEWERS = 1000000000.0


def _weighted_quantile(values: List[float], weights: List[float], tau: float) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x)
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return pairs[len(pairs) // 2]
    target = tau * total_w
    acc = 0.0
    for i, (val, w) in enumerate(pairs):
        prev_acc = acc
        acc += w
        if acc >= target:
            if i == 0 or w <= 0:
                return val
            frac = (target - prev_acc) / w
            prev_val, _ = pairs[i - 1]
            return prev_val + frac * (val - prev_val)
    return pairs[-1]


def _extract_p(known: Any) -> float:
    if isinstance(known, dict):
        return float(known.get("viewers") or 0.0)
    if isinstance(known, list) and known:
        for pt in known:
            if isinstance(pt, dict):
                try:
                    if pt.get("hour") == 1 or int(str(pt.get("from", ""))[11:13]) == 1:
                        v = pt.get("viewers")
                        if v is not None:
                            return float(v)
                except (ValueError, IndexError):
                    continue
        if isinstance(known, dict):
            return float(known.get("viewers") or 0.0)
    return 0.0


class MultiQuantileForecaster:
    def __init__(self, blend_median: float = 0.35, blend_spread: float = 0.25, sigma_mult: float = 1.45) -> None:
        self.blend_median = blend_median
        self.blend_spread = blend_spread
        self.sigma_mult = sigma_mult
        self.comp_ratios: List[Dict[int, float]] = []
        self.comp_log_p: List[float] = []

    def fit(self, comparables: List[Dict]) -> "MultiQuantileForecaster":
        self.comp_ratios = []
        self.comp_log_p = []
        for comp in comparables or []:
            by_h: Dict[int, float] = {}
            for pt in comp.get("hourly") or []:
                try:
                    h = int(pt["from"][11:13])
                    v = pt.get("viewers")
                    if v is not None and float(v) > 0:
                        by_h[h] = float(v)
                except (ValueError, KeyError, IndexError):
                    continue
            if 1 not in by_h or by_h.get(1, 0) <= 0:
                continue
            p_val = max(1.0, by_h.get(1, 1.0))
            log_p = math.log(p_val)
            ratios: Dict[int, float] = {}
            for h in range(2, 25):
                if h in by_h:
                    ratios[h] = math.log(max(1.0, by_h[h])) - log_p
            if ratios:
                self.comp_ratios.append(ratios)
                self.comp_log_p.append(log_p)
        return self

    def predict(self, known: Any, target_meta: Optional[Any] = None) -> List[Dict]:
        raw_p = _extract_p(known)
        p = max(1.0, raw_p)
        log_p = math.log(p)
        weights: List[float] = [math.exp(-0.25 * abs(log_p - clp)) for clp in self.comp_log_p]
        hours_out: List[Dict] = []
        for h in range(2, 25):
            vals_h = [ratios[h] for ratios in self.comp_ratios if h in ratios]
            wts_h = [w for ratios, w in zip(self.comp_ratios, weights) if h in ratios]
            if len(vals_h) >= 3:
                emp_med = _weighted_quantile(vals_h, wts_h, 0.5)
                emp_qs = [_weighted_quantile(vals_h, wts_h, tau) for tau in TAUS]
            else:
                emp_med = 0.0
                emp_qs = [z * BASE_SIGMA * self.sigma_mult * math.sqrt(h - 1) for z in Z_TAU]
            qs: List[float] = []
            for idx, z in enumerate(Z_TAU):
                base_offset = z * BASE_SIGMA * self.sigma_mult * math.sqrt(h - 1)
                emp_spread = emp_qs[idx] - emp_med
                blended_offset = self.blend_median * emp_med + self.blend_spread * emp_spread + (1.0 - self.blend_spread) * base_offset
                q = math.exp(log_p + blended_offset)
                qs.append(max(MIN_VIEWERS, min(MAX_VIEWERS, q)))
            monotonic_qs: List[float] = []
            running_max = MIN_VIEWERS
            for q in qs:
                if q < running_max:
                    q = running_max
                monotonic_qs.append(round(q, 2))
                running_max = q
            hours_out.append({"hour": h, "q": monotonic_qs})
        return hours_out


def house_baseline_forecast(p: float) -> List[Dict]:
    p = max(1.0, float(p))
    hours_out: List[Dict] = []
    for h in range(2, 25):
        qs = [max(MIN_VIEWERS, min(MAX_VIEWERS, p * math.exp(z * 0.15 * math.sqrt(h - 1)))) for z in Z_TAU]
        monotonic_qs: List[float] = []
        running_max = MIN_VIEWERS
        for q in qs:
            if q < running_max:
                q = running_max
            monotonic_qs.append(q)
            running_max = q
        hours_out.append({"hour": h, "q": monotonic_qs})
    return hours_out
