"""Deterministic statistics shared by modules. No LLM involvement."""
from __future__ import annotations

import math
from typing import Sequence

from scipy.stats import norm



def safe_rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return float(lo), float(hi)


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """z statistic for p1 - p2 using the pooled standard error (0 when undefined)."""
    if n1 == 0 or n2 == 0:
        return 0.0
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (x1 / n1 - x2 / n2) / se


def one_sample_z(x: int, n: int, expected_rate: float) -> float:
    """z statistic for an observed proportion vs an externally expected rate."""
    if n == 0 or expected_rate <= 0 or expected_rate >= 1:
        return 0.0
    return (x / n - expected_rate) / math.sqrt(expected_rate * (1 - expected_rate) / n)


def standardized_rate(weights: Sequence[float], reference_rates: Sequence[float]) -> float:
    """Direct standardization: expected rate for a slice with `weights` (e.g. sessions per
    device) if each stratum behaved like the reference population."""
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(w * r for w, r in zip(weights, reference_rates)) / total


def pct_deviation(observed: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else (observed - baseline) / baseline


# Frequency bins by share of all tagged evidence (customer-signal-synthesizer.md, frequency 1-5).
FREQUENCY_BINS = (0.02, 0.05, 0.10, 0.20)


def frequency_score(count: int, total: int, bins: Sequence[float] = FREQUENCY_BINS) -> int:
    """Bin a theme's share of evidence into 1-5: <2%, <5%, <10%, <20%, >=20%."""
    if count <= 0 or total <= 0:
        return 1
    share = count / total
    return 1 + sum(share >= edge for edge in bins)


def score_to_strength(value: float, edges: Sequence[float]) -> int:
    """Map a value onto a 1-5 strength using four ascending edges."""
    return 1 + sum(value >= e for e in edges)


def sample_size_per_arm(baseline: float, mde_relative: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Visitors per arm for a two-sided two-proportion test (normal approximation, pooled
    variance under H0). Returns 0 when the inputs cannot define a test."""
    p1 = baseline
    p2 = baseline * (1 + mde_relative)
    if not (0 < p1 < 1) or not (0 < p2 < 1) or p1 == p2:
        return 0
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p2 - p1) ** 2))


def weeks_to_sample(n_per_arm: int, weekly_volume: float, arms: int = 2) -> float:
    """Weeks of eligible volume needed to fill every arm (inf when there is no volume)."""
    if weekly_volume <= 0:
        return math.inf
    return arms * n_per_arm / weekly_volume
