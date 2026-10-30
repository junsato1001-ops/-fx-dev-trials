"""Inference for the Development Gate (SPEC v0.3 §8, §10; EVAL-SPEC-V1 §5).

- Stationary bootstrap (Politis-Romano), circular, mean block length L, B replicates,
  numpy Generator(PCG64(seed)) re-initialized for every test.
  Index generation (fixed here, before any result): for each replicate, draw u ~ U(0,1)^T and
  s ~ Uniform{0..T-1}^T in that order; position 0 always starts a block; position i>0 starts a new
  block iff u[i] < 1/L; a block started at position p begins at s[p] and continues circularly.
- One-sided p-values (EVAL-SPEC-V1 §5.2):
  p_sup = (1 + #{b: D*_b - D <= D}) / (B + 1);  p_inf = (1 + #{b: D*_b - D >= D}) / (B + 1)
- HAC: Newey-West with Bartlett kernel, lag = 4, variance of the mean; one-sided normal p-values.
"""
from __future__ import annotations

import math

import numpy as np


def stationary_bootstrap_means(d: np.ndarray, mean_block: float, B: int, seed: int) -> np.ndarray:
    T = d.size
    rng = np.random.Generator(np.random.PCG64(seed))
    p = 1.0 / mean_block
    out = np.empty(B, dtype=np.float64)
    pos = np.arange(T)
    for b in range(B):
        u = rng.random(T)
        s = rng.integers(0, T, size=T)
        new = u < p
        new[0] = True
        starts_at = np.flatnonzero(new)
        grp = np.cumsum(new) - 1
        first_pos = starts_at[grp]
        idx = (s[first_pos] + (pos - first_pos)) % T
        out[b] = d[idx].mean()
    return out


def bootstrap_test(d: np.ndarray, mean_block: float, B: int, seed: int) -> dict:
    D = float(d.mean())
    Dstar = stationary_bootstrap_means(d, mean_block, B, seed)
    p_sup = (1 + int(np.count_nonzero(Dstar - D <= D))) / (B + 1)
    p_inf = (1 + int(np.count_nonzero(Dstar - D >= D))) / (B + 1)
    return {"D": D, "p_sup": p_sup, "p_inf": p_inf, "mean_block": mean_block, "B": B, "seed": seed,
            "ci90_lower": float(np.percentile(Dstar, 5)), "ci90_upper": float(np.percentile(Dstar, 95))}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def hac_test(d: np.ndarray, lag: int) -> dict:
    T = d.size
    D = float(d.mean())
    e = d - D
    g0 = float(np.dot(e, e)) / T
    var = g0
    for l in range(1, lag + 1):
        gl = float(np.dot(e[l:], e[:-l])) / T
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    se = math.sqrt(var / T) if var > 0 else float("nan")
    t = D / se if se and se == se else float("nan")
    p_sup = _norm_cdf(t) if t == t else float("nan")
    p_inf = 1.0 - p_sup if p_sup == p_sup else float("nan")
    return {"D": D, "t": t, "p_sup": p_sup, "p_inf": p_inf, "lag": lag}


def verdict(p_sup: float, p_inf: float, alpha: float) -> str:
    if p_sup < alpha and p_inf < alpha:
        return "INFERENCE_ERROR"
    if p_sup < alpha:
        return "SUP"
    if p_inf < alpha:
        return "INF"
    return "INCONCLUSIVE"


def disagreement(v1: str, v2: str) -> bool:
    return {v1, v2} == {"SUP", "INF"}


def halves(n: int) -> tuple[int, int]:
    """First/second half sizes; the second half gets the extra element when n is odd."""
    first = n // 2
    return first, n - first
