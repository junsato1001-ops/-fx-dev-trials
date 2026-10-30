# M0-RW0 / VOL_STATE_V1 reference implementation
# Freeze: M0 specification v1.2
# Standard library only.

from __future__ import annotations
from math import erf, log, sqrt
from statistics import NormalDist

LAMBDA = 0.94
WINDOW = 252
KAPPA = 0.25
MODEL_ID = "M0-RW0"
VOL_STATE_ID = "TRUNCATED_NORMALIZED_EWMA_252_L094"

class WarmupInsufficient(ValueError):
    pass

def isfinite(x):
    return x == x and x not in (float("inf"), float("-inf"))

def finite_ewma_variance(one_session_returns, lam=LAMBDA, window=WINDOW):
    """252本有限窓・正規化EWMA。平均は引かず、最新returnの重みを1とする。"""
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1)")
    if len(one_session_returns) < window:
        raise WarmupInsufficient(
            f"M0_VOL_WARMUP_INSUFFICIENT: need {window} returns, got {len(one_session_returns)}"
        )
    xs = [float(x) for x in one_session_returns[-window:]]
    if any(not isfinite(x) for x in xs):
        raise ValueError("non-finite return encountered")
    numerator = 0.0
    denominator = 0.0
    weight = 1.0
    for r in reversed(xs):  # newest first => k=0
        numerator += weight * r * r
        denominator += weight
        weight *= lam
    return numerator / denominator

def sigma_h_from_variance(v_1session, horizon_sessions):
    v = float(v_1session)
    h = int(horizon_sessions)
    if v < 0 or not isfinite(v):
        raise ValueError("invalid variance")
    if h < 1:
        raise ValueError("horizon_sessions must be >=1")
    return sqrt(h * v)

def m0_category_probabilities(kappa=KAPPA, sigma_h=1.0):
    if kappa < 0:
        raise ValueError("kappa must be >=0")
    if sigma_h < 0 or not isfinite(float(sigma_h)):
        raise ValueError("invalid sigma_h")
    if sigma_h == 0:
        return (0.0, 1.0, 0.0)
    phi = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    p_neutral = phi(kappa) - phi(-kappa)
    p_down = phi(-kappa)
    p_up = 1.0 - phi(kappa)
    return p_down, p_neutral, p_up

def prediction_interval(sigma_h, coverage=0.90):
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be in (0,1)")
    if sigma_h < 0:
        raise ValueError("sigma_h must be >=0")
    if sigma_h == 0:
        return (0.0, 0.0)
    alpha = (1.0 - coverage) / 2.0
    nd = NormalDist(mu=0.0, sigma=sigma_h)
    return nd.inv_cdf(alpha), nd.inv_cdf(1.0-alpha)

def walk_forward(dates, prices, horizon_sessions=1, lam=LAMBDA, window=WINDOW, kappa=KAPPA):
    """
    Canonical observations only, ascending and consecutive on the target's canonical calendar.
    For target-start index i:
      latest information = price[i-1]
      volatility history = window returns ending at price[i-1]
      target = log(price[i+h]/price[i])
    """
    if len(dates) != len(prices):
        raise ValueError("dates/prices length mismatch")
    if any(dates[j] >= dates[j+1] for j in range(len(dates)-1)):
        raise ValueError("dates must be strictly increasing")
    ps = [float(x) for x in prices]
    if any((not isfinite(x)) or x <= 0 for x in ps):
        raise ValueError("prices must be finite and >0")
    h = int(horizon_sessions)
    if h < 1:
        raise ValueError("horizon_sessions must be >=1")
    rets = [log(ps[j+1] / ps[j]) for j in range(len(ps)-1)]
    out = []
    for i in range(window + 1, len(ps) - h):
        history = rets[i-1-window:i-1]
        if len(history) != window:
            raise AssertionError("internal alignment error")
        v = finite_ewma_variance(history, lam=lam, window=window)
        sigma = sigma_h_from_variance(v, h)
        p_down, p_neutral, p_up = m0_category_probabilities(kappa, sigma)
        realized = log(ps[i+h] / ps[i])
        out.append({
            "model_id": MODEL_ID,
            "vol_state_id": VOL_STATE_ID,
            "last_info_date": dates[i-1],
            "target_start_date": dates[i],
            "target_end_date": dates[i+h],
            "horizon_sessions": h,
            "variance_1session": v,
            "sigma_h": sigma,
            "point_log_return": 0.0,
            "p_down": p_down,
            "p_neutral": p_neutral,
            "p_up": p_up,
            "realized_log_return": realized,
            "normalized_error": None if sigma == 0 else realized / sigma,
            "da_applicable": False,
            "safety_veto_applicable": False,
        })
    return out
