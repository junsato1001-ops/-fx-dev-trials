"""Report-only diagnostics required by SPEC v0.3 §10 ("Gate には使わず、必ず報告するもの").

Nothing here feeds the Gate. Definitions inherited from EVAL-SPEC-V1:
- Coverage (§4.2): S = scheduled in-scope opportunities minus NOT_APPLICABLE;
  Prerequisite readiness = (S - MODEL_PREREQ_UNAVAILABLE) / S;
  Decision coverage = FORECAST / (FORECAST + ABSTAIN_MODEL);
  Operational completion = 1 - ABSTAIN_SYSTEM / (S - MODEL_PREREQ_UNAVAILABLE).
- Realized class (§3.3): b = 0.25 * sigma_M0; Down y < -b; Neutral -b <= y <= b; Up y > b.
- RPS (§3.2): sum_{k=1}^{2} (P_cum_k - O_cum_k)^2, order Down, Neutral, Up; not normalized.
- Multiclass Brier (§3.2): sum_c (p_c - o_c)^2.
- IS / sigma_M0 (§3.2, alpha = 0.2, l = PI10, u = PI90); only on sigma_M0 > 0 (§4.1a).
- PI80 coverage (§3.2): 1(l <= y <= u).
- Deterministic Verification-Rank Calibration (§3.4): L = #{x_i < y}, E = #{x_i = y};
  mass 1 spread uniformly over [L/(N+1), (L+E+1)/(N+1)] into bins [0,.1), ..., [.9,1.0].
- RPS / Brier / PI80 coverage / VRC use the whole common sample (FORECAST and OBSERVED, incl. sigma = 0);
  CRPS/sigma and IS/sigma use the Primary Metric Eligible subset (sigma > 0).
Direction diagnostics (descriptive; definitions fixed in this implementation, flagged for audit):
- modal-class hit: argmax of (P_Down, P_Neutral, P_Up) equals the realized class; rows whose
  maximum is tied are counted separately and excluded from the rate.
- point-sign hit: sign(Point) == sign(y), over rows with Point != 0 and y != 0 (others counted).
- mean probability assigned to the realized class.
Each sample is reported for the Gate sample (ESTIMATED) and for the full common sample.
"""
from __future__ import annotations

import numpy as np

from devtrial import summary

CLASSES = ("Down", "Neutral", "Up")
KAPPA = 0.25


def realized_class(y: float, sigma: float) -> int:
    b = KAPPA * sigma
    if y < -b:
        return 0
    if y > b:
        return 2
    return 1


def rps(p, o) -> float:
    return (p[0] - o[0]) ** 2 + ((p[0] + p[1]) - (o[0] + o[1])) ** 2


def brier(p, o) -> float:
    return sum((p[c] - o[c]) ** 2 for c in range(3))


def interval_score(l: float, u: float, y: float, alpha: float = 0.2) -> float:
    return (u - l) + (2 / alpha) * (l - y) * (y < l) + (2 / alpha) * (y - u) * (y > u)


def vrc_bins(sorted_x: np.ndarray, y: float) -> np.ndarray:
    n = sorted_x.size
    L = int(np.searchsorted(sorted_x, y, side="left"))
    E = int(np.searchsorted(sorted_x, y, side="right")) - L
    lo, hi = L / (n + 1), (L + E + 1) / (n + 1)
    edges = np.linspace(0.0, 1.0, 11)
    out = np.zeros(10)
    width = hi - lo
    for k in range(10):
        a, b = max(lo, edges[k]), min(hi, edges[k + 1])
        if b > a:
            out[k] = (b - a) / width
    return out


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def _model_block(rows, key, cand):
    """rows: list of (rec, ch_summary). key: 'ch' or 's0'."""
    rps_v, br_v, cov_v, is_v, crps_v = [], [], [], [], []
    vrc = np.zeros(10)
    modal_hit, modal_n, modal_tie = 0, 0, 0
    sign_hit, sign_n, sign_excl = 0, 0, 0
    p_real = []
    for rec, chs in rows:
        s = summary.as_float(chs if key == "ch" else summary.s0_summary(rec))
        p = (s["P_Down"], s["P_Neutral"], s["P_Up"])
        y, sig = rec["y"], rec["sigma"]
        rc = realized_class(y, sig)
        o = tuple(1.0 if c == rc else 0.0 for c in range(3))
        rps_v.append(rps(p, o)); br_v.append(brier(p, o))
        cov_v.append(1.0 if s["PI10"] <= y <= s["PI90"] else 0.0)
        if sig > 0:
            is_v.append(interval_score(s["PI10"], s["PI90"], y) / sig)
            crps_v.append(rec["loss_ch"] if key == "ch" else rec["loss_s0"])
        x = sig * summary.pool_z(rec, cand)
        if key == "ch" and rec["mu"] != 0.0:
            x = x + rec["mu"]
        vrc += vrc_bins(x, y)
        m = max(p)
        if sum(1 for q in p if q == m) > 1:
            modal_tie += 1
        else:
            modal_n += 1
            modal_hit += int(p.index(m) == rc)
        if s["Point"] == 0.0 or y == 0.0:
            sign_excl += 1
        else:
            sign_n += 1
            sign_hit += int((s["Point"] > 0) == (y > 0))
        p_real.append(p[rc])
    n = len(rows)
    return {
        "N_common": n, "N_primary_eligible": len(crps_v),
        "mean_CRPS_over_sigma": _mean(crps_v), "mean_RPS": _mean(rps_v), "mean_Brier": _mean(br_v),
        "mean_IS_over_sigma": _mean(is_v), "PI80_coverage": _mean(cov_v),
        "VRC_bin_mass": vrc.tolist(), "VRC_bin_freq": (vrc / n).tolist() if n else None,
        "direction": {
            "modal_class_hit_rate": modal_hit / modal_n if modal_n else None,
            "modal_class_N": modal_n, "modal_class_tied_N": modal_tie,
            "point_sign_hit_rate": sign_hit / sign_n if sign_n else None,
            "point_sign_N": sign_n, "point_sign_excluded_N": sign_excl,
            "mean_prob_realized_class": _mean(p_real),
        },
    }


def _coverage(scope, cls_key):
    c = {}
    for r in scope:
        c[r[cls_key]] = c.get(r[cls_key], 0) + 1
    S = len(scope) - c.get("NOT_APPLICABLE", 0)
    mpu = c.get("MODEL_PREREQ_UNAVAILABLE", 0)
    fc, am, asys = c.get("FORECAST", 0), c.get("ABSTAIN_MODEL", 0), c.get("ABSTAIN_SYSTEM", 0)
    return {"counts": dict(sorted(c.items())), "S": S,
            "prerequisite_readiness": (S - mpu) / S if S else None,
            "decision_coverage": fc / (fc + am) if (fc + am) else None,
            "operational_completion": 1 - asys / (S - mpu) if (S - mpu) else None}


def _beta_block(scope):
    est = [r for r in scope if r["param_state"] == "ESTIMATED"]
    b = np.array([r["beta"] for r in est], dtype=np.float64)
    if not b.size:
        return {"N_ESTIMATED_rows": 0}
    signs = np.sign(b)
    nz = signs[signs != 0]
    return {"N_ESTIMATED_rows": int(b.size), "first_origin": est[0]["origin"].isoformat(),
            "last_origin": est[-1]["origin"].isoformat(), "first": float(b[0]), "last": float(b[-1]),
            "min": float(b.min()), "median": float(np.median(b)), "max": float(b.max()),
            "N_positive": int((b > 0).sum()), "N_negative": int((b < 0).sum()), "N_zero": int((b == 0).sum()),
            "N_sign_changes": int((nz[1:] != nz[:-1]).sum()) if nz.size > 1 else 0,
            "note": "per-origin beta_t is in ROWS.jsonl (Beta)"}


def diagnostics(recs, cand, s0mod) -> dict:
    scope = [r for r in recs if r["in_scope"]]
    common = [r for r in scope if r["class"] == "FORECAST" and r["outcome"] == "OBSERVED" and r["y"] is not None]
    rows = [(r, summary.challenger_summary(r, cand, s0mod)) for r in common]
    gate_rows = [(r, s) for r, s in rows if r["param_state"] == "ESTIMATED"]
    fc_scope = [r for r in scope if r["class"] == "FORECAST"]

    def sample(rs):
        return {"challenger": _model_block(rs, "ch", cand), "s0": _model_block(rs, "s0", cand)}

    return {
        "note": "Report only (SPEC v0.3 §10). None of these values enters the Gate.",
        "coverage": {"challenger": _coverage(scope, "class"), "s0": _coverage(scope, "s0_class"),
                     "note": "No ABSTAIN_MODEL rule exists for either model; not independent safety evidence."},
        "counts": {
            "DEFAULT_PARENT_forecast_rows": sum(1 for r in fc_scope if r["param_state"] == "DEFAULT_PARENT"),
            "ESTIMATED_forecast_rows": sum(1 for r in fc_scope if r["param_state"] == "ESTIMATED"),
            "METRIC_UNDEFINED_ZERO_SCALE": sum(1 for r in fc_scope if r["metric"] == "METRIC_UNDEFINED_ZERO_SCALE"),
            "METRIC_UNDEFINED_ZERO_SCALE_in_gate_sample_scope": sum(
                1 for r in fc_scope if r["metric"] == "METRIC_UNDEFINED_ZERO_SCALE" and r["param_state"] == "ESTIMATED"),
            "mu_zero_forecast_rows": sum(1 for r in fc_scope if r["mu"] == 0.0),
        },
        "beta_trajectory": _beta_block(scope),
        "secondary_calibration_direction": {
            "gate_sample_ESTIMATED": sample(gate_rows),
            "full_common_sample": sample(rows),
        },
    }
