"""DEV-USDJPY-SIMPLE-01 model layer (SPEC v0.3 §2-§7; Freeze-prep v0.1 §2-§3).

Pipeline (all point-in-time, USD/JPY 1D only):
1. allowlist rows (Date, JPY, USD) -> USD/JPY series and the full date list `obs`.
2. ledger rows by V1 ledger_builder.make_one() (exact replica of the frozen ledger's one()).
3. S0 v1.1 generate_s0() on those rows; pools captured as in V1 m1macro/s0link.py
   (pool at origin t = sorted first Pool_N candidates in S0 candidate order).
4. F_MOM(t) = log(P(v) / P(v20)), v = Last_Available_Fix, v20 = 20 TARGET publication days before v.
5. beta_t: prequential minimizer of sum_i CRPS(F_i^{S0}(. - beta F_i), y_i) / sigma_i over E_t;
   unique minimizer = weighted median; on an exact tie the point of the minimizing interval
   closest to 0 (minimum L2 norm, MODEL-PART2C-V1 §5). No intercept, no penalty, no sign constraint.
6. N_WARMUP = 250 distinct origins in E_t -> ESTIMATED (sticky); before that beta = 0 (DEFAULT_PARENT).
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np


# --------------------------------------------------------------------------- ledger

def build_ledger(lb, m0, mp, allow_rows, pair="USDJPY"):
    """Rows for `pair` 1D, in the frozen order. Uses only make_one(); input comes from the allowlist reader."""
    obs = [d for d, _, _ in allow_rows]
    vals = [(d, jpy / usd) for d, jpy, usd in allow_rows if jpy is not None and usd is not None]
    one = lb.make_one(m0, mp, pair, vals, obs[0])
    rows = [one(d, "1D") for d in obs]
    series = dict(vals)
    return rows, series


# --------------------------------------------------------------------------- S0 pools

def regenerate_s0(s0mod, string_rows):
    """Run frozen generate_s0 and capture each FORECAST pool (same method as V1 s0link.regenerate)."""
    captured = []
    orig = s0mod.empirical_summary

    def cap(sorted_z, sigma_t):
        captured.append(len(sorted_z))
        return orig(sorted_z, sigma_t)

    s0mod.empirical_summary = cap
    try:
        outputs = s0mod.generate_s0(string_rows)
    finally:
        s0mod.empirical_summary = orig
    groups = s0mod._group_rows(string_rows)
    if len(groups) != 1:
        raise RuntimeError("expected exactly one (pair, horizon) group")
    (key, group), = groups.items()
    cand = np.array([c.z for c in s0mod._candidate_pool_source(group)], dtype=np.float64)
    it = iter(captured)
    pool_n = {}
    for r in outputs:
        if r["Decision_Class"] != "FORECAST":
            continue
        n = next(it)
        if str(n) != r["Pool_N"]:
            raise RuntimeError("POOL_N_CAPTURE_MISMATCH")
        pool_n[r["Origin"]] = n
    if next(it, None) is not None:
        raise RuntimeError("UNMATCHED_POOL_CAPTURES")
    return outputs, cand, pool_n


# --------------------------------------------------------------------------- features

def f_mom(mp, series, v: dt.date, window: int):
    """log(P(v)/P(v20)); None if either fix is not in the USD/JPY series."""
    if v not in series:
        return None
    x = v
    for _ in range(window):
        x = mp.previous_target_day(x)
    if x not in series:
        return None
    return math.log(series[v] / series[x])


# --------------------------------------------------------------------------- CRPS

def crps_sorted(sorted_x: np.ndarray, y: float) -> float:
    """CRPS of the discrete empirical distribution (EVAL-SPEC-V1 §3.1, not the fair CRPS)."""
    n = sorted_x.size
    t1 = np.abs(sorted_x - y).sum() / n
    k = np.arange(1, n + 1, dtype=np.float64)
    t2 = ((2.0 * k - n - 1.0) * sorted_x).sum() / (n * n)  # = (1/(2n^2)) sum_jl |x_j - x_l|
    return float(t1 - t2)


# --------------------------------------------------------------------------- estimator

class _Fenwick:
    def __init__(self, m):
        self.m = m
        self.w = np.zeros(m + 1, dtype=np.float64)
        self.c = np.zeros(m + 1, dtype=np.int64)
        self.step = 1 << (m.bit_length() - 1) if m > 0 else 0

    def add(self, ranks0, weights):
        idx = ranks0.astype(np.int64) + 1
        w = weights.astype(np.float64)
        while idx.size:
            np.add.at(self.w, idx, w)
            np.add.at(self.c, idx, 1)
            idx = idx + (idx & -idx)
            keep = idx <= self.m
            idx = idx[keep]
            w = w[keep]

    def prefix_w(self, k):
        s = 0.0
        while k > 0:
            s += self.w[k]
            k -= k & -k
        return s

    def prefix_c(self, k):
        s = 0
        while k > 0:
            s += int(self.c[k])
            k -= k & -k
        return s

    def lower_bound_w(self, target):
        """Smallest k (1-based) with prefix_w(k) >= target."""
        pos, rem, step = 0, target, self.step
        while step:
            nxt = pos + step
            if nxt <= self.m and self.w[nxt] < rem:
                pos = nxt
                rem -= self.w[nxt]
            step >>= 1
        return pos + 1

    def lower_bound_c(self, target):
        pos, rem, step = 0, target, self.step
        while step:
            nxt = pos + step
            if nxt <= self.m and self.c[nxt] < rem:
                pos = nxt
                rem -= int(self.c[nxt])
            step >>= 1
        return pos + 1


def weighted_median_min_norm(values_sorted, fen: _Fenwick, total_w: float) -> float:
    """argmin_b sum w_k |b - v_k| over the inserted points; exact-tie interval -> point closest to 0."""
    if total_w <= 0.0:
        return 0.0
    half = total_w / 2.0
    k = fen.lower_bound_w(half)
    lo = float(values_sorted[k - 1])
    if fen.prefix_w(k) != half:
        return lo
    # Exact tie: minimizers are [v_k, v_next] where v_next is the next inserted point with weight.
    c = fen.prefix_c(k)
    k2 = fen.lower_bound_c(c + 1)
    while k2 <= fen.m and fen.prefix_w(k2) == fen.prefix_w(k2 - 1):
        c += 1
        k2 = fen.lower_bound_c(c + 1)
    if k2 > fen.m:
        return lo
    hi = float(values_sorted[k2 - 1])
    return min(max(0.0, lo), hi)


def run_model(cfg, m0, mp, s0mod, lb, allow_rows, return_pool_source=False):
    """Return per-origin records (list of dicts) for USD/JPY 1D. Values stay in memory; callers decide output.

    With return_pool_source=True, return (records, cand) where cand is the S0 candidate z array
    (pool at origin t = sorted cand[:Pool_N_t])."""
    rows, series = build_ledger(lb, m0, mp, allow_rows)
    srows = lb.to_strings(rows)
    s0_out, cand, pool_n = regenerate_s0(s0mod, srows)
    if len(s0_out) != len(srows):
        raise RuntimeError("S0 output length mismatch")
    window = cfg["f_mom_window_target_days"]
    n_warm = cfg["n_warmup"]
    cutoff = dt.date.fromisoformat(cfg["development_target_end_max"])

    recs = []
    for r, so in zip(srows, s0_out):
        if r["Origin"] != so["Origin"]:
            raise RuntimeError("row order mismatch")
        origin = dt.date.fromisoformat(r["Origin"])
        laf = dt.date.fromisoformat(r["Last_Available_Fix"]) if r["Last_Available_Fix"] else None
        tend = dt.date.fromisoformat(r["Target_End"]) if r["Target_End"] else None
        sig = float(r["Sigma_H"]) if r["Sigma_H"] else None
        y = float(r["Realized_LogReturn"]) if r["Realized_LogReturn"] else None
        fm = f_mom(mp, series, laf, window) if laf is not None else None
        recs.append({"origin": origin, "laf": laf, "target_end": tend, "sigma": sig, "y": y,
                     "outcome": r["Outcome_Status"], "prereq": r["Prereq_Status"],
                     "s0_class": so["Decision_Class"], "s0_reason": so["Reason_Code"],
                     "pool_n": pool_n.get(r["Origin"]), "f_mom": fm, "s0_row": so})

    # Development scope: Target_End <= cutoff (rows without Target_End are scope/pre-listing rows).
    for rec in recs:
        rec["in_scope"] = rec["target_end"] is not None and rec["target_end"] <= cutoff

    # Training points per eligible origin i (S0 FORECAST, F defined, sigma>0, y observed).
    elig = [i for i, rec in enumerate(recs)
            if rec["in_scope"] and rec["s0_class"] == "FORECAST" and rec["f_mom"] is not None
            and rec["sigma"] is not None and rec["sigma"] > 0 and rec["y"] is not None
            and rec["outcome"] == "OBSERVED"]
    pts_v, pts_w, pts_owner = [], [], []
    for i in elig:
        rec = recs[i]
        n = rec["pool_n"]
        z = np.sort(cand[:n], kind="stable")
        F = rec["f_mom"]
        if F == 0.0:
            continue  # contributes no beta-dependent loss
        r_ = (rec["y"] - rec["sigma"] * z) / F
        pts_v.append(r_)
        pts_w.append(np.full(n, abs(F) / (n * rec["sigma"]), dtype=np.float64))
        pts_owner.append(np.full(n, i, dtype=np.int64))
    if pts_v:
        V = np.concatenate(pts_v); W = np.concatenate(pts_w); O = np.concatenate(pts_owner)
    else:
        V = np.zeros(0); W = np.zeros(0); O = np.zeros(0, dtype=np.int64)
    order = np.argsort(V, kind="stable")
    V_sorted = V[order]
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    fen = _Fenwick(int(V.size))
    owner_slices = {}
    if O.size:
        bounds = np.flatnonzero(np.diff(O)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [O.size]])
        for s, e in zip(starts, ends):
            owner_slices[int(O[s])] = (int(s), int(e))

    # Prequential: candidates mature when Target_End_i <= LAF_t (sorted by Target_End then origin).
    elig_sorted = sorted(elig, key=lambda i: (recs[i]["target_end"], recs[i]["origin"]))
    cursor = 0
    n_E = 0
    total_w = 0.0
    estimated = False
    for rec in recs:
        rec["n_E"] = None
        rec["param_state"] = None
        rec["beta"] = None
        if not rec["in_scope"] or rec["laf"] is None:
            continue
        while cursor < len(elig_sorted) and recs[elig_sorted[cursor]]["target_end"] <= rec["laf"]:
            i = elig_sorted[cursor]
            if recs[i]["origin"] >= rec["origin"]:
                raise AssertionError("training leakage: candidate origin not in the past")
            n_E += 1
            if i in owner_slices:
                s, e = owner_slices[i]
                fen.add(rank[s:e], W[s:e])
                total_w += float(W[s:e].sum())
            cursor += 1
        if n_E >= n_warm:
            estimated = True
        rec["n_E"] = n_E
        rec["param_state"] = "ESTIMATED" if estimated else "DEFAULT_PARENT"
        rec["beta"] = weighted_median_min_norm(V_sorted, fen, total_w) if estimated else 0.0

    # Decision classes and losses.
    for rec in recs:
        rec["class"] = None; rec["reason"] = ""; rec["mu"] = None
        rec["loss_ch"] = None; rec["loss_s0"] = None; rec["metric"] = None
        if not rec["in_scope"]:
            continue
        if rec["s0_class"] == "NOT_APPLICABLE":
            rec["class"] = "NOT_APPLICABLE"; rec["reason"] = rec["s0_reason"]
            continue
        if rec["s0_class"] != "FORECAST":
            rec["class"] = "MODEL_PREREQ_UNAVAILABLE"; rec["reason"] = "PARENT_PREREQ_UNAVAILABLE"
            continue
        if rec["f_mom"] is None:
            rec["class"] = "MODEL_PREREQ_UNAVAILABLE"; rec["reason"] = "FEATURE_PREREQ_UNAVAILABLE"
            continue
        rec["class"] = "FORECAST"
        rec["mu"] = rec["beta"] * rec["f_mom"]
        if rec["sigma"] is None or not (rec["sigma"] > 0):
            rec["metric"] = "METRIC_UNDEFINED_ZERO_SCALE"
            continue
        if rec["outcome"] != "OBSERVED" or rec["y"] is None:
            continue
        z = np.sort(cand[:rec["pool_n"]], kind="stable")
        x = rec["sigma"] * z
        rec["loss_s0"] = crps_sorted(x, rec["y"]) / rec["sigma"]
        rec["loss_ch"] = crps_sorted(x + rec["mu"], rec["y"]) / rec["sigma"]
        rec["metric"] = "ELIGIBLE"
    if return_pool_source:
        return recs, cand
    return recs
