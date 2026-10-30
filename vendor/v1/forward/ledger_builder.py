"""Formal forecast-ready ledger builder for forward operation.

This module reproduces the logic of frozen/formal_forecast_ledger_v1.py
(SHA-256 b6ffcf34...) as reusable functions, so that the same ledger can be
built from any ECB snapshot and extended with forward origins that lie after
the last published fix.

Equivalence is enforced by tests/test_reproduce_frozen.py: run on the frozen
snapshot (eurofxref-hist.zip, SHA-256 f36e0aff...), build_rows() + write_csv()
must reproduce formal_forecast_ledger_v1.csv byte for byte (SHA-256 e4edb850...).

Frozen modules (M0 v1.2, ORIGIN_TARGET_MAP_V1, S0 v1.1) are loaded only after
their SHA-256 has been verified, and are never modified.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import io
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "frozen"

FROZEN_SHA256 = {
    "m0_walkforward_v1_2.py": "4ca2f37ab729d08f24f008d75c5ceb95f11d79d5bf9575941f44ffd32a77e3d5",
    "origin_target_map_v1.py": "902f0b43a84a0ec309a6e99b6617ca2816e240054bd77e9dbf7d3b2812f3c988",
    "s0_empclim_std_exp_v1_1.py": "84c317661e6798b906063a50fc416c9d594708ffcfbb6010a5bd83f636487676",
    "formal_forecast_ledger_v1.py": "b6ffcf34fe6fc605c5a7d7f4953831778bdb9ff45effdf4f3d04535886b16d77",
}

# Scope of the formal forward ledger. RFX-HOLDOUT-A-V1 pairs (GBPJPY, AUDJPY,
# CADJPY, SGDJPY, KRWJPY) are sealed and MUST NOT be computed here.
PAIRS = {"USDJPY": ("JPY", "USD"), "CHFJPY": ("JPY", "CHF"), "INRJPY": ("JPY", "INR")}
SEALED_CURRENCIES = {"GBP", "AUD", "CAD", "SGD", "KRW"}

FIELDS = [
    "Pair", "Origin", "Cutoff_JST", "Horizon", "Last_Available_Fix", "Target_Start", "Ideal_Target_End", "Target_End",
    "H_Canonical_Intervals", "EWMA_Var_1", "Sigma_H", "M0_Point_LogReturn", "M0_P_Down", "M0_P_Neutral", "M0_P_Up",
    "M0_PI10", "M0_PI90", "Realized_LogReturn", "Normalized_Error", "Abs_Error", "Prereq_Status", "Outcome_Status",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def load_frozen():
    """Verify SHA-256 of every frozen module, then load M0, MAP and S0."""
    for name, expected in FROZEN_SHA256.items():
        got = sha256_file(FROZEN / name)
        if got != expected:
            raise RuntimeError(f"FROZEN_SHA_MISMATCH: {name} {got} != {expected}")

    def load(modname, fname):
        spec = importlib.util.spec_from_file_location(modname, FROZEN / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    m0 = load("frozen_m0_v1_2", "m0_walkforward_v1_2.py")
    mp = load("frozen_origin_target_map_v1", "origin_target_map_v1.py")
    s0 = load("frozen_s0_v1_1", "s0_empclim_std_exp_v1_1.py")
    return m0, mp, s0


def read_ecb_zip(zip_bytes: bytes):
    """Return (csv_bytes, raw_rows_sorted). Same parsing as the frozen ledger."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        names = z.namelist()
        if names != ["eurofxref-hist.csv"]:
            raise RuntimeError(f"Unexpected ZIP members: {names}")
        csv_bytes = z.read("eurofxref-hist.csv")
    text = csv_bytes.decode("utf-8-sig")
    rdr = csv.DictReader(io.StringIO(text))
    cols = [c for c in (rdr.fieldnames or []) if c]

    def num(x):
        x = (x or "").strip()
        return None if not x or x == "N/A" else float(x)

    raw = []
    for r in rdr:
        d = dt.date.fromisoformat(r["Date"])
        # Only the columns needed for in-scope pairs are parsed into numbers.
        raw.append({"Date": d, **{c: num(r.get(c, "")) for c in ("JPY", "USD", "CHF", "INR")}})
    raw.sort(key=lambda r: r["Date"])
    if len({r["Date"] for r in raw}) != len(raw):
        raise RuntimeError("Duplicate dates in ECB file")
    return csv_bytes, raw


def pair_series(raw):
    pdata = {p: [] for p in PAIRS}
    for r in raw:
        for p, (a, b) in PAIRS.items():
            if r[a] is not None and r[b] is not None:
                pdata[p].append((r["Date"], r[a] / r[b]))
    return pdata


def make_one(m0, mp, pair, vals, obs0):
    """Exact replica of one() in frozen/formal_forecast_ledger_v1.py."""
    dates = [d for d, _ in vals]
    rates = [v for _, v in vals]
    pos = {d: i for i, d in enumerate(dates)}

    def one(origin, horizon):
        if mp.next_target_day_on_or_after(origin) < dates[0]:
            return {"Pair": pair, "Origin": origin, "Horizon": horizon, "Prereq_Status": "PRE_LISTING", "Outcome_Status": "NOT_APPLICABLE"}
        if origin == obs0 and horizon == "1D":
            return {"Pair": pair, "Origin": origin, "Horizon": horizon, "Prereq_Status": "PRE_SCOPE", "Outcome_Status": "NOT_APPLICABLE"}
        z = mp.map_origin(origin, horizon)
        base = {"Pair": pair, "Origin": origin, "Cutoff_JST": "08:30", "Horizon": horizon,
                "Last_Available_Fix": z.last_available_fix, "Target_Start": z.target_start,
                "Ideal_Target_End": z.ideal_target_end, "Target_End": z.target_end,
                "H_Canonical_Intervals": z.h_canonical_intervals}
        if z.last_available_fix not in pos:
            if z.last_available_fix < dates[0]:
                return {**base, "Prereq_Status": "PRE_LISTING", "Outcome_Status": "NOT_APPLICABLE"}
            return {**base, "Prereq_Status": "PAIR_COMPONENT_GAP", "Outcome_Status": "UNAVAILABLE"}
        j = pos[z.last_available_fix]
        realized = None
        outcome = "PENDING"
        if z.target_start in pos and z.target_end in pos:
            s = pos[z.target_start]
            e = pos[z.target_end]
            if s != j + 1:
                raise RuntimeError("ORIGIN_START_ADJACENCY_FAIL")
            realized = math.log(rates[e] / rates[s])
            outcome = "OBSERVED"
        if j < 252:
            return {**base, "Realized_LogReturn": realized, "Prereq_Status": "M0_VOL_WARMUP_INSUFFICIENT", "Outcome_Status": outcome}
        hr = rates[j - 252:j + 1]
        rets = [math.log(hr[k + 1] / hr[k]) for k in range(252)]
        v = m0.finite_ewma_variance(rets)
        sig = m0.sigma_h_from_variance(v, z.h_canonical_intervals)
        pd, pn, pu = m0.m0_category_probabilities(0.25, sig)
        lo, hi = m0.prediction_interval(sig, coverage=0.80)
        return {**base, "EWMA_Var_1": v, "Sigma_H": sig, "M0_Point_LogReturn": 0.0,
                "M0_P_Down": pd, "M0_P_Neutral": pn, "M0_P_Up": pu, "M0_PI10": lo, "M0_PI90": hi,
                "Realized_LogReturn": realized,
                "Normalized_Error": None if realized is None or sig == 0 else realized / sig,
                "Abs_Error": None if realized is None else abs(realized),
                "Prereq_Status": "FORECAST_READY", "Outcome_Status": outcome}

    return one


def build_rows(m0, mp, raw, extra_origins=()):
    """Historical ledger in the frozen order, plus extra (forward) origins.

    extra_origins: iterable of (pair, origin_date, horizon) appended after the
    historical rows of their pair, in the given order.
    """
    obs = [r["Date"] for r in raw]
    pdata = pair_series(raw)
    rows = []
    for pair, vals in pdata.items():
        one = make_one(m0, mp, pair, vals, obs[0])
        for d in obs:
            rows.append(one(d, "1D"))
        d = obs[0]
        while d.weekday() != 6:
            d += dt.timedelta(days=1)
        while d <= obs[-1]:
            rows.append(one(d, "1W"))
            rows.append(one(d, "4W"))
            d += dt.timedelta(days=7)
        for (p, o, h) in extra_origins:
            if p == pair:
                rows.append(one(o, h))
    return rows


def fmt(v):
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, float):
        return format(v, ".17g")
    if v is None:
        return ""
    return v


def to_strings(rows):
    return [{k: fmt(r.get(k, "")) for k in FIELDS} for r in rows]


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in to_strings(rows):
            w.writerow(r)
