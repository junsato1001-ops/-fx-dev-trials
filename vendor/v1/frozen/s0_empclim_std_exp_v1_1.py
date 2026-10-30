#!/usr/bin/env python3
"""S0-EMPCLIM-STD-EXP-V1 reference implementation.

Frozen specification date: 2026-10-16 JST

Implementation v1.1 separates forecast issuance from outcome availability.
This module implements the S0 model layer only. It consumes a formal opportunity
ledger produced by the frozen ORIGIN_TARGET_MAP_V1 + VOL_STATE_V1 pipeline and
builds an expanding empirical distribution of standardized historical outcomes.
It does not modify the frozen M0 or mapping implementations.

Forecast distribution at origin t:
    z_i = y_i / sigma_i
    P(Y_t = sigma_t * z_i) = 1/N_t, i=1..N_t

Historical inclusion requires:
    Target_End_i <= Last_Available_Fix_t
    realized y_i finite
    historical sigma_i finite and > 0

Warm-up:
    N_overlap_equiv = N * min(1, nominal_origin_interval_days /
                                  nominal_horizon_calendar_days)
    require N_overlap_equiv >= 100.

Quantiles (1-indexed nearest-rank, integer arithmetic):
    k10 = (N + 9) // 10
    k90 = (9*N + 9) // 10

Classes use kappa=0.25 in standardized space:
    Down:    z < -kappa
    Neutral: -kappa <= z <= +kappa
    Up:      z > +kappa

The implementation is intentionally deterministic and standard-library only.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPEC_ID = "S0-EMPCLIM-STD-EXP-V1"
IMPLEMENTATION_VERSION = "1.1"
SPEC_FREEZE_DATE = "2026-10-16"
VOL_STATE_ID = "TRUNCATED_NORMALIZED_EWMA_252_L094"
ORIGIN_TARGET_MAP_ID = "ORIGIN_TARGET_MAP_V1"
KAPPA = 0.25
MIN_OVERLAP_EQUIV = 100

# Nominal schedule values. These are not realized holiday gaps.
HORIZON_RULES: Dict[str, Tuple[int, int]] = {
    "1D": (1, 1),   # origin interval, nominal horizon calendar days
    "1W": (7, 7),
    "4W": (7, 28),
}

REQUIRED_COLUMNS = {
    "Pair", "Origin", "Cutoff_JST", "Horizon", "Last_Available_Fix",
    "Target_Start", "Target_End", "H_Canonical_Intervals", "Sigma_H",
    "Realized_LogReturn", "Prereq_Status", "Outcome_Status",
}

OUTPUT_COLUMNS = [
    "Pair", "Origin", "Cutoff_JST", "Horizon", "Last_Available_Fix",
    "Target_Start", "Target_End", "H_Canonical_Intervals",
    "Sigma_H", "Pool_N", "N_Overlap_Equiv_Num", "N_Overlap_Equiv_Den",
    "Mean_Z", "Q10_Z", "Q90_Z", "S0_Point_LogReturn",
    "P_Down", "P_Neutral", "P_Up", "PI10", "PI90",
    "Realized_LogReturn", "Outcome_Status", "S0_Status", "Decision_Class", "Reason_Code",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_date(value: str) -> Optional[dt.date]:
    value = (value or "").strip()
    return dt.date.fromisoformat(value) if value else None


def finite_float(value: str | float | int | None) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def f17(x: Optional[float]) -> str:
    if x is None:
        return ""
    if not math.isfinite(x):
        raise ValueError("non-finite output")
    return format(x, ".17g")


def normalize_row_keys(row: dict) -> dict:
    return {str(k).lstrip("\ufeff"): v for k, v in row.items()}


def overlap_equiv_ratio(horizon: str) -> Tuple[int, int]:
    """Return reduced numerator/denominator for min(1, delta_origin/horizon)."""
    if horizon not in HORIZON_RULES:
        raise ValueError(f"unsupported horizon: {horizon}")
    origin_days, horizon_days = HORIZON_RULES[horizon]
    if origin_days <= 0 or horizon_days <= 0:
        raise ValueError("nominal days must be positive")
    if origin_days >= horizon_days:
        return 1, 1
    g = math.gcd(origin_days, horizon_days)
    return origin_days // g, horizon_days // g


def warmup_ready(n: int, horizon: str) -> bool:
    num, den = overlap_equiv_ratio(horizon)
    # n * num / den >= MIN_OVERLAP_EQUIV without floating point.
    return n * num >= MIN_OVERLAP_EQUIV * den


def quantile_ranks(n: int) -> Tuple[int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return (n + 9) // 10, (9 * n + 9) // 10


def nearest_rank_q10_q90(sorted_values: Sequence[float]) -> Tuple[float, float]:
    n = len(sorted_values)
    k10, k90 = quantile_ranks(n)
    return sorted_values[k10 - 1], sorted_values[k90 - 1]


def class_probabilities(sorted_z: Sequence[float], kappa: float = KAPPA) -> Tuple[float, float, float]:
    if kappa < 0 or not math.isfinite(kappa):
        raise ValueError("invalid kappa")
    n = len(sorted_z)
    if n == 0:
        raise ValueError("empty empirical distribution")
    # Neutral includes both boundaries exactly.
    left = bisect.bisect_left(sorted_z, -kappa)   # count z < -kappa
    right = bisect.bisect_right(sorted_z, kappa) # count z <= +kappa
    p_down = left / n
    p_neutral = (right - left) / n
    p_up = (n - right) / n
    return p_down, p_neutral, p_up


def empirical_summary(sorted_z: Sequence[float], sigma_t: float) -> dict:
    n = len(sorted_z)
    if n == 0:
        raise ValueError("empty empirical distribution")
    if not math.isfinite(sigma_t) or sigma_t < 0:
        raise ValueError("current sigma must be finite and >= 0")

    mean_z = math.fsum(sorted_z) / n
    q10_z, q90_z = nearest_rank_q10_q90(sorted_z)

    if sigma_t == 0.0:
        # Frozen edge case: current distribution degenerates at zero.
        return {
            "mean_z": mean_z,
            "q10_z": q10_z,
            "q90_z": q90_z,
            "point": 0.0,
            "p_down": 0.0,
            "p_neutral": 1.0,
            "p_up": 0.0,
            "pi10": 0.0,
            "pi90": 0.0,
        }

    p_down, p_neutral, p_up = class_probabilities(sorted_z)
    return {
        "mean_z": mean_z,
        "q10_z": q10_z,
        "q90_z": q90_z,
        "point": sigma_t * mean_z,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "p_up": p_up,
        "pi10": sigma_t * q10_z,
        "pi90": sigma_t * q90_z,
    }


@dataclass(frozen=True)
class Candidate:
    target_end: dt.date
    origin: dt.date
    z: float


def _group_rows(rows: Iterable[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for raw in rows:
        row = normalize_row_keys(raw)
        missing = REQUIRED_COLUMNS.difference(row.keys())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        key = (row["Pair"], row["Horizon"])
        groups.setdefault(key, []).append(row)
    for key in groups:
        groups[key].sort(key=lambda r: (parse_date(r["Origin"]), r["Pair"], r["Horizon"]))
    return groups


def _candidate_pool_source(group: Sequence[dict]) -> List[Candidate]:
    out: List[Candidate] = []
    for r in group:
        sigma_i = finite_float(r["Sigma_H"])
        y_i = finite_float(r["Realized_LogReturn"])
        target_end = parse_date(r["Target_End"])
        origin = parse_date(r["Origin"])
        if sigma_i is None or sigma_i <= 0.0:
            continue
        if y_i is None or target_end is None or origin is None:
            continue
        z = y_i / sigma_i
        if not math.isfinite(z):
            raise ValueError("non-finite standardized historical outcome")
        out.append(Candidate(target_end=target_end, origin=origin, z=z))
    out.sort(key=lambda c: (c.target_end, c.origin, c.z))
    return out


def _base_output(r: dict) -> dict:
    return {
        "Pair": r["Pair"],
        "Origin": r["Origin"],
        "Cutoff_JST": r["Cutoff_JST"],
        "Horizon": r["Horizon"],
        "Last_Available_Fix": r["Last_Available_Fix"],
        "Target_Start": r["Target_Start"],
        "Target_End": r["Target_End"],
        "H_Canonical_Intervals": r["H_Canonical_Intervals"],
        "Sigma_H": r["Sigma_H"],
        "Pool_N": "",
        "N_Overlap_Equiv_Num": "",
        "N_Overlap_Equiv_Den": "",
        "Mean_Z": "",
        "Q10_Z": "",
        "Q90_Z": "",
        "S0_Point_LogReturn": "",
        "P_Down": "",
        "P_Neutral": "",
        "P_Up": "",
        "PI10": "",
        "PI90": "",
        "Realized_LogReturn": r["Realized_LogReturn"],
        "Outcome_Status": r["Outcome_Status"],
        "S0_Status": "",
        "Decision_Class": "",
        "Reason_Code": "",
    }


def generate_s0(rows: Iterable[dict]) -> List[dict]:
    groups = _group_rows(rows)
    outputs: List[dict] = []

    for (pair, horizon), group in sorted(groups.items()):
        num, den = overlap_equiv_ratio(horizon)
        candidates = _candidate_pool_source(group)
        cursor = 0
        sorted_pool: List[float] = []

        for r in group:
            out = _base_output(r)
            origin_t = parse_date(r["Origin"])
            laf_t = parse_date(r["Last_Available_Fix"])

            # Mature every historical outcome that was knowable by current origin.
            while laf_t is not None and cursor < len(candidates) and candidates[cursor].target_end <= laf_t:
                c = candidates[cursor]
                # A target that is known by LastAvailableFix should originate strictly in the past.
                if origin_t is not None and c.origin >= origin_t:
                    raise AssertionError("historical pool leakage: candidate origin is not in the past")
                bisect.insort(sorted_pool, c.z)
                cursor += 1

            n = len(sorted_pool)
            out["Pool_N"] = str(n)
            out["N_Overlap_Equiv_Num"] = str(n * num)
            out["N_Overlap_Equiv_Den"] = str(den)

            source_status = (r["Prereq_Status"] or "").strip()
            sigma_t = finite_float(r["Sigma_H"])

            # Scope states are not model abstentions.
            if source_status == "PRE_LISTING":
                out["S0_Status"] = "PRE_LISTING"
                out["Decision_Class"] = "NOT_APPLICABLE"
                out["Reason_Code"] = "PRE_LISTING"
                outputs.append(out)
                continue
            if "PRE_SCOPE" in source_status:
                out["S0_Status"] = "PRE_SCOPE"
                out["Decision_Class"] = "NOT_APPLICABLE"
                out["Reason_Code"] = "PRE_SCOPE"
                outputs.append(out)
                continue

            if sigma_t is None:
                out["S0_Status"] = "S0_VOL_STATE_UNAVAILABLE"
                if source_status == "M0_VOL_WARMUP_INSUFFICIENT":
                    out["Decision_Class"] = "MODEL_PREREQ_UNAVAILABLE"
                    out["Reason_Code"] = "M0_VOL_WARMUP_INSUFFICIENT"
                else:
                    out["Decision_Class"] = "ABSTAIN_SYSTEM"
                    out["Reason_Code"] = "S0_VOL_STATE_INPUT_MISSING"
                outputs.append(out)
                continue
            if sigma_t < 0:
                raise ValueError(f"negative current sigma: {pair} {horizon} {r['Origin']}")

            if not warmup_ready(n, horizon):
                out["S0_Status"] = "S0_WARMUP_INSUFFICIENT"
                out["Decision_Class"] = "MODEL_PREREQ_UNAVAILABLE"
                out["Reason_Code"] = "S0_WARMUP_INSUFFICIENT"
                outputs.append(out)
                continue

            s = empirical_summary(sorted_pool, sigma_t)
            out.update({
                "Mean_Z": f17(s["mean_z"]),
                "Q10_Z": f17(s["q10_z"]),
                "Q90_Z": f17(s["q90_z"]),
                "S0_Point_LogReturn": f17(s["point"]),
                "P_Down": f17(s["p_down"]),
                "P_Neutral": f17(s["p_neutral"]),
                "P_Up": f17(s["p_up"]),
                "PI10": f17(s["pi10"]),
                "PI90": f17(s["pi90"]),
                "S0_Status": "FORECAST",
                "Decision_Class": "FORECAST",
                "Reason_Code": "",
            })
            outputs.append(out)

    # Stable audit order identical to pair/horizon grouping then origin.
    return outputs


def load_csv(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [normalize_row_keys(r) for r in csv.DictReader(f)]


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="formal opportunity ledger CSV")
    ap.add_argument("output_csv", help="S0 output CSV")
    args = ap.parse_args()

    rows = load_csv(args.input_csv)
    out = generate_s0(rows)
    write_csv(args.output_csv, out)
    print(f"{SPEC_ID}: wrote {len(out)} rows")
    print(f"input_sha256={sha256_file(args.input_csv)}")
    print(f"output_sha256={sha256_file(args.output_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
