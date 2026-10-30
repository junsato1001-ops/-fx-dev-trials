"""Challenger forecast summary per FORECAST row (SPEC v0.3 §3.2; MODEL-PART1-V1 §2-§3).

- mu == 0 (DEFAULT_PARENT, or ESTIMATED with beta == 0, or F_MOM == 0): Point, PI10, PI90 and class
  probabilities are taken verbatim from the frozen S0 output strings. The challenger row is therefore
  identical to S0 at the serialized-value level (same f17 strings), not merely numerically close.
- mu != 0: shifted points x'_i = sigma * Z_i + mu on the S0 pool;
  Point = sigma * mean(Z) + mu with mean(Z) = fsum(Z)/N (as S0); PI = sigma * Z_(k10|k90) + mu
  (S0 ranks); classes counted on x' against b = KAPPA * sigma (EVAL-SPEC-V1 §3.3).
All values are returned as S0 f17 strings; `as_float` gives the parsed values.
"""
from __future__ import annotations

import math

import numpy as np

FIELDS = ("Point", "PI10", "PI90", "P_Down", "P_Neutral", "P_Up")
S0_FIELDS = {"Point": "S0_Point_LogReturn", "PI10": "PI10", "PI90": "PI90",
             "P_Down": "P_Down", "P_Neutral": "P_Neutral", "P_Up": "P_Up"}


def pool_z(rec, cand) -> np.ndarray:
    return np.sort(cand[:rec["pool_n"]], kind="stable")


def s0_summary(rec) -> dict:
    so = rec["s0_row"]
    return {k: so[S0_FIELDS[k]] for k in FIELDS}


def challenger_summary(rec, cand, s0mod) -> dict:
    if rec["class"] != "FORECAST":
        raise ValueError("summary only for FORECAST rows")
    if rec["mu"] == 0.0:
        out = s0_summary(rec)
        out["Source"] = "S0_VERBATIM"
        return out
    z = pool_z(rec, cand)
    n = z.size
    sig, mu = rec["sigma"], rec["mu"]
    k10, k90 = s0mod.quantile_ranks(n)
    mean_z = math.fsum(z.tolist()) / n
    x = sig * z + mu
    b = s0mod.KAPPA * sig
    left = int(np.searchsorted(x, -b, side="left"))    # x' < -b
    right = int(np.searchsorted(x, b, side="right"))   # x' <= +b
    f = s0mod.f17
    return {"Point": f(sig * mean_z + mu), "PI10": f(float(sig * z[k10 - 1] + mu)),
            "PI90": f(float(sig * z[k90 - 1] + mu)), "P_Down": f(left / n),
            "P_Neutral": f((right - left) / n), "P_Up": f((n - right) / n), "Source": "SHIFTED"}


def as_float(summary: dict) -> dict:
    return {k: float(summary[k]) for k in FIELDS}
