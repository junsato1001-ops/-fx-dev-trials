"""DEV-USDJPY-SIMPLE-01 entry point.

Modes:
  selftest   offline tests on synthetic data only (no snapshot is read)
  preflight  dates / counts / states only (Freeze-prep v0.1 §5.2) -> requires AUTHORIZATION.json allowing "preflight"
  run        full Development execution and Gate -> requires AUTHORIZATION.json allowing "run"

Neither preflight nor run is authorized at the time of writing. Without the authorization file both
modes stop before reading the snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from devtrial import frozen, model, reader, stats  # noqa: E402

TRIAL_DIR = ROOT / "trials" / "DEV-USDJPY-SIMPLE-01"


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_authorization(cfg: dict, mode: str) -> dict:
    p = ROOT / cfg["authorization_file"]
    if not p.is_file():
        raise SystemExit(f"NOT AUTHORIZED: {cfg['authorization_file']} does not exist; mode '{mode}' refused")
    auth = json.loads(p.read_text(encoding="utf-8"))
    if auth.get("trial_id") != cfg["trial_id"] or mode not in auth.get("modes", []):
        raise SystemExit(f"NOT AUTHORIZED: mode '{mode}' not listed in {cfg['authorization_file']}")
    for k in ("freeze_commit", "authorized_by", "authorized_at_jst"):
        if not auth.get(k):
            raise SystemExit(f"NOT AUTHORIZED: missing {k}")
    return auth


def load_inputs(cfg: dict):
    snap = ROOT / cfg["snapshot"]["path"]
    if frozen.sha256_file(snap) != cfg["snapshot"]["sha256"]:
        raise SystemExit("SNAPSHOT_SHA_MISMATCH")
    allow = reader.read_snapshot_zip(str(snap), cfg["snapshot"]["member"])
    m0, mp, s0, lb = frozen.load_modules(cfg)
    return allow, m0, mp, s0, lb


def _gate_sample(recs):
    return [r for r in recs if r["class"] == "FORECAST" and r["metric"] == "ELIGIBLE"
            and r["param_state"] == "ESTIMATED"]


def preflight(cfg: dict) -> dict:
    allow, m0, mp, s0, lb = load_inputs(cfg)
    return preflight_summary(cfg, model.run_model(cfg, m0, mp, s0, lb, allow))


def preflight_summary(cfg: dict, recs) -> dict:
    """Dates, counts and states only. No price, sigma, pool, F_MOM, beta, loss, d, D or p-value."""
    scope = [r for r in recs if r["in_scope"]]

    def count(key):
        out = {}
        for r in scope:
            k = r[key] if r[key] is not None else ""
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    fc = [r for r in scope if r["class"] == "FORECAST"]
    est = [r for r in scope if r["param_state"] == "ESTIMATED"]
    gate = _gate_sample(recs)
    return {
        "schema_version": "DEV_PREFLIGHT_V1",
        "trial_id": cfg["trial_id"],
        "first_evaluation_origin": fc[0]["origin"].isoformat() if fc else None,
        "first_estimated_origin": est[0]["origin"].isoformat() if est else None,
        "last_in_scope_origin": scope[-1]["origin"].isoformat() if scope else None,
        "counts": {
            "in_scope_rows": len(scope),
            "prereq_status": count("prereq"),
            "outcome_status": count("outcome"),
            "s0_decision_class": count("s0_class"),
            "challenger_decision_class": count("class"),
            "challenger_reason": count("reason"),
            "param_state": count("param_state"),
            "metric_status": count("metric"),
            "N_EVAL": len(gate),
        },
        "n_eval_min": cfg["n_eval_min"],
        "evaluable": len(gate) >= cfg["n_eval_min"],
    }


def gate(cfg: dict, recs) -> dict:
    g = _gate_sample(recs)
    full = [r for r in recs if r["class"] == "FORECAST" and r["metric"] == "ELIGIBLE"]
    out = {"schema_version": "DEV_GATE_V1", "trial_id": cfg["trial_id"], "N_EVAL": len(g),
           "n_eval_min": cfg["n_eval_min"]}
    if len(g) < cfg["n_eval_min"]:
        out["result_class"] = "NOT_EVALUABLE"
        return out
    d = np.array([r["loss_ch"] - r["loss_s0"] for r in g], dtype=np.float64)
    bs = cfg["bootstrap"]
    prim = stats.bootstrap_test(d, bs["primary_mean_block_length"], bs["replicates"], bs["seed"])
    sens12 = stats.bootstrap_test(d, bs["sensitivity_mean_block_length"], bs["replicates"], bs["seed"])
    hac = stats.hac_test(d, cfg["hac_lag"])
    a = cfg["alpha_dev"]
    v_p = stats.verdict(prim["p_sup"], prim["p_inf"], a)
    v_12 = stats.verdict(sens12["p_sup"], sens12["p_inf"], a)
    v_h = stats.verdict(hac["p_sup"], hac["p_inf"], a)
    n1, n2 = stats.halves(len(g))
    D1 = float(d[:n1].mean()) if n1 else float("nan")
    D2 = float(d[n1:].mean())
    d_full = np.array([r["loss_ch"] - r["loss_s0"] for r in full], dtype=np.float64)
    cond_p = prim["p_sup"] < a
    cond_recent = D2 < 0
    out.update({
        "p_dev": prim["p_sup"], "primary": prim, "sensitivity_block12": sens12, "hac": hac,
        "BLOCK_SENSITIVITY_DISAGREEMENT": stats.disagreement(v_p, v_12),
        "SENSITIVITY_DISAGREEMENT": stats.disagreement(v_p, v_h),
        "halves": {"n_first": n1, "n_second": n2, "D_first": D1, "D_second": D2},
        "full_common_sample": {"N": int(d_full.size), "D": float(d_full.mean()) if d_full.size else None},
        "conditions": {"N_EVAL_ge_min": True, "p_dev_lt_alpha": cond_p, "second_half_D_lt_0": cond_recent},
        "result_class": "FORWARD_CANDIDATE_ELIGIBLE" if (cond_p and cond_recent) else "NOT_ELIGIBLE",
        "note": "p_dev is a selection score, not a confirmatory significance test (SPEC v0.3 §8).",
    })
    return out


def challenger_row(r, s0mod, cand) -> dict:
    """Prediction row fields (SPEC v0.3 §13). For FORECAST rows only."""
    z = np.sort(cand[:r["pool_n"]], kind="stable")
    n = z.size
    k10, k90 = s0mod.quantile_ranks(n)
    sig, mu = r["sigma"], r["mu"]
    x = sig * z + mu
    b = 0.25 * sig
    left = int(np.searchsorted(x, -b, side="left"))
    right = int(np.searchsorted(x, b, side="right"))
    return {"Origin": r["origin"].isoformat(), "Decision_Class": r["class"], "Parameter_State": r["param_state"],
            "Beta": r["beta"], "F_MOM": r["f_mom"], "Mu": mu, "Sigma_M0": sig, "Pool_N": n,
            "Point": sig * float(z.sum()) / n + mu, "PI10": float(x[k10 - 1]), "PI90": float(x[k90 - 1]),
            "P_Down": left / n, "P_Neutral": (right - left) / n, "P_Up": (n - right) / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "preflight", "run"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = frozen.load_config()
    if args.mode == "selftest":
        import subprocess
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                               cwd=str(ROOT))
    auth = check_authorization(cfg, args.mode)
    out_dir = Path(args.out or TRIAL_DIR / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        res = preflight(cfg)
        res["authorization"] = {k: auth[k] for k in ("freeze_commit", "authorized_by", "authorized_at_jst")}
        p = out_dir / "DEV-USDJPY-SIMPLE-01-PREFLIGHT.json"
        if p.exists():
            raise SystemExit("preflight artifact already exists; not overwritten")
        p.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(p, _sha_bytes(p.read_bytes()))
        return 0
    # mode == "run"
    allow, m0, mp, s0, lb = load_inputs(cfg)
    recs, cand = model.run_model(cfg, m0, mp, s0, lb, allow, return_pool_source=True)
    res = gate(cfg, recs)
    res["authorization"] = {k: auth[k] for k in ("freeze_commit", "authorized_by", "authorized_at_jst")}
    gp = out_dir / "DEV-USDJPY-SIMPLE-01-GATE.json"
    rp = out_dir / "DEV-USDJPY-SIMPLE-01-ROWS.jsonl"
    if gp.exists() or rp.exists():
        raise SystemExit("run artifacts already exist; not overwritten")
    with rp.open("w", encoding="utf-8") as f:
        for r in recs:
            if not r["in_scope"]:
                continue
            row = {"Origin": r["origin"].isoformat(), "Decision_Class": r["class"], "Reason_Code": r["reason"],
                   "S0_Decision_Class": r["s0_class"], "Parameter_State": r["param_state"], "N_E": r["n_E"],
                   "Metric_Status": r["metric"], "Loss_Challenger": r["loss_ch"], "Loss_S0": r["loss_s0"]}
            if r["class"] == "FORECAST":
                row.update(challenger_row(r, s0, cand))
            f.write(json.dumps(row, sort_keys=True) + "\n")
    gp.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for p in (rp, gp):
        print(p, _sha_bytes(p.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
