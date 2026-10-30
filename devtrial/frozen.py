"""Load the vendored V1 frozen modules after verifying their SHA-256 (Freeze-prep v0.1 §3.3-§3.4).

Only these functions are used:
- M0 v1.2 and ORIGIN_TARGET_MAP_V1: through ledger_builder.make_one() and the TARGET calendar.
- S0 v1.1: generate_s0(), _group_rows(), _candidate_pool_source(), empirical_summary().
- ledger_builder: make_one() and to_strings() only. read_ecb_zip(), pair_series(), build_rows(),
  load_frozen() and write_csv() are NOT called (they would read non-allowlisted columns or need
  files that are not vendored).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "trials" / "DEV-USDJPY-SIMPLE-01" / "config.json"


class FrozenMismatch(RuntimeError):
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_vendor(cfg: dict) -> None:
    for rel, expected in cfg["vendor_sha256"].items():
        got = sha256_file(ROOT / rel)
        if got != expected:
            raise FrozenMismatch(f"VENDOR_SHA_MISMATCH {rel}")


def _load(modname: str, rel: str):
    spec = importlib.util.spec_from_file_location(modname, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_modules(cfg: dict):
    """Return (m0, mp, s0, lb) after the SHA-256 check."""
    verify_vendor(cfg)
    m0 = _load("devtrial_frozen_m0_v1_2", "vendor/v1/frozen/m0_walkforward_v1_2.py")
    mp = _load("devtrial_frozen_origin_target_map_v1", "vendor/v1/frozen/origin_target_map_v1.py")
    s0 = _load("devtrial_frozen_s0_v1_1", "vendor/v1/frozen/s0_empclim_std_exp_v1_1.py")
    lb = _load("devtrial_v1_ledger_builder", "vendor/v1/forward/ledger_builder.py")
    return m0, mp, s0, lb
