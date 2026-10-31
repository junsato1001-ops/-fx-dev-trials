# DEV-USDJPY-SIMPLE-02 — Implementation Notes

DEV-USDJPY-SIMPLE-02 = DEV-USDJPY-SIMPLE-01 implementation a7921007 with only: the workflow YAML
quoting fix (I-8), a YAML parse regression test, and Trial ID mechanical changes. DEV-USDJPY-SIMPLE-01
was ABORTed before any job ran (V1 DEV-TRIAL-REGISTRY Event_Seq 2). Sections I-1 to I-7 are unchanged
from the 01 candidate v0.2 and apply to 02.

Points where the implementation had to fix a detail the spec does not spell out. Each is flagged
for むすび's implementation audit; none was chosen after seeing any Development value (none has
been computed on the real snapshot).

## I-1 Data access
- Reader is a byte scanner: the header is decoded only to find the three allowlisted names; data
  rows are scanned by delimiter position and only the Date/JPY/USD byte ranges are sliced. Bytes of
  other columns pass through `bytes.find`/`bytes.count` but are never sliced, decoded or stored.
- Fail-closed on: any `"`; field count ≠ header; duplicate date; allowlisted name missing or duplicated.
- V1 `ledger_builder.read_ecb_zip()`/`pair_series()`/`build_rows()` are NOT called (they would read
  every column). Only `make_one()` and `to_strings()` are used; rows are built for every snapshot
  date as origin (as `build_rows` does for 1D).

## I-2 S0 pools
- Pools are regenerated as in V1 `m1macro/s0link.py`: `empirical_summary` is wrapped to capture
  `len(sorted_z)` and checked against `Pool_N`; pool at origin t = `sort(cand[:Pool_N_t])` where
  `cand` is `_candidate_pool_source(group)` order. The S0 output itself is the frozen function's.
- Exact S0 identity (SPEC v0.3 §3.2; MODEL-PART1-V1 §3), revised in candidate v0.1: whenever
  mu == 0 (DEFAULT_PARENT, ESTIMATED with beta == 0, or F_MOM == 0) the challenger Point / PI10 /
  PI90 / P_Down / P_Neutral / P_Up are the frozen S0 output strings verbatim (`Source = S0_VERBATIM`),
  mu is stored as +0.0, and loss_ch is set to loss_s0 (d = 0 exactly). Tests compare strings with
  assertEqual for all three cases (`test_default_parent_exact_s0`, `test_estimated_beta_zero_exact_s0`,
  `test_f_mom_zero_exact_s0`).
- mu != 0: x'_i = sigma Z_i + mu; Point = sigma * fsum(Z)/N + mu (same mean as S0); PI by S0 ranks;
  classes counted on x' against b = 0.25 sigma. All numbers are serialized with S0's `f17`.

## I-3 Decision classes
- S0 `NOT_APPLICABLE` is passed through as `NOT_APPLICABLE` (reason = S0 reason), not mapped to
  MODEL_PREREQ_UNAVAILABLE. **Interpretation flag.**
- Other non-FORECAST S0 → MODEL_PREREQ_UNAVAILABLE / PARENT_PREREQ_UNAVAILABLE.
- F_MOM undefined (fix at v or at v−20 TARGET days missing from the USD/JPY series) →
  MODEL_PREREQ_UNAVAILABLE / FEATURE_PREREQ_UNAVAILABLE.
- sigma ≤ 0 → FORECAST with metric METRIC_UNDEFINED_ZERO_SCALE (excluded from loss).
- F_MOM window counts TARGET publication days via `previous_target_day` (calendar days, not
  series positions). **Interpretation flag.**

## I-4 Estimator
- Objective: sum_i CRPS(F_i^{S0}(· − beta F_i), y_i)/sigma_i = sum over pool points of
  (|F_i|/(n_i sigma_i)) |beta − (y_i − sigma_i z)/F_i| + const (verified numerically).
  Its minimizer is the weighted median (LP optimum). Origins with F_i = 0 count in n_E but add no
  points.
- Exact tie (cumulative weight equals half exactly in floating point): the minimizing interval
  [v_k, v_next] → point closest to 0 (min L2 norm). Ties are decided by exact float equality.
  **Interpretation flag.**
- Training set E_t: in-scope, S0 FORECAST, F defined, sigma > 0, y observed, Target_End_i ≤ LAF_t.
  An assertion stops the run if any such i has origin ≥ t.
- ESTIMATED once n_E ≥ 250 (sticky); before that beta = 0 → challenger ≡ S0.
- Gate sample = FORECAST ∧ ELIGIBLE ∧ ESTIMATED; full-common-sample D̄ (incl. DEFAULT_PARENT) is report-only.

## I-5 Inference
- Stationary bootstrap index algorithm (fixed before any result): per replicate draw
  u ~ U(0,1)^T then s ~ U{0..T−1}^T; position 0 starts a block; i > 0 starts a block iff u_i < 1/L;
  block started at p uses s_p and continues circularly. Generator PCG64(20261016) re-created for
  each test (block 4 and block 12 use the same seed). Reference-loop test included.
- p_sup = (1 + #{D*_b − D̄ ≤ D̄})/(B+1); p_inf = (1 + #{D*_b − D̄ ≥ D̄})/(B+1). p_dev = primary p_sup.
- HAC: Newey-West Bartlett, lag 4, normal approximation, one-sided. Report only.
- Half split: first ⌊N/2⌋, second N − ⌊N/2⌋ (the later half takes the extra row).
- Gate: N_EVAL ≥ 250 ∧ p_dev < 0.05 ∧ D̄_second < 0 → FORWARD_CANDIDATE_ELIGIBLE; else NOT_ELIGIBLE;
  N_EVAL < 250 → NOT_EVALUABLE (no inference computed). Flags BLOCK_SENSITIVITY_DISAGREEMENT /
  SENSITIVITY_DISAGREEMENT are true only for SUP vs INF; they do not change the class.
- 90% percentile interval of D* is reported (descriptive only).

## I-6 Outputs
- preflight: dates, counts and states only (test checks no float appears).
- run: GATE.json and ROWS.jsonl (in-scope rows). Existing artifacts are never overwritten.
- GATE.json `diagnostics` (candidate v0.1; SPEC v0.3 §10 "必ず報告するもの"; never used by the Gate,
  computed also when NOT_EVALUABLE): Coverage per EVAL §4.2 for both models; DEFAULT_PARENT and
  METRIC_UNDEFINED_ZERO_SCALE counts; beta trajectory summary; RPS, Multiclass Brier, IS/sigma,
  PI80 coverage, Deterministic VRC (EVAL §3.2-§3.4) and direction diagnostics, each for challenger
  and S0, on the Gate sample (ESTIMATED) and the full common sample. Definitions in
  `devtrial/diagnostics.py` docstring.
  **Interpretation flag:** the direction diagnostics (modal-class hit with ties excluded and counted,
  point-sign hit excluding zeros, mean probability of the realized class) are not defined in any
  frozen document; they are fixed here before any result and are descriptive only.
- GATE.json `provenance` (§13): snapshot, config, devtrial code and vendor SHA-256; numpy/python versions.
- ROWS.jsonl FORECAST rows carry F_MOM_Fix_Date (v) and F_MOM_Base_Fix_Date (v − 20 TARGET days) (§13).
  The workflow writes to the runner temp dir and uploads as an Actions artifact; nothing is
  committed back.

## I-7 Environment (candidate v0.2: runtime pinning, Freeze-prep v0.1 §3.5; F-FC-1 option A)
- Python 3.13.17 (`setup-python` exact patch version; stable ubuntu-24.04 x64 build listed in
  actions/python-versions versions-manifest.json at commit 160ca1e2).
- numpy 2.5.3 from `requirements.txt`, installed with `--require-hashes --only-binary=:all: --no-deps`.
  Wheel numpy-2.5.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl, SHA-256
  a5fa86b8…e25321 confirmed by むすび on the PyPI file page and PyPI provenance (Claude could not reach PyPI).
- A workflow step stops unless Python == 3.13.17 and numpy == 2.5.3; GATE provenance records both.
- GitHub Actions pinned by commit SHA, resolved with `git ls-remote` from the tags used before:
  checkout v4 = v4.4.0 11d5960a…, setup-python v5 = v5.6.0 a26af69b…, upload-artifact v4 = v4.6.2 ea165f8d….
- Local tests ran on Python 3.13.17 and numpy 2.5.3 (local install not hash-verified).

## I-8 Workflow YAML (DEV-USDJPY-SIMPLE-02)
- 01's workflow line `run: python -m pip install --require-hashes --only-binary=:all: --no-deps -r requirements.txt`
  was not valid YAML (`: ` inside a plain scalar); GitHub could not load the workflow. The value is now a
  double-quoted scalar with identical content.
- `tests/test_workflow_yaml.py`: (1) always: every single-line `run:` value containing `: ` must be quoted;
  (2) when PyYAML is installed (local audit environment; not installed in CI, where the test is skipped):
  full parse, triggers == {workflow_dispatch} only, the install command string equals the frozen command.
