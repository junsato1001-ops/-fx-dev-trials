# DEV-USDJPY-SIMPLE-01 — Implementation Notes (candidate)

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
- Test `test_s0_point_matches_frozen_output`: with mu = 0 the challenger row reproduces S0 point,
  PI10/PI90 and P_Down/Neutral/Up (synthetic data).

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
  The workflow writes to the runner temp dir and uploads as an Actions artifact; nothing is
  committed back.

## I-7 Environment
- Python 3.13, numpy 2.5.3 (pinned in the workflow; local tests ran with the same versions).
