# -fx-dev-trials

Development diagnostic trials for the FX forecasting project. Separate from the V1 ledger
(`-fx-forward-ledger`, never modified) and the V2 pipeline (`-fx-forward-ledger-v2`).

## DEV-USDJPY-SIMPLE-02 — status

Successor of DEV-USDJPY-SIMPLE-01 (ABORT before execution: workflow YAML not loadable). Same scientific
spec, model and Gate; only the YAML quoting fix, a YAML parse test and Trial ID changes.

| Item | State |
|---|---|
| Branch `impl/dev-usdjpy-simple-02` | Implementation Candidate (not merged to `main`) |
| Freeze | NOT APPROVED |
| Preflight (Freeze-prep v0.1 §5.2) | NOT AUTHORIZED |
| Development run / Gate | NOT AUTHORIZED |
| Forward | Out of scope for this repository state |

`preflight` and `run` refuse to start unless `trials/DEV-USDJPY-SIMPLE-02/AUTHORIZATION.json`
exists and lists the mode. That file is not part of this candidate. The authorization check runs
before the snapshot is opened.

Specs: `DEV-USDJPY-SIMPLE-01-SPEC-V0.3.draft.md` and `...-FREEZE-PREP-V0.1.draft.md` in
`-fx-forward-ledger/drafts/` (commits, blobs and SHA-256 in `trials/DEV-USDJPY-SIMPLE-02/config.json`).

## Layout

- `vendor/v1/` — byte copies of V1 files at commit 17cc7c6d; SHA-256 verified before import.
- `data/eurofxref-hist_20261009.zip` — the V1 snapshot (SHA-256 f36e0aff…), read only through `devtrial/reader.py`.
- `devtrial/reader.py` — allowlist reader (Date, JPY, USD only).
- `devtrial/frozen.py` — vendor verification and loading.
- `devtrial/model.py` — ledger, S0 pools, F_MOM, prequential beta, decision classes, losses.
- `devtrial/stats.py` — stationary bootstrap, p-values, HAC, half split.
- `devtrial/summary.py` — challenger Point/PI/class probabilities (S0 verbatim when mu = 0).
- `devtrial/diagnostics.py` — report-only §10 diagnostics (not used by the Gate).
- `devtrial/run.py` — `selftest` | `preflight` | `run`.
- `tests/` — synthetic data only; no test opens the snapshot.
- `.github/workflows/dev-trial.yml` — `workflow_dispatch` only.

`python devtrial/run.py selftest` runs the test suite.

See `IMPLEMENTATION-NOTES.md` for interpretation points to audit.
