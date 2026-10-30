"""End-to-end on synthetic data only, through the vendored frozen modules."""
import datetime as dt
import unittest

import numpy as np

from devtrial import diagnostics, model, run, summary
from tests import _synth

M0, MP, S0, LB = _synth.modules()
FLAT = (dt.date(2017, 9, 1), dt.date(2017, 10, 31))
ROWS = _synth.allow_rows(MP, gaps={dt.date(2017, 3, 15)}, flat=FLAT)
CUTOFF = "2018-06-01"


def run_cfg(**over):
    return _synth.cfg(development_target_end_max=CUTOFF, **over)


class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs, cls.cand = model.run_model(run_cfg(n_warmup=100), M0, MP, S0, LB, ROWS, return_pool_source=True)

    def test_beta_zero_identity_with_s0(self):
        recs = model.run_model(run_cfg(n_warmup=10 ** 9), M0, MP, S0, LB, ROWS)
        g = [r for r in recs if r["metric"] == "ELIGIBLE"]
        self.assertTrue(g)
        for r in g:
            self.assertEqual(r["param_state"], "DEFAULT_PARENT")
            self.assertEqual(r["mu"], 0.0)
            self.assertEqual(r["loss_ch"], r["loss_s0"])

    def _assert_exact_s0(self, rs, cand, require_state=None):
        n = 0
        for r in rs:
            if r["class"] != "FORECAST" or r["mu"] != 0.0:
                continue
            if require_state and r["param_state"] != require_state:
                continue
            ch = summary.challenger_summary(r, cand, S0)
            self.assertEqual(ch["Source"], "S0_VERBATIM")
            self.assertEqual({k: ch[k] for k in summary.FIELDS}, summary.s0_summary(r))  # string equality
            row = run.challenger_row(r, S0, cand)
            for k, sk in summary.S0_FIELDS.items():
                self.assertEqual(row[k], r["s0_row"][sk])
            self.assertEqual(row["Mu"], S0.f17(0.0))
            if r["metric"] == "ELIGIBLE":
                self.assertEqual(r["loss_ch"], r["loss_s0"])
            n += 1
        return n

    def test_default_parent_exact_s0(self):
        rs, cand = model.run_model(run_cfg(n_warmup=10 ** 9), M0, MP, S0, LB, ROWS, return_pool_source=True)
        self.assertGreater(self._assert_exact_s0(rs, cand, "DEFAULT_PARENT"), 100)

    def test_estimated_beta_zero_exact_s0(self):
        orig = model.weighted_median_min_norm
        model.weighted_median_min_norm = lambda *a, **k: 0.0
        try:
            rs, cand = model.run_model(run_cfg(n_warmup=100), M0, MP, S0, LB, ROWS, return_pool_source=True)
        finally:
            model.weighted_median_min_norm = orig
        self.assertTrue(any(r["param_state"] == "ESTIMATED" for r in rs))
        self.assertGreater(self._assert_exact_s0(rs, cand, "ESTIMATED"), 100)

    def test_f_mom_zero_exact_s0(self):
        rows = [r for r in self.recs if r["class"] == "FORECAST" and r["f_mom"] == 0.0
                and r["param_state"] == "ESTIMATED"]
        self.assertTrue(rows)
        self.assertTrue(all(r["mu"] == 0.0 and r["beta"] != 0.0 for r in rows))
        self.assertEqual(self._assert_exact_s0(rows, self.cand), len(rows))

    def test_shifted_summary_formula(self):
        rows = [r for r in self.recs if r["class"] == "FORECAST" and r["mu"] != 0.0]
        self.assertTrue(rows)
        for r in rows[:: max(1, len(rows) // 50)]:
            s = summary.as_float(summary.challenger_summary(r, self.cand, S0))
            z = np.sort(self.cand[:r["pool_n"]]); x = r["sigma"] * z + r["mu"]; b = 0.25 * r["sigma"]
            k10, k90 = S0.quantile_ranks(z.size)
            self.assertAlmostEqual(s["Point"], r["sigma"] * z.mean() + r["mu"], places=14)
            self.assertEqual(s["PI10"], float(x[k10 - 1])); self.assertEqual(s["PI90"], float(x[k90 - 1]))
            self.assertEqual(s["P_Down"], np.sum(x < -b) / z.size)
            self.assertEqual(s["P_Up"], np.sum(x > b) / z.size)
            self.assertAlmostEqual(s["P_Down"] + s["P_Neutral"] + s["P_Up"], 1.0, places=15)

    def test_diagnostics_report(self):
        c = run_cfg(n_warmup=100, n_eval_min=50, bootstrap=dict(_synth.CFG["bootstrap"], replicates=200))
        g = run.gate(c, self.recs)
        rec = run.gate_record(c, self.recs, self.cand, S0)
        self.assertEqual({k: rec[k] for k in g}, g)  # diagnostics never change the Gate fields
        d = rec["diagnostics"]
        for k in ("coverage", "counts", "beta_trajectory", "secondary_calibration_direction"):
            self.assertIn(k, d)
        self.assertEqual(d["coverage"]["challenger"]["decision_coverage"], 1.0)
        self.assertEqual(d["coverage"]["s0"]["decision_coverage"], 1.0)
        self.assertGreater(d["counts"]["DEFAULT_PARENT_forecast_rows"], 0)
        self.assertIn("METRIC_UNDEFINED_ZERO_SCALE", d["counts"])
        for samp in ("gate_sample_ESTIMATED", "full_common_sample"):
            for m in ("challenger", "s0"):
                blk = d["secondary_calibration_direction"][samp][m]
                for k in ("mean_RPS", "mean_Brier", "mean_IS_over_sigma", "PI80_coverage", "VRC_bin_mass",
                          "direction", "mean_CRPS_over_sigma"):
                    self.assertIn(k, blk)
                self.assertAlmostEqual(sum(blk["VRC_bin_mass"]), blk["N_common"], places=9)
        gs = d["secondary_calibration_direction"]["gate_sample_ESTIMATED"]
        self.assertEqual(gs["challenger"]["N_primary_eligible"], g["N_EVAL"])
        self.assertAlmostEqual(gs["challenger"]["mean_CRPS_over_sigma"] - gs["s0"]["mean_CRPS_over_sigma"],
                               g["primary"]["D"], places=12)
        bt = d["beta_trajectory"]
        self.assertEqual(bt["N_positive"] + bt["N_negative"] + bt["N_zero"], bt["N_ESTIMATED_rows"])
        prov = rec["provenance"]
        self.assertIn("devtrial/model.py", prov["code_sha256"])
        # NOT_EVALUABLE still carries diagnostics
        ne = run.gate_record(run_cfg(n_eval_min=10 ** 9), self.recs, self.cand, S0)
        self.assertEqual(ne["result_class"], "NOT_EVALUABLE")
        self.assertIn("diagnostics", ne)

    def test_row_fix_dates(self):
        r = next(r for r in self.recs if r["class"] == "FORECAST")
        row = run.challenger_row(r, S0, self.cand)
        self.assertEqual(row["F_MOM_Fix_Date"], r["laf"].isoformat())
        self.assertEqual(dt.date.fromisoformat(row["F_MOM_Base_Fix_Date"]),
                         model.f_mom_base_date(MP, r["laf"], 20))
        self.assertEqual(MP.count_target_intervals(r["f_mom_base_fix"], r["laf"]), 20)

    def test_classes(self):
        scope = [r for r in self.recs if r["in_scope"]]
        classes = {r["class"] for r in scope}
        self.assertIn("FORECAST", classes)
        for r in scope:
            if r["s0_class"] == "NOT_APPLICABLE":
                self.assertEqual(r["class"], "NOT_APPLICABLE")
            elif r["s0_class"] != "FORECAST":
                self.assertEqual(r["reason"], "PARENT_PREREQ_UNAVAILABLE")
            if r["target_end"] is not None:
                self.assertLessEqual(r["target_end"], dt.date.fromisoformat(CUTOFF))
        self.assertFalse(any(r["in_scope"] for r in self.recs if r["target_end"] and
                             r["target_end"] > dt.date.fromisoformat(CUTOFF)))

    def test_beta_equals_bruteforce_on_matured_set(self):
        recs = self.recs
        est = [r for r in recs if r["param_state"] == "ESTIMATED"]
        self.assertTrue(est)
        rng = np.random.Generator(np.random.PCG64(0))
        picks = [est[0], est[-1]] + [est[i] for i in rng.integers(0, len(est), 3)]
        for t in picks:
            train = [r for r in recs if r["in_scope"] and r["s0_class"] == "FORECAST" and r["f_mom"] is not None
                     and r["sigma"] and r["sigma"] > 0 and r["y"] is not None and r["outcome"] == "OBSERVED"
                     and r["target_end"] <= t["laf"]]
            self.assertTrue(all(r["origin"] < t["origin"] for r in train))
            self.assertEqual(len(train), t["n_E"])

            def obj(b):
                return sum(model.crps_sorted(r["sigma"] * np.sort(self.cand[:r["pool_n"]]) + b * r["f_mom"], r["y"])
                           / r["sigma"] for r in train)
            cands = [(r["y"] - r["sigma"] * z) / r["f_mom"] for r in train if r["f_mom"] != 0
                     for z in self.cand[:r["pool_n"]]]
            best = min(obj(c) for c in cands[:: max(1, len(cands) // 400)])
            self.assertLessEqual(obj(t["beta"]), best + 1e-9)
            for eps in (1e-6, -1e-6):
                self.assertLessEqual(obj(t["beta"]), obj(t["beta"] + eps) + 1e-12)

    def test_feature_gap(self):
        gap_rows = [r for r in self.recs if r["in_scope"] and r["s0_class"] == "FORECAST" and r["f_mom"] is None]
        self.assertTrue(gap_rows)
        for r in gap_rows:
            self.assertEqual(r["reason"], "FEATURE_PREREQ_UNAVAILABLE")

    def test_gate_runs_and_is_deterministic(self):
        c = run_cfg(n_warmup=100, n_eval_min=50, bootstrap=dict(_synth.CFG["bootstrap"], replicates=300))
        g1 = run.gate(c, self.recs)
        g2 = run.gate(c, self.recs)
        self.assertEqual(g1, g2)
        self.assertIn(g1["result_class"], {"FORWARD_CANDIDATE_ELIGIBLE", "NOT_ELIGIBLE"})
        self.assertEqual(g1["halves"]["n_first"] + g1["halves"]["n_second"], g1["N_EVAL"])
        g3 = run.gate(run_cfg(n_eval_min=10 ** 9), self.recs)
        self.assertEqual(g3["result_class"], "NOT_EVALUABLE")
        self.assertNotIn("p_dev", g3)

    def test_preflight_summary_has_no_values(self):
        pf = run.preflight_summary(run_cfg(n_warmup=100), self.recs)
        allowed_top = {"schema_version", "trial_id", "first_evaluation_origin", "first_estimated_origin",
                       "last_in_scope_origin", "counts", "n_eval_min", "evaluable"}
        self.assertEqual(set(pf), allowed_top)

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            else:
                self.assertTrue(o is None or isinstance(o, (int, str, bool)), repr(o))
                self.assertFalse(isinstance(o, float))
        walk(pf)
        c = pf["counts"]
        self.assertEqual(sum(c["challenger_decision_class"].values()), c["in_scope_rows"])

    def test_authorization_guard(self):
        with self.assertRaises(SystemExit) as cm:
            run.check_authorization(_synth.CFG, "preflight")
        self.assertIn("NOT AUTHORIZED", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
