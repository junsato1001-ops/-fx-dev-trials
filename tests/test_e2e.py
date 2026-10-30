"""End-to-end on synthetic data only, through the vendored frozen modules."""
import datetime as dt
import unittest

import numpy as np

from devtrial import model, run
from tests import _synth

M0, MP, S0, LB = _synth.modules()
ROWS = _synth.allow_rows(MP, gaps={dt.date(2017, 3, 15)})
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

    def test_s0_point_matches_frozen_output(self):
        """Challenger row with mu=0 reproduces the frozen S0 point / PI / probabilities."""
        recs = model.run_model(run_cfg(n_warmup=10 ** 9), M0, MP, S0, LB, ROWS, return_pool_source=True)
        rs, cand = recs
        checked = 0
        for r in rs:
            if r["class"] != "FORECAST":
                continue
            row = run.challenger_row(r, S0, cand)
            so = r["s0_row"]
            self.assertAlmostEqual(row["PI10"], float(so["PI10"]), places=14)
            self.assertAlmostEqual(row["PI90"], float(so["PI90"]), places=14)
            self.assertAlmostEqual(row["Point"], float(so["S0_Point_LogReturn"]), places=14)
            for k in ("P_Down", "P_Neutral", "P_Up"):
                self.assertAlmostEqual(row[k], float(so[k]), places=14)
            checked += 1
        self.assertGreater(checked, 100)

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
