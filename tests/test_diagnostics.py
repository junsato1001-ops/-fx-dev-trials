import unittest

import numpy as np

from devtrial import diagnostics as dg


class DiagnosticsUnit(unittest.TestCase):
    def test_realized_class_boundaries(self):
        self.assertEqual(dg.realized_class(-0.25, 1.0), 1)
        self.assertEqual(dg.realized_class(0.25, 1.0), 1)
        self.assertEqual(dg.realized_class(-0.2500001, 1.0), 0)
        self.assertEqual(dg.realized_class(0.2500001, 1.0), 2)
        self.assertEqual(dg.realized_class(0.0, 0.0), 1)
        self.assertEqual(dg.realized_class(1e-9, 0.0), 2)

    def test_rps_brier(self):
        p, o = (0.2, 0.5, 0.3), (0.0, 0.0, 1.0)
        self.assertAlmostEqual(dg.rps(p, o), 0.2 ** 2 + 0.7 ** 2)
        self.assertAlmostEqual(dg.brier(p, o), 0.04 + 0.25 + 0.49)

    def test_interval_score(self):
        self.assertAlmostEqual(dg.interval_score(-1, 1, 0), 2)
        self.assertAlmostEqual(dg.interval_score(-1, 1, 2), 2 + 10 * 1)
        self.assertAlmostEqual(dg.interval_score(-1, 1, -1.5), 2 + 10 * 0.5)

    def test_vrc(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        b = dg.vrc_bins(x, 2.5)          # L=2, E=0 -> [0.4, 0.6]
        self.assertTrue(np.allclose(b[[4, 5]], [0.5, 0.5])); self.assertAlmostEqual(b.sum(), 1)
        b = dg.vrc_bins(x, 2.0)          # L=1, E=1 -> [0.2, 0.6]
        self.assertTrue(np.allclose(b[[2, 3, 4, 5]], [0.25, 0.25, 0.25, 0.25]))
        b = dg.vrc_bins(x, 9.0)          # L=4 -> [0.8, 1.0]
        self.assertTrue(np.allclose(b[[8, 9]], [0.5, 0.5]))

    def test_coverage(self):
        scope = [{"c": "NOT_APPLICABLE"}, {"c": "MODEL_PREREQ_UNAVAILABLE"}, {"c": "FORECAST"},
                 {"c": "FORECAST"}, {"c": "ABSTAIN_SYSTEM"}]
        cv = dg._coverage(scope, "c")
        self.assertEqual(cv["S"], 4)
        self.assertAlmostEqual(cv["prerequisite_readiness"], 3 / 4)
        self.assertEqual(cv["decision_coverage"], 1.0)
        self.assertAlmostEqual(cv["operational_completion"], 1 - 1 / 3)


if __name__ == "__main__":
    unittest.main()
