import unittest

import numpy as np

from devtrial import stats


class StatsTests(unittest.TestCase):
    def test_bootstrap_deterministic_and_seed_reset(self):
        d = np.random.Generator(np.random.PCG64(9)).standard_normal(300)
        a = stats.stationary_bootstrap_means(d, 4, 200, 20261016)
        b = stats.stationary_bootstrap_means(d, 4, 200, 20261016)
        self.assertTrue(np.array_equal(a, b))
        c = stats.stationary_bootstrap_means(d, 12, 200, 20261016)
        self.assertFalse(np.array_equal(a, c))

    def test_constant_series(self):
        d = np.full(50, 0.3)
        self.assertTrue(np.allclose(stats.stationary_bootstrap_means(d, 4, 20, 1), 0.3))

    def test_index_algorithm_reference(self):
        """Explicit loop implementation of the documented algorithm gives identical means."""
        d = np.random.Generator(np.random.PCG64(5)).standard_normal(37)
        L, B, seed = 4, 30, 11
        rng = np.random.Generator(np.random.PCG64(seed))
        ref = []
        for _ in range(B):
            u = rng.random(37); s = rng.integers(0, 37, size=37)
            idx = []
            cur = None
            for i in range(37):
                if i == 0 or u[i] < 1 / L:
                    cur = s[i]
                else:
                    cur = (cur + 1) % 37
                idx.append(cur)
            ref.append(d[idx].mean())
        self.assertTrue(np.allclose(stats.stationary_bootstrap_means(d, L, B, seed), ref, rtol=0, atol=1e-15))

    def test_pvalue_formula(self):
        d = np.random.Generator(np.random.PCG64(6)).standard_normal(120) - 0.3
        res = stats.bootstrap_test(d, 4, 500, 3)
        Ds = stats.stationary_bootstrap_means(d, 4, 500, 3)
        D = d.mean()
        self.assertEqual(res["p_sup"], (1 + np.sum(Ds - D <= D)) / 501)
        self.assertEqual(res["p_inf"], (1 + np.sum(Ds - D >= D)) / 501)
        self.assertLess(res["p_sup"], 0.05)

    def test_hac_lag0_iid(self):
        d = np.random.Generator(np.random.PCG64(8)).standard_normal(200)
        h = stats.hac_test(d, 0)
        se = np.sqrt(np.mean((d - d.mean()) ** 2) / d.size)
        self.assertAlmostEqual(h["t"], d.mean() / se, places=12)

    def test_verdicts_and_halves(self):
        self.assertEqual(stats.verdict(0.01, 0.99, 0.05), "SUP")
        self.assertEqual(stats.verdict(0.99, 0.01, 0.05), "INF")
        self.assertEqual(stats.verdict(0.5, 0.5, 0.05), "INCONCLUSIVE")
        self.assertTrue(stats.disagreement("SUP", "INF"))
        self.assertFalse(stats.disagreement("SUP", "INCONCLUSIVE"))
        self.assertEqual(stats.halves(251), (125, 126))
        self.assertEqual(stats.halves(250), (125, 125))


if __name__ == "__main__":
    unittest.main()
