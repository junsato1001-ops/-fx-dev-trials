import unittest

import numpy as np

from devtrial import model


def crps_bruteforce(x, y):
    n = x.size
    return np.abs(x - y).mean() - np.abs(x[:, None] - x[None, :]).sum() / (2 * n * n)


def fen_median(vals, wts):
    order = np.argsort(vals, kind="stable")
    vs = vals[order]
    rank = np.empty(vals.size, dtype=np.int64); rank[order] = np.arange(vals.size)
    fen = model._Fenwick(vals.size)
    fen.add(rank, wts)
    return model.weighted_median_min_norm(vs, fen, float(wts.sum()))


class EstimatorTests(unittest.TestCase):
    def test_crps_matches_bruteforce(self):
        rng = np.random.Generator(np.random.PCG64(1))
        for n in (1, 2, 7, 50):
            x = np.sort(rng.standard_normal(n))
            for y in (-1.3, 0.0, 0.4):
                self.assertAlmostEqual(model.crps_sorted(x, y), crps_bruteforce(x, y), places=12)

    def test_weighted_median_is_minimizer(self):
        rng = np.random.Generator(np.random.PCG64(2))
        for _ in range(200):
            n = int(rng.integers(1, 40))
            v = rng.standard_normal(n) * 3
            w = rng.random(n) + 0.01
            b = fen_median(v, w)
            obj = lambda t: float(np.sum(w * np.abs(t - v)))
            best = min(obj(t) for t in v)
            self.assertLessEqual(obj(b), best + 1e-9)

    def test_partial_insertion(self):
        rng = np.random.Generator(np.random.PCG64(3))
        v = rng.standard_normal(30); w = rng.random(30) + 0.1
        order = np.argsort(v, kind="stable"); vs = v[order]
        rank = np.empty(30, dtype=np.int64); rank[order] = np.arange(30)
        fen = model._Fenwick(30); tot = 0.0
        for k in range(0, 30, 5):
            fen.add(rank[k:k + 5], w[k:k + 5]); tot += float(w[k:k + 5].sum())
            b = model.weighted_median_min_norm(vs, fen, tot)
            vv, ww = v[:k + 5], w[:k + 5]
            obj = lambda t: float(np.sum(ww * np.abs(t - vv)))
            self.assertLessEqual(obj(b), min(obj(t) for t in vv) + 1e-9)

    def test_exact_tie_min_norm(self):
        self.assertEqual(fen_median(np.array([-2.0, 3.0]), np.array([1.0, 1.0])), 0.0)
        self.assertEqual(fen_median(np.array([1.0, 3.0]), np.array([1.0, 1.0])), 1.0)
        self.assertEqual(fen_median(np.array([-5.0, -2.0]), np.array([1.0, 1.0])), -2.0)
        self.assertEqual(fen_median(np.zeros(0), np.zeros(0)), 0.0)

    def test_loss_equals_weighted_abs_plus_const(self):
        """sum_i CRPS(sigma_i z_i + beta F_i, y_i)/sigma_i - sum_pts w|beta - r| is constant in beta."""
        rng = np.random.Generator(np.random.PCG64(4))
        origins = []
        for _ in range(6):
            n = int(rng.integers(3, 12))
            origins.append((np.sort(rng.standard_normal(n)), rng.random() + 0.2,
                            rng.standard_normal() * 0.5, rng.standard_normal() * 0.01))
        def lhs(beta):
            return sum(model.crps_sorted(s * z + beta * F, y) / s for z, s, y, F in origins)
        def rhs(beta):
            t = 0.0
            for z, s, y, F in origins:
                r = (y - s * z) / F
                t += np.sum(abs(F) / (z.size * s) * np.abs(beta - r))
            return t
        consts = [lhs(b) - rhs(b) for b in (-30.0, -1.0, 0.0, 0.7, 12.0)]
        for c in consts:
            self.assertAlmostEqual(c, consts[0], places=9)


if __name__ == "__main__":
    unittest.main()
