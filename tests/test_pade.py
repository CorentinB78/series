import unittest
import numpy as np
from mpmath import pade as mp_pade
from series.resummation import one_over_series, IdentityTransform, EulerTransform
from series.pade import pade, eval_pade
from series.pade import find_poles_with_pade


class TestPade(unittest.TestCase):

    def test_mpmath_pade(self):
        ### compare with mpmath pade
        series = np.array([2., 4.5, -6.])
        for rank in [(1, 1), (0, 2), (2, 0)]:
            p1, q1 = pade(series, rank[0], rank[1])
            p2, q2 = mp_pade(series, rank[0], rank[1])

            p2 = np.asarray(p2, dtype=float)
            q2 = np.asarray(q2, dtype=float)

            np.testing.assert_allclose(p1, p2, rtol=1e-14)
            np.testing.assert_allclose(q1, q2, rtol=1e-14)

    def test_real_exponential(self):
        exp_series = np.array([1., 1., 1/2., 1/6., 1/24., 1/120.])
        rtol = 1e-13
        atol = 1e-16

        p, q = pade(exp_series, 0, 1)
        np.testing.assert_allclose(p, [1.], rtol, atol)
        np.testing.assert_allclose(q, [1., -1.], rtol, atol)

        p, q = pade(exp_series, 3, 1)
        np.testing.assert_allclose(p, [1., 3/4., 1/4., 1/24.], rtol, atol)
        np.testing.assert_allclose(q, [1., -1/4.], rtol, atol)

        p, q = pade(exp_series, 2, 0)
        np.testing.assert_allclose(p, [1., 1., 1/2.], rtol, atol)
        np.testing.assert_allclose(q, [1.], rtol, atol)

        p, q = pade(exp_series, 2, 3)
        # print(p - np.array([1., 2/5., 1/20.]))
        np.testing.assert_allclose(p, [1., 2/5., 1/20.], rtol, atol)
        np.testing.assert_allclose(q, [1., -3/5., 3/20., -1/60.], rtol, atol)

    def test_imag_exponential(self):
        exp_series = np.array([1., 1.j, -1/2., -1j/6., 1/24., 1j/120.])
        rtol = 1e-13
        atol = 1e-16

        p, q = pade(exp_series, 0, 1)
        np.testing.assert_allclose(p, [1.], rtol, atol)
        np.testing.assert_allclose(q, [1., -1.j], rtol, atol)

        p, q = pade(exp_series, 3, 1)
        np.testing.assert_allclose(p, [1., 3j/4., -1/4., -1j/24.], rtol, atol)
        np.testing.assert_allclose(q, [1., -1j/4.], rtol, atol)

        p, q = pade(exp_series, 2, 0)
        np.testing.assert_allclose(p, [1., 1.j, -1/2.], rtol, atol)
        np.testing.assert_allclose(q, [1.], rtol, atol)

        p, q = pade(exp_series, 2, 3)
        # print(p - np.array([1., 2/5., 1/20.]))
        np.testing.assert_allclose(p, [1., 2j/5., -1/20.], rtol, atol)
        np.testing.assert_allclose(q, [1., -3j/5., -3/20., 1j/60.], rtol, atol)

    def test_orders(self):
        series = np.array([1., 3., 2.])
        p, q = pade(series, 2, 0)
        self.assertEqual(len(p), 3)
        self.assertEqual(len(q), 1)
        np.testing.assert_allclose(series, p)
        np.testing.assert_allclose([1.], q)

        p, q = pade(series, 0, 2)
        self.assertEqual(len(p), 1)
        self.assertEqual(len(q), 3)
        np.testing.assert_allclose([1.], p)
        np.testing.assert_allclose(one_over_series(series), q)

        for l in range(4):
            self.assertRaises(ValueError, pade, series, l, 3-l)


class TestEvalPade(unittest.TestCase):

    def test_A(self):
        x = eval_pade([1., -2.], [0., 3., 4.], -2.)
        self.assertAlmostEqual(x, 0.5)


class TestFindPolesWithPade(unittest.TestCase):

    def test_exec(self):
        poles, zeros = find_poles_with_pade(np.ones(10), 3., transforms=[IdentityTransform(), EulerTransform(-1.)], tol=1e-15)
        # print(poles, zeros)


if __name__ == '__main__':
    unittest.main()
