import unittest
import numpy as np
from series.resummation import resum_series
from series.resummation import Rconv_1d
from series.resummation import Rconv_robust_1d
from series.resummation import error_sum_series
from series.resummation import EulerTransform

# from series.resummation import sum_series


class TestRconv1D(unittest.TestCase):
    def test_A(self):
        a = np.ones(10)
        Rc, err, r_val = Rconv_1d(a)
        self.assertAlmostEqual(Rc, 1.0)

    def test_B(self):
        a = 3.0 * np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        Rc, err, r_val = Rconv_1d(a)
        self.assertAlmostEqual(Rc, 0.5)

    def test_C(self):
        a = 3.0 * np.array([1.0, 0.0, 4.0, 0.0, 16.0])
        Rc, err, r_val = Rconv_1d(a)
        self.assertAlmostEqual(Rc, 0.5)


class TestRconv1DRobust(unittest.TestCase):
    def test_A(self):
        a = np.ones(10)
        Rc, lo, up = Rconv_robust_1d(a)
        self.assertAlmostEqual(Rc, 1.0)

    def test_B(self):
        a = 3.0 * np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        Rc, lo, up = Rconv_robust_1d(a)
        self.assertAlmostEqual(Rc, 0.5)

    def test_C(self):
        a = 3.0 * np.array([1.0, 0.0, 4.0, 0.0, 16.0])
        Rc, lo, up = Rconv_robust_1d(a)
        self.assertAlmostEqual(Rc, 0.5)


class TestResumSeries(unittest.TestCase):
    def test_A(self):
        a = (-1.0) ** np.arange(10)  # a = 1 / (X + 1)
        phi = EulerTransform(-1.0)
        a_resum = resum_series(a, phi)
        res = np.zeros(10)
        res[:2] = [1.0, -1.0]
        np.testing.assert_allclose(a_resum, res)


class TestErrorSumSeries(unittest.TestCase):
    def test_A(self):
        a = np.arange(10)
        U = [0.2, 0.2j, -0.3]
        error = error_sum_series(a, U, start_geom=2, verbose=False)
        # print(error)
        self.assertEqual(error.shape, (3,))

    def test_B(self):
        a = np.ones(5)
        U = 0.3
        error = error_sum_series(a, U, start_geom=0, verbose=False)
        self.assertEqual(error, 0.3**5 / (1.0 - 0.3))

    def test_C(self):
        a = 2.0 ** np.arange(5)
        U = 0.3
        error = error_sum_series(a, U, start_geom=0, Rc=0.5, verbose=False)
        self.assertEqual(error, (0.3 / 0.5) ** 5 / (1.0 - (0.3 / 0.5)))

    def test_D(self):
        a = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.25, 0.125]]
        a = np.array(a)
        U = np.array([0.1, 0.2, 0.3])
        ref = [
            [0.1**4 / (1.0 - 0.1), 0.2**4 / (1.0 - 0.2), 0.3**4 / (1.0 - 0.3)],
            [
                (0.1 / 2.0) ** 4 / (1.0 - 0.1 / 2.0),
                (0.2 / 2.0) ** 4 / (1.0 - 0.2 / 2.0),
                (0.3 / 2.0) ** 4 / (1.0 - 0.3 / 2.0),
            ],
        ]
        error = error_sum_series(a[..., None], U, start_geom=0, axis=1, verbose=False)
        np.testing.assert_equal(error, ref)


if __name__ == "__main__":
    unittest.main()
