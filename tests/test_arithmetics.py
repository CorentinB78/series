import unittest
import numpy as np
from series.arithmetics import rescale_series
from series.arithmetics import prod_series
from series.arithmetics import one_over_series
from series.arithmetics import divide_series
from series.arithmetics import compose_series
from series.arithmetics import reverse_series


class TestRescaleSeries(unittest.TestCase):
    def test_A(self):
        a = np.ones(5, dtype=complex)
        U = np.array([2.0, 3.0j], dtype=complex)
        res = np.array(
            [[1.0, 1.0], [2.0, 3.0j], [4.0, -9.0], [8.0, -27.0j], [16.0, 81.0]],
            dtype=complex,
        )
        assert np.isclose(rescale_series(a, U, axis=0), res).all()
        assert np.array_equal(a, np.ones(5))

    def test_B(self):
        U = np.linspace(-1, 4, 3)
        a = np.random.randn(5, 3)
        a_copy = a.copy()
        a_U = rescale_series(a, U, axis=0)
        assert np.array_equal(a, a_copy)
        a_bis = rescale_series(a_U, 1.0 / U, axis=0)
        assert np.isclose(a, a_bis).all()

    def test_C(self):
        U = np.random.randn(3)
        V = np.random.randn(3) * 1.0j + 2.0
        a = np.random.randn(5, 3)
        a_U = rescale_series(a, U, axis=0)
        a_UV = rescale_series(a_U, V, axis=0)
        a_UV_once = rescale_series(a, U * V, axis=0)
        assert np.isclose(a_UV, a_UV_once).all()

    def test_D(self):
        a = [[1.0, 2.0, 4.0], [-0.5, 0.5, 0.25]]
        b = rescale_series(a, 2.0, axis=1)
        assert np.array_equal(b, np.array([[1.0, 4.0, 16.0], [-0.5, 1.0, 1.0]]))


class TestProdSeries(unittest.TestCase):
    def test_A(self):
        a = np.ones(10)  # a = 1/(1 - X)
        b = np.zeros(7)
        b[:2] = [1.0, -1.0]  # b = 1 - X
        res = np.zeros(7)
        res[0] = 1.0
        np.testing.assert_allclose(prod_series(a, b), res)

    def test_B(self):
        a = [2.0, -3.0, 5.0]
        b = [1.0j, 2.0j, 3.0j]
        assert prod_series(a, b).dtype == complex

    def test_C(self):
        a = [2, -3, 5]
        b = [1.0, 2.0, 3.0]
        assert prod_series(a, b).dtype == float

    def test_D(self):
        c = np.arange(10)
        a = np.array([1.0, 2.0])[:, None] * c
        b = a.copy()
        assert np.array_equal(
            prod_series(a, b, axis1=-1, axis2=-1),
            np.array([prod_series(c, c), 4.0 * prod_series(c, c)]),
        )


class TestOneOverSeries(unittest.TestCase):
    def test_A(self):
        # TODO: remove random
        a = np.random.randn(10)
        b = one_over_series(a)
        c = one_over_series(b)
        np.testing.assert_allclose(a, c, atol=1e-3)

    def test_B(self):
        a = np.ones(10)  # a = 1/(1 - X)
        res = np.zeros(10)
        res[:2] = [1.0, -1.0]  # res = 1 - X
        np.testing.assert_allclose(one_over_series(a), res)
        assert np.array_equal(a, np.ones(10))  # initial series hasn't changed

    def test_C(self):
        a = np.array(
            [
                0.74932355,
                0.55984366,
                1.67792072,
                0.00910207,
                -0.93652222,
                0.07237222,
                -1.10551458,
                1.00120716,
                0.51693182,
                -0.68287985,
            ]
        )
        b_ref = np.array(
            [
                1.334537,
                -0.9970754,
                -2.24341133,
                3.89261225,
                3.79530453,
                -12.89992674,
                0.35338191,
                30.40358168,
                -21.88083204,
                -57.58221028,
            ]
        )
        b = one_over_series(a)
        assert np.allclose(b_ref, b)


class TestDivideSeries(unittest.TestCase):
    def test_A(self):
        a = np.zeros(10)
        a[0] = 1.0  # a = 1
        b = np.zeros(12)
        b[:2] = [-1.0, 1.0]  # b = X - 1
        res = -np.ones(10)
        np.testing.assert_allclose(divide_series(a, b), res)

    def test_B(self):
        a = np.array([0.0, 2.0, -3.0j, 0.0, 0.0], dtype=complex)
        b = np.array([1.0, 0.0, 0.0, -4.0, 0.0], dtype=complex)
        res = np.array([0.0, 2.0, -3.0j, 0.0, 8.0], dtype=complex)
        np.testing.assert_allclose(divide_series(a, b), res)
        assert divide_series(a, b)[0] == 0.0

    def test_C(self):
        a = np.random.randn(10)
        b = np.random.randn(5)
        ab = prod_series(a, b)
        np.testing.assert_allclose(divide_series(ab, a), b)
        np.testing.assert_allclose(divide_series(ab, b), a[:5])

    def test_D(self):
        a = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 0.0], [0.0, 2.0, -3.0j, 0.0, 0.0]], dtype=complex
        )
        b = np.array(
            [[1.0, 0.0, 0.0, -4.0, 0.0], [1.0, 0.0, 0.0, -4.0, 0.0]], dtype=complex
        )
        res = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 8.0], [0.0, 2.0, -3.0j, 0.0, 8.0]], dtype=complex
        )
        np.testing.assert_allclose(divide_series(a, b, axis1=1, axis2=1), res)


class TestComposeSeries(unittest.TestCase):
    def test_A(self):
        a = np.array([1.0, 1, 0])  # a = 1 + X
        b = np.array([0.0, 0, 1])  # b = X^2
        a_of_b = np.array([1.0, 0, 1])  # a_of_b = 1 + X^2
        np.testing.assert_allclose(compose_series(a, b), a_of_b)

    def test_B(self):
        a = np.ones(10)
        a[0] = 0.0  # a = X / (1 - X)
        rev_a = -((-1.0) ** np.arange(15))
        rev_a[0] = 0.0  # rev_a = X / (1 + X)
        res = np.zeros(10)
        res[1] = 1.0  # res = X
        np.testing.assert_allclose(compose_series(a, rev_a), res)


class TestReverseSeries(unittest.TestCase):
    def test_A(self):
        a = np.ones(10)
        a[0] = 0.0  # a = X / (1 - X)
        rev_a = -((-1.0) ** np.arange(10))
        rev_a[0] = 0.0  # rev_a = X / (1 + X)
        np.testing.assert_allclose(reverse_series(a), rev_a)
        res = np.zeros(10)
        res[1] = 1.0  # res = X
        np.testing.assert_allclose(compose_series(a, rev_a), res)

    def test_B(self):
        a = np.ones(10)  # a = 1 + X / (1 - X)
        rev_a = -((-1.0) ** np.arange(10))
        rev_a[
            0
        ] = 0.0  # rev_a = X / (1 + X) (same as previous, constant coeff doesnt matter)
        np.testing.assert_allclose(reverse_series(a), rev_a)


if __name__ == "__main__":
    unittest.main()
