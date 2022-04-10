import unittest
import numpy as np
import xarray as xr
from series import arithmetics
from series.xarray_wrappers import prod_series
from series.xarray_wrappers import one_over_series
from series.xarray_wrappers import divide_series
from series.xarray_wrappers import error_sum_series

# TODO: write rescale_series for xr
# class TestRescaleSeries(unittest.TestCase):

#     def test_A(self):
#         a = np.ones(5, dtype=complex)
#         U = np.array([2., 3.j], dtype=complex)
#         res = np.array([[1., 1.], [2., 3.j], [4., -9.], [8., -27.j], [16., 81.]], dtype=complex)
#         assert np.isclose(rescale_series(a, U), res).all()
#         assert np.array_equal(a, np.ones(5))

#     def test_B(self):
#         U = np.linspace(-1, 4, 3)
#         a = np.random.randn(5, 3)
#         a_copy = a.copy()
#         a_U = rescale_series(a, U)
#         assert np.array_equal(a, a_copy)
#         a_bis = rescale_series(a_U, 1. / U)
#         assert np.isclose(a, a_bis).all()

#     def test_C(self):
#         U = np.random.randn(3)
#         V = np.random.randn(3) * 1.j + 2.
#         a = np.random.randn(5, 3)
#         a_U = rescale_series(a, U)
#         a_UV = rescale_series(a_U, V)
#         a_UV_once = rescale_series(a, U*V)
#         assert np.isclose(a_UV, a_UV_once).all()

#     def test_D(self):
#         a = [[1., 2., 4.], [-0.5, 0.5, 0.25]]
#         b = rescale_series(a, 2., axis=1)
#         assert np.array_equal(b, np.array([[1., 4., 16.], [-0.5, 1., 1.]]))


class TestProdSeries(unittest.TestCase):
    def test_A(self):
        a = np.ones(10)  # a = 1/(1 - X)
        b = np.zeros(7)
        b[:2] = [1.0, -1.0]  # b = 1 - X
        res = np.zeros(7)
        res[0] = 1.0
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
        b = xr.DataArray(b, {"order": range(len(b))}, ["order"])
        res = xr.DataArray(res, {"order": range(len(res))}, ["order"])
        xr.testing.assert_allclose(prod_series(a, b, "order"), res)

    def test_B(self):
        a = [2.0, -3.0, 5.0]
        b = [1.0j, 2.0j, 3.0j]
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
        b = xr.DataArray(b, {"order": range(len(b))}, ["order"])
        assert prod_series(a, b, "order").dtype == complex

    def test_C(self):
        a = [2, -3, 5]
        b = [1.0, 2.0, 3.0]
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
        b = xr.DataArray(b, {"order": range(len(b))}, ["order"])
        assert prod_series(a, b, "order").dtype == float

    def test_D(self):
        c = np.arange(10)
        a = np.array([1.0, 2.0])[:, None] * c
        a = xr.DataArray(a, {"order": range(len(c))}, ["x", "order"])
        b = a.copy()
        assert np.array_equal(
            prod_series(a, b, dim="order").data,
            np.array(
                [arithmetics.prod_series(c, c), 4.0 * arithmetics.prod_series(c, c)]
            ),
        )


class TestOneOverSeries(unittest.TestCase):
    def test_A(self):
        # TODO: remove random
        a = np.random.randn(10)
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
        b = one_over_series(a, "order")
        c = one_over_series(b, "order")
        xr.testing.assert_allclose(a, c)

    def test_B(self):
        a = np.ones(10)  # a = 1/(1 - X)
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
        res = np.zeros(10)
        res[:2] = [1.0, -1.0]  # res = 1 - X
        res = xr.DataArray(res, {"order": range(len(res))}, ["order"])
        xr.testing.assert_allclose(one_over_series(a, "order"), res)
        assert np.array_equal(a.data, np.ones(10))  # initial series hasn't changed

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
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])
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
        b_ref = xr.DataArray(b_ref, {"order": range(len(b_ref))}, ["order"])
        b = one_over_series(a, "order")
        xr.testing.assert_allclose(b_ref, b)


class TestDivideSeries(unittest.TestCase):
    def test_A(self):
        a = np.zeros(10)
        a[0] = 1.0
        a = xr.DataArray(a, {"order": range(len(a))}, ["order"])

        b = np.zeros(12)
        b[:2] = [-1.0, 1.0]  # b = X - 1
        b = xr.DataArray(b, {"order": range(len(b))}, ["order"])

        res = xr.DataArray(-np.ones(10), {"order": range(10)}, ["order"])
        xr.testing.assert_allclose(divide_series(a, b, "order"), res)

    def test_B(self):
        a = xr.DataArray([0.0, 2.0, -3.0j, 0.0, 0.0], {"order": range(5)}, ["order"])
        b = xr.DataArray([1.0, 0.0, 0.0, -4.0, 0.0], {"order": range(5)}, ["order"])

        res = xr.DataArray([0.0, 2.0, -3.0j, 0.0, 8.0], {"order": range(5)}, ["order"])
        xr.testing.assert_allclose(divide_series(a, b, "order"), res)

    def test_C(self):
        a = xr.DataArray(np.random.randn(10), {"order": range(10)}, ["order"])
        b = xr.DataArray(np.random.randn(5), {"order": range(5)}, ["order"])

        ab = prod_series(a, b, "order")
        xr.testing.assert_allclose(divide_series(ab, a, "order"), b)
        xr.testing.assert_allclose(divide_series(ab, b, "order"), a[:5])

    def test_D(self):
        a = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 0.0], [0.0, 2.0, -3.0j, 0.0, 0.0]], dtype=complex
        )
        a = xr.DataArray(a, {"order": range(5), "x": ["a", "b"]}, ["x", "order"])
        b = np.array(
            [[1.0, 0.0, 0.0, -4.0, 0.0], [1.0, 0.0, 0.0, -4.0, 0.0]], dtype=complex
        )
        b = xr.DataArray(b, {"order": range(5), "x": ["a", "b"]}, ["x", "order"])
        res = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 8.0], [0.0, 2.0, -3.0j, 0.0, 8.0]], dtype=complex
        )
        res = xr.DataArray(res, {"order": range(5), "x": ["a", "b"]}, ["x", "order"])
        xr.testing.assert_allclose(divide_series(a, b, "order"), res)

    def test_E(self):
        a = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 0.0], [0.0, 2.0, -3.0j, 0.0, 0.0]], dtype=complex
        )
        a = xr.DataArray(a, {"order": range(5), "x": ["a", "b"]}, ["x", "order"])
        b = np.array([1.0, 0.0, 0.0, -4.0, 0.0], dtype=complex)
        b = xr.DataArray(b, {"order": range(5)}, ["order"])
        res = np.array(
            [[0.0, 2.0, -3.0j, 0.0, 8.0], [0.0, 2.0, -3.0j, 0.0, 8.0]], dtype=complex
        )
        res = xr.DataArray(res, {"order": range(5), "x": ["a", "b"]}, ["x", "order"])
        xr.testing.assert_allclose(divide_series(a, b, "order"), res)


# TODO: finish transposing tests
# class TestComposeSeries(unittest.TestCase):

#     def test_A(self):
#         a = np.array([1., 1, 0]) # a = 1 + X
#         b = np.array([0., 0, 1]) # b = X^2
#         a_of_b = np.array([1., 0, 1]) # a_of_b = 1 + X^2
#         assert np.allclose(compose_series(a, b), a_of_b)

#     def test_B(self):
#         a = np.ones(10)
#         a[0] = 0. # a = X / (1 - X)
#         rev_a = - (-1.) ** np.arange(15)
#         rev_a[0] = 0. # rev_a = X / (1 + X)
#         res = np.zeros(10)
#         res[1] = 1. # res = X
#         assert np.allclose(compose_series(a, rev_a), res)


# class TestReverseSeries(unittest.TestCase):

#     def test_A(self):
#         a = np.ones(10)
#         a[0] = 0. # a = X / (1 - X)
#         rev_a = - (-1.) ** np.arange(10)
#         rev_a[0] = 0. # rev_a = X / (1 + X)
#         assert np.allclose(reverse_series(a), rev_a)
#         res = np.zeros(10)
#         res[1] = 1. # res = X
#         assert np.allclose(compose_series(a, rev_a), res)

#     def test_B(self):
#         a = np.ones(10) # a = 1 + X / (1 - X)
#         rev_a = - (-1.) ** np.arange(10)
#         rev_a[0] = 0. # rev_a = X / (1 + X) (same as previous, constant coeff doesnt matter)
#         assert np.allclose(reverse_series(a), rev_a)


class TestErrorSumSeries(unittest.TestCase):
    def test_B(self):
        a = xr.DataArray(np.ones(5), dims=["order"])
        U = 0.3
        error = error_sum_series(a, U, start_geom=0, dim="order", verbose=False)
        self.assertEqual(error, 0.3**5 / (1.0 - 0.3))

    def test_D(self):
        a = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.25, 0.125]]
        a = xr.DataArray(a, dims=["x", "order"])
        U = xr.DataArray([0.1, 0.2, 0.3], dims=["U"])
        ref = [
            [0.1**4 / (1.0 - 0.1), 0.2**4 / (1.0 - 0.2), 0.3**4 / (1.0 - 0.3)],
            [
                (0.1 / 2.0) ** 4 / (1.0 - 0.1 / 2.0),
                (0.2 / 2.0) ** 4 / (1.0 - 0.2 / 2.0),
                (0.3 / 2.0) ** 4 / (1.0 - 0.3 / 2.0),
            ],
        ]
        ref = xr.DataArray(ref, dims=["x", "U"])
        error = error_sum_series(a, U, start_geom=0, dim="order", verbose=False)
        xr.testing.assert_equal(error, ref)


if __name__ == "__main__":
    unittest.main()
