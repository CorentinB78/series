import unittest
import numpy as np
from series.resummation import (
    resum_series,
    Rconv_1d,
    Rconv_robust_1d,
    error_sum_series,
    EulerTransform,
    ParabolaTransform,
    RealSeries,
    sum_series,
    compose_series,
    rescale_series,
)


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


class TestSumSeries(unittest.TestCase):
    def test_unidim(self):
        series = np.array([2, 5, 3])
        val = sum_series(series, -2)
        assert val == 2 + 5 * (-2) + 3 * (-2) ** 2
        assert np.array_equal(series, np.array([2, 5, 3]))

    def test_multidim_1(self):
        series = np.array([[2, 3], [5, -7], [3, 4]])
        val = sum_series(series, -2, axis=1)
        assert val.shape == (3,)
        assert val[0] == 2 - 3 * 2
        assert val[1] == 5 + 2 * 7
        assert val[2] == 3 - 2 * 4
        assert np.array_equal(series, np.array([[2, 3], [5, -7], [3, 4]]))

    def test_multidim_2(self):
        series = np.array([[2, 3], [5, -7], [3, 4]])
        val = sum_series(series, np.array([-2, 9]))
        assert val.shape == (2,)
        assert val[0] == 2 + 5 * (-2) + 3 * (-2) ** 2
        assert val[1] == 3 - 7 * 9 + 4 * 9**2
        assert np.array_equal(series, np.array([[2, 3], [5, -7], [3, 4]]))


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

    def test_E(self):
        A = 3.6j
        N = 9  # max order
        series = A * rescale_series(np.ones(N + 1), 0.5)  # pole in 2.0
        U = 0.5
        assert error_sum_series(series, U, start_geom=5) == (
            np.abs(A * (U / 2.0) ** (N + 1)) / np.abs(1.0 - np.abs(U / 2.0))
        )


class TestTransforms(unittest.TestCase):
    def test_Euler(self):
        phi = EulerTransform(3.0 + 1.0j)
        assert phi(2.0) == 2.0 / (2.0 - (3.0 + 1.0j))
        assert phi.rev(phi(2.0)) == 2.0
        delta = np.abs(sum_series(phi.series(50), 0.1) - phi(0.1))
        assert delta < 1e-15, delta
        assert np.allclose(
            compose_series(phi.series(10), phi.rev_series(10)),
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )

    def test_parabola_1(self):
        phi = ParabolaTransform(-1.0)

        assert np.isclose(phi(1.0), 0.4300660362066224)
        assert np.isclose(phi(1.0j), 0.22754989416298005 + 0.5362002977975425j)
        assert np.isclose(phi(-1.0), -1.0), "{} != -1".format(phi(-1.0))
        assert np.isclose(phi.rev(-1.0), -1.0), "{} != -1".format(phi.rev(-1.0))
        assert np.isclose(phi.rev(phi(1.0)), 1.0)

        assert np.isclose(sum_series(phi.rev_series(10), 0.1j), phi.rev(0.1j))
        rev_series_ref = np.array(
            [
                (-0 + 0j),
                (1.6211389382774044 + 0j),
                (1.0807592921849363 + 0j),
                (0.8285821240084511 + 0j),
                (0.6793344122305313 + 0j),
                (0.579492839523923 + 0j),
                (0.5074734107892904 + 0j),
                (0.4527919228554073 + 0j),
                (0.40970242365079307 + 0j),
                (0.37477561146918464 + 0j),
                (0.3458303605237262 + 0j),
            ],
            dtype=complex,
        )
        assert np.isclose(phi.rev_series(10), rev_series_ref).all()

        assert np.isclose(sum_series(phi.series(10), 0.1j), phi(0.1j))
        series_ref = np.array(
            [
                0j,
                (0.6168502750680849 + 0j),
                (-0.253669507901048 + 0j),
                (0.08866979324425008 + 0j),
                (-0.028497068520022227 + 0j),
                (0.008707305639892427 + 0j),
                (-0.0025726087167125752 + 0j),
                (0.000742098530249626 + 0j),
                (-0.00021026124589426672 + 0j),
                (5.874946562974001e-05 + 0j),
                (-1.623340497248011e-05 + 0j),
            ],
            dtype=complex,
        )
        assert np.isclose(phi.series(10), series_ref).all()

    def test_parabola_2(self):
        phi = ParabolaTransform(2.0j)

        assert np.isclose(phi(-2.0j), 0.4300660362066224)
        assert np.isclose(phi(2.0), 0.22754989416298005 + 0.5362002977975425j)
        assert np.isclose(phi.rev(phi(1.0)), 1.0)

        assert np.isclose(sum_series(phi.rev_series(10), 0.1j), phi.rev(0.1j))
        rev_series_ref = np.array(
            [
                0j,
                (-3.242277876554809j),
                (-2.1615185843698725j),
                (-1.6571642480169022j),
                (-1.3586688244610625j),
                (-1.158985679047846j),
                (-1.0149468215785808j),
                (-0.9055838457108146j),
                (-0.8194048473015861j),
                (-0.7495512229383693j),
                (-0.6916607210474524j),
            ],
            dtype=complex,
        )
        assert np.isclose(phi.rev_series(10), rev_series_ref).all()

        assert np.isclose(sum_series(phi.series(10), 0.1j), phi(0.1j))
        series_ref = np.array(
            [
                0j,
                (+0.30842513753404244j),
                (0.063417376975262),
                (-0.01108372415553126j),
                (-0.0017810667825013892),
                (+0.00027210330124663836j),
                (4.019701119863399e-05),
                (-5.797644767575203e-06j),
                (-8.213329917744794e-07),
                (+1.1474505005808595e-07j),
                (1.5852934543437607e-08),
            ],
            dtype=complex,
        )
        assert np.isclose(phi.series(10), series_ref).all()


class TestRealSeries(unittest.TestCase):
    def test_A(self):

        ### test RealSeries
        series = RealSeries(
            [
                -0.06494468,
                0.00451195,
                0.01216893,
                0.01400438,
                0.24504612,
                -0.11547953,
                -0.44090799,
                -0.20236113,
                -0.92070055,
            ],
            [
                5.98923777e-08,
                1.68121180e-07,
                1.73906957e-06,
                2.71475618e-05,
                2.75597361e-04,
                2.04736675e-03,
                4.07836446e-02,
                1.01054319e-02,
                8.34967472e-02,
            ],
        )
        z = 0.01

        ref, _, _ = series.sum(z)
        print(ref)
        atol = 1e-10

        assert (
            np.abs(series.conformal_transform(ParabolaTransform(-0.25)).sum(z)[0] - ref)
            < atol
        )
        assert np.abs(series.one_over_series().sum(z)[0] - ref) < atol
        assert np.abs(series.eliminate_order0().sum(z)[0] - ref) < atol
        assert np.abs(series.eliminate_pole().sum(z)[0] - ref) < atol
        assert np.abs(series.shift(10.0).sum(z)[0] - ref) < atol

        assert (
            np.abs(
                series.conformal_transform(ParabolaTransform(-0.25))
                .conformal_transform(EulerTransform(-0.25))
                .sum(z)[0]
                - ref
            )
            < atol
        )

        assert np.abs(series.one_over_series().eliminate_pole().sum(z)[0] - ref) < atol

        assert (
            np.abs(
                series.conformal_transform(ParabolaTransform(-0.25))
                .one_over_series()
                .conformal_transform(EulerTransform(-0.25))
                .eliminate_pole()
                .sum(z)[0]
                - ref
            )
            < atol
        )


if __name__ == "__main__":
    unittest.main()
