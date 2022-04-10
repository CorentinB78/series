"""
Resummation of series.
"""
from __future__ import division
from __future__ import print_function
from .arithmetics import (
    rescale_series,
    prod_series,
    compose_series,
    one_over_series,
    reverse_series,
    _power_gen,
)
from .creation import poly_to_series, power_law_series, arctan_series
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from functools import reduce
from scipy import stats
import numpy.ma as ma
from scipy.stats import mstats

# import warnings


# TODO: cleanup

# ########## Polynomials manipulation ##########

# def prod_poly(poly1, poly2, dtype=complex):
#     """
#     Product of two polynomials.
#     """
#     d1, d2 = len(poly1)-1, len(poly2)-1
#     output = np.zeros(d1 + d2 + 1, dtype=dtype)

#     for n in range(len(output)):
#         for k in range(max(0, n-d2), min(n, d1)+1):
#             output[n] += poly1[k] * poly2[n - k]
#     return output

# def compose_poly(poly1, poly2, dtype=complex):
#     """
#     Composition of two polynomials.
#     """
#     d1 = len(poly1)-1
#     output = [poly1[-1]]

#     for j in range(d1-1, -1, -1):
#         output = prod_poly(poly2, output, dtype=dtype)
#         output[0] += poly1[j]

#     return output

########## Transformations ##########


def resum_series(series, conformal_trans, axis=0):
    return compose_series(
        series, conformal_trans.rev_series(series.shape[axis] - 1), axis, 0
    )


class IdentityTransform(object):
    def __init__(self):
        pass

    def __call__(self, z):
        return np.complex128(z)

    def rev(self, w):
        return np.complex128(w)

    def series(self, N):
        output = np.zeros(N + 1, dtype=complex)
        output[1] = 1.0
        return output

    def rev_series(self, N):
        output = np.zeros(N + 1, dtype=complex)
        output[1] = 1.0
        return output

    def __repr__(self):
        return "Identity Transform"


class NaNTransform(object):
    def __init__(self):
        pass

    def __call__(self, z):
        return z * np.nan

    def rev(self, w):
        return w * np.nan

    def series(self, N):
        output = np.zeros(N + 1)
        output[1:] = np.nan
        return output

    def rev_series(self, N):
        output = np.zeros(N + 1)
        output[1:] = np.nan
        return output


class EulerTransform(object):
    """
    Euler transform \phi(U) = U / (U - p)
    """

    def __init__(self, pole):
        """
        `pole` is a complex number or a 1D array. It has to be non zero.
        """
        self.pole = np.complex128(pole)

        if np.any(self.pole == 0.0):
            raise ValueError("The transformation pole cannot be zero.")

    def __call__(self, z):
        return old_div(z, (z - self.pole))

    def rev(self, w):
        return old_div(self.pole * w, (w - 1.0))

    def series(self, N):
        """
        Returns a (N+1)D array or a (N+1, M)D array where M is the number of poles.
        """
        try:
            output = -((old_div(1, self.pole)) ** np.arange(0, N + 1))
        except ValueError:
            output = -((old_div(1, self.pole[None, :])) ** np.arange(0, N + 1)[:, None])

        output[0] = 0.0
        return output

    def rev_series(self, N):
        """
        Returns a (N+1)D array or a (N+1, M)D array where M is the number of poles.
        """
        try:
            output = -self.pole * np.ones(N + 1)
        except ValueError:
            output = -self.pole[None, :] * np.ones(N + 1)[:, None]

        output[0] = 0.0
        return output

    def __repr__(self):
        try:
            return "{} Euler transforms (vect)".format(len(self.pole))
        except:
            return "Euler Transform (p=%.1f+%.1fj)" % (self.pole.real, self.pole.imag)


class NormalEulerTransform(object):
    """
    Euler transform \phi(U) = p U / (p - U)
    So that near U=0 we have \phi(U) \approx U
    """

    def __init__(self, pole):
        """
        `pole` is a complex number. It has to be non zero.
        """
        self.pole = np.complex128(pole)

    def __call__(self, z):
        return old_div(self.pole * z, (self.pole - z))

    def rev(self, w):
        return old_div(self.pole * w, (self.pole + w))

    def series(self, N):
        """
        Returns a 1D array of size N+1
        """
        output = (old_div(1, self.pole)) ** np.arange(-1, N)

        output[0] = 0.0
        return output

    def rev_series(self, N):
        """
        Returns a 1D array of size N+1
        """
        output = (old_div(-1, self.pole)) ** np.arange(-1, N)

        output[0] = 0.0
        return output

    def __repr__(self):
        return "Normal Euler Transform (p=%.1f+%.1fj)" % (
            self.pole.real,
            self.pole.imag,
        )

    def diff_norm(self, radius):
        radius = np.abs(radius)
        if radius >= np.abs(self.pole):
            return np.inf
        else:
            return old_div(radius * radius, (np.abs(self.pole) - radius))

    def compose_with_normal_Euler(self, phi):
        if self.pole + phi.pole == 0.0:
            raise ValueError("Cannot compose two normal Euler woth opposite poles.")

        return NormalEulerTransform(
            old_div(self.pole * phi.pole, (self.pole + phi.pole))
        )


class MultiPolesTransform(object):
    def __new__(cls, poles):
        n = np.sum(~np.isnan(poles))
        if n == 0:
            return IdentityTransform()
        elif n == 1:
            return EulerTransform(poles[0])
        else:
            return super(MultiPolesTransform, cls).__new__(cls)

    def __init__(self, poles):
        self.poles = np.complex128(poles)
        self.poles = self.poles[~np.isnan(self.poles)]

        if np.prod(self.poles) == 0:
            raise ValueError("The transformation poles cannot be zero.")

        self.expon = 1 / float(len(poles))
        self.norm = np.prod((-self.poles) ** self.expon)

        self.rev_poly = np.zeros(len(self.poles) + 1, dtype=complex)
        self.rev_poly[0] = 1.0
        for p in self.poles:
            tmp_poly = np.zeros_like(self.rev_poly)
            tmp_poly[:2] = [-p, 1.0]
            self.rev_poly = prod_series(self.rev_poly, tmp_poly)

    def __call__(self, z):
        z = np.complex128(z)
        if isinstance(z, np.ndarray):
            output = old_div(
                old_div(z, self.norm),
                np.prod((1 - old_div(z[..., None], self.poles)) ** self.expon, axis=-1),
            )
            output[np.isinf(z)] = 1.0
            return output
        else:
            if np.isinf(z):
                return 1.0
            else:
                return old_div(
                    old_div(z, self.norm),
                    np.prod((1 - old_div(z, self.poles)) ** self.expon, axis=-1),
                )

    def rev(self, w, rtol=1e-05, atol=1e-08):
        poly = self.rev_poly.copy()
        poly *= w ** len(self.poles)
        poly[-1] -= 1.0

        candidates = np.roots(poly[::-1])
        # if w == 1.:
        #     candidates = np.append(candidates, np.inf)
        accepted = np.abs(self(candidates) - w) <= atol + rtol * np.abs(w)
        candidates = candidates[accepted]
        if len(candidates) == 0:
            return np.nan  # no possible value
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return candidates.tolist()

    def series(self, N):
        s = power_law_series(-self.expon, N - 1)

        series = rescale_series(s, old_div(-1, self.poles[0]))
        for k in range(1, len(self.poles)):
            series = prod_series(series, rescale_series(s, old_div(-1, self.poles[k])))

        return old_div(np.append([0.0], series), self.norm)

    def rev_series(self, N):
        return reverse_series(self.series(N))

    def __repr__(self):
        return "%d Poles Transform" % len(self.poles)


class TriangleTransform(object):
    def __init__(self, a, l, direction=0.0):
        self.a = float(a)
        self.l = float(l)
        self.dir = float(direction)

    def __call__(self, z):
        z = z * np.exp(-1.0j * self.dir)
        x = 1.0j * (1.0 + self.l * z) ** (old_div(1, self.a))
        return old_div(-(1.0j - x), (1.0j + x))

    def rev(self, w):
        x = old_div(1.0j * (1.0 + w), (-w + 1.0))
        return old_div(np.exp(1.0j * self.dir) * ((-1.0j * x) ** self.a - 1.0), self.l)

    def series(self, N):
        return reverse_series(self.rev_series(N))

    def rev_series(self, N):
        output = power_law_series(self.a, N + 1)
        geom_series = 2.0 * prod_series(
            rescale_series(power_law_series(-1.0, N + 1), -1.0),
            poly_to_series([0, 1], N + 1),
        )
        output = compose_series(output, geom_series)
        output[0] -= 1.0
        output *= old_div(np.exp(1.0j * self.dir), self.l)
        return output

    def __repr__(self):
        return "Triangle Transform (a=%.1fpi, l=%.1f, dir=%.1fpi)" % (
            old_div(self.a, np.pi),
            self.l,
            old_div(self.dir, np.pi),
        )


class ParabolaTransform(object):
    ### TODO: vectorize

    def __init__(self, p):
        """
        `p` is the bottom of the parabola
        """
        self.p = complex(p)

    def __call__(self, z):
        return -np.tan(np.pi / 4.0 * np.sqrt(old_div((z + 0.0j), self.p))) ** 2

    def rev(self, w):
        return self.p * (np.arctan(np.sqrt(-w + 0.0j)) * 4.0 / np.pi) ** 2

    def series(self, N):
        return reverse_series(self.rev_series(N))

    def rev_series(self, N):
        output = arctan_series(2 * N)
        output = old_div(16.0 * self.p, np.pi**2) * prod_series(output, output)
        output = output[::2]
        output = rescale_series(output, -1.0)
        return output

    def __repr__(self):
        return "Parabola Transform (p=%.1f+%.1fj)" % (self.p.real, self.p.imag)

    def diff_norm(self, radius):
        """To check, at least it is a lower bound"""
        radius = np.abs(radius)
        if radius >= np.abs(self.p):
            return np.inf
        else:
            return np.abs(
                (old_div(4, np.pi)) ** 2
                * np.abs(self.p)
                * self(old_div(radius * self.p, np.abs(self.p)))
                + radius
            )


class ParaboleTransform(object):
    """Deprecated"""

    def __init__(self, p, direction=0.0):
        self.p = float(p)
        self.dir = float(direction)

    def __call__(self, z):
        z = z * np.exp(-1.0j * self.dir)
        return -np.tan(np.pi / 4.0 * np.sqrt(old_div(-(z + 0.0j), self.p))) ** 2

    def rev(self, w):
        return (
            -np.exp(1.0j * self.dir)
            * self.p
            * (np.arctan(np.sqrt(-w + 0.0j)) * 4.0 / np.pi) ** 2
        )

    def series(self, N):
        return reverse_series(self.rev_series(N))

    def rev_series(self, N):
        output = arctan_series(2 * N)
        output = old_div(-16.0 * self.p, np.pi**2) * prod_series(output, output)
        output = output[::2]
        output = np.exp(1.0j * self.dir) * rescale_series(output, -1.0)
        return output

    def __repr__(self):
        return "Parabole Transform (p=%.1f, dir=%.1fpi)" % (
            self.p,
            old_div(self.dir, np.pi),
        )


class PtDTransform(object):
    r"""
    Plane to Disk transform
    \varphi(U) = 1 - \frac{2p}{U} \left(1 - \sqrt{1 - \frac{U}{p}} \right)

    From https://arxiv.org/abs/1712.10001
    """

    def __init__(self, pole):
        """
        `pole` is a complex number or a 1D array. It has to be non zero.
        """
        self.pole = np.complex128(pole)

        if np.any(self.pole == 0.0):
            raise ValueError("The transformation pole cannot be zero.")

    def __call__(self, z):
        x = old_div(np.asarray(z), self.pole)

        try:
            if x == 0.0:
                return 0.0
            else:
                return 1.0 - 2.0 / x * (1 - np.sqrt(1 - x))

        except ValueError:  # x is an array
            output = 1.0 - 2.0 / x * (1 - np.sqrt(1 - x))
            output[x == 0.0] = 0.0
            return output

    def rev(self, w):
        return old_div(-4.0 * self.pole * w, (w - 1.0) ** 2)

    def series(self, N):
        """
        Returns a (N+1)D array or a (N+1, M)D array where M is the number of poles.
        """
        return reverse_series(self.rev_series(N))

    def rev_series(self, N):
        """
        Returns a (N+1)D array or a (N+1, M)D array where M is the number of poles.
        """
        try:
            output = -4 * self.pole * np.arange(0, N + 1)
        except ValueError:
            output = -4 * self.pole[None, :] * np.arange(0, N + 1)[:, None]

        output[0] = 0.0
        return output


########## Convergence radius ##########


def _prepare_rconv_series(series, x=None):
    y = np.squeeze(np.asarray(series))
    if y.ndim != 1:
        raise ValueError("`series` should be a 1D array-like")
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.squeeze(np.asarray(x))
    mask = y == 0.0
    if np.sum(~mask) < 2:
        raise ValueError("Needs at least two finite values to compute a slope!")
    x = ma.masked_array(x, mask=mask)
    y = ma.masked_array(y, mask=mask)
    y = ma.log(ma.abs(y))
    return x, y


def Rconv_1d(series, x=None):
    x, y = _prepare_rconv_series(series, x=x)
    slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(x, y)

    rconv = np.exp(-slope)
    return rconv, std_err * rconv, r_value


def Rconv_robust_1d(series, x=None, alpha=0.7, interc=False):
    x, y = _prepare_rconv_series(series, x=x)
    slope, intercept, lo_bound, up_bound = mstats.theilslopes(y, x=x, alpha=alpha)

    if interc:
        return np.exp(intercept), np.exp(-slope)
    else:
        return np.exp(-slope), np.exp(-up_bound), np.exp(-lo_bound)


def Rconv(series):
    if series.ndim == 1:
        series = series[..., None]
    x = np.arange(len(series))
    y = np.log(np.abs(series))
    mask = np.isfinite(y)
    if (np.sum(mask, axis=0) < 2).any():
        print("Problem with Rconv")
        return None, None
    slopes = []
    errs = []
    for k in range(y.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x[mask[:, k]], y[mask[:, k], k]
        )
        slopes.append(slope)
        errs.append(std_err)
    slopes = np.squeeze(np.array(slopes))
    errs = np.squeeze(np.array(errs))

    rconv = np.exp(-slopes)
    return rconv, errs * rconv


def Rconv_robust(series, x=None, alpha=0.7, axis=0):
    """
    Returns (Rc, lower, upper)
    """
    if series.ndim <= 1:
        return Rconv_robust_1d(series, x=x, alpha=alpha)
    series_ = np.moveaxis(series, axis, 0)
    if x is None:
        x = np.arange(len(series_))
    y = np.log(np.abs(series_)).reshape((len(series_), -1))
    slopes = []
    lo_bounds = []
    up_bounds = []
    for k in range(y.shape[1]):
        mask = np.isfinite(y[:, k])
        if np.sum(mask) > 1:
            slope, intercept, lo_bound, up_bound = mstats.theilslopes(
                y[:, k][mask], x=x[mask], alpha=alpha
            )
        else:
            slope, intercept, lo_bound, up_bound = np.nan, np.nan, np.nan, np.nan
        slopes.append(slope)
        lo_bounds.append(lo_bound)
        up_bounds.append(up_bound)
    slopes = np.array(slopes).reshape(series_.shape[1:])
    lo_bounds = np.array(lo_bounds).reshape(series_.shape[1:])
    up_bounds = np.array(up_bounds).reshape(series_.shape[1:])

    return np.exp(-slopes), np.exp(-up_bounds), np.exp(-lo_bounds)


########## Series evaluation ##########


def sum_series(series, U, axis=0):
    """
    `series` ND array whose axis `axis` represents the orders.
    `U` scalar or 1D array

    Returns the sum of the series evaluated at `U`
    s_0 + s_1 U + \ldots + s_N U^N
    Broadcasting rules apply between terms of the series and `U`.
    """

    power_U = _power_gen(U)
    next(power_U)
    series_ = np.moveaxis(series, axis, 0)

    output = reduce(lambda s, x: s + x * next(power_U), series_)

    return output


class _ReduceFunc(object):
    def __init__(self, U, output_array):
        self.k = 0
        self.power_U = _power_gen(U)
        next(self.power_U)
        self.output = output_array

    def __call__(self, s, x):
        if self.k == 0:
            self.output[0, ...] = s
            self.k = 1

        res = s + x * next(self.power_U)
        self.output[self.k, ...] = res
        self.k += 1
        return res


def partial_sum_series(series, U, axis=0):
    series_ = series.copy()
    series_ = np.moveaxis(series_, axis, 0)
    U_ = np.atleast_1d(U)

    output = np.empty(series_.shape + (len(U_),), dtype=series_.dtype)

    series_ = series_[..., None]
    f = _ReduceFunc(U_, output)
    reduce(f, series_)

    output = np.squeeze(np.moveaxis(output, 0, axis))
    return output


# print partial_sum_series(np.ones((5, 10)), np.arange(3), axis=1)
# print sum_series(np.ones((5, 10)), np.arange(3), axis=1)
# print

# print partial_sum_series(np.ones((5, 10)), 4., axis=0)
# print sum_series(np.ones((5, 10)), 4., axis=0)


def error_sum_series(series, U_list, start_geom, Rc=None, axis=-1, verbose=True):
    """
    Returns an upper bound estimate of the truncation error when evaluating `series` at `U`.

    `start_geom` an int, the series is considered geometric after this order
    `Rc` a scalar or array, the convergence radius of the series.
    If not provided, will be computed on the geometric part of the series using robust method.

    Broadcasting rules apply.
    """
    series = np.moveaxis(series, axis, 0)
    U_list = np.abs(np.asarray(U_list))

    if Rc is None:
        Rc = Rconv_robust(series[start_geom:], axis=0)[0]
        if verbose:
            print("Rc computed:", Rc)
    else:
        Rc = np.asarray(Rc)

    series_unit = rescale_series(series, Rc, axis=0)[start_geom:]
    prefact = np.max(np.abs(series_unit), axis=0)
    N = len(series) - 1

    error = prefact * (U_list / Rc) ** (N + 1) / (1.0 - U_list / Rc)

    try:
        error[error < 0.0] = +np.inf
    except TypeError:
        if error < 0.0:
            error = +np.inf

    return error


class RealSeries(object):
    def __init__(self, series, errors, start_geom=None, nb_samples=100):
        self._series_sampled = np.array(series, dtype=float).copy()
        errors = np.array(errors, dtype=float).copy()

        rand_nb = np.random.standard_normal((nb_samples, len(self._series_sampled)))
        self._series_sampled = rand_nb * errors + self._series_sampled

        if start_geom is None:
            start_geom = len(series) // 2
        if start_geom >= len(series) - 2:
            start_geom = len(series) - 3

        self._start_geom = start_geom
        self._phi_right_list = []
        self._phi_left_list = []
        self._phi_left_der_list = []

    @property
    def series(self):
        return np.mean(self._series_sampled, axis=0)

    @property
    def series_error(self):
        return np.std(self._series_sampled, axis=0, ddof=1)

    def plot(self):
        series = self.series
        series_error = self.series_error
        n_arr = np.arange(len(series))

        plt.errorbar(n_arr, np.abs(series), series_error, fmt="-k", zorder=0)
        plt.plot(n_arr, series, "ob")
        plt.plot(n_arr, -series, "or")

        plt.semilogy()
        plt.draw()

    def rconv(self, full_output=False, verbose=False, alpha=0.7):
        rconv, rconv_lower, rconv_upper = self._rconv(alpha=alpha)

        if full_output or verbose:
            output = {
                "rconv": np.median(rconv),
                "rconv_stat_error": np.std(rconv),
                "rconv_lower": np.median(rconv_lower),
                "rconv_upper": np.median(rconv_upper),
            }

            if verbose:
                key_list = ["rconv", "rconv_stat_error", "rconv_lower", "rconv_upper"]
                for key in key_list:
                    print((key, ":\t", output[key]))
                print()
            if full_output:
                return output

        return np.median(rconv)

    def _rconv(self, alpha=0.7):
        return Rconv_robust(
            self._series_sampled[:, self._start_geom :], axis=1, alpha=alpha
        )

    def phi_right(self, z):
        for phi in self._phi_right_list:
            z = phi(z)
        return z

    def phi_left(self, x, z, dx=0.0):
        ### z is fixed and unique (not a vector)
        for (phi, phi_der) in zip(
            self._phi_left_list[::-1], self._phi_left_der_list[::-1]
        ):
            dx = np.abs(phi_der(x, z)) * dx
            x = phi(x, z)
        return x, dx

    def sum(
        self, z, full_output=False, verbose=False, alpha=0.7, use_rconv_lower=False
    ):
        z = self.phi_right(z)

        value_arr_0 = sum_series(self._series_sampled, z, axis=1)

        rconv, rconv_lower, rconv_upper = self._rconv(alpha=alpha)

        if use_rconv_lower:
            rconv = rconv_lower

        trunc_error = error_sum_series(
            self._series_sampled.T, z, self._start_geom, Rc=rconv
        )

        value_arr, trunc_error = self.phi_left(value_arr_0, z, trunc_error)
        trunc_error = np.median(trunc_error)
        stat_error = np.std(value_arr)

        if full_output or verbose:
            trunc_error_lower = error_sum_series(
                self._series_sampled.T, z, self._start_geom, Rc=rconv_upper
            )
            trunc_error_upper = error_sum_series(
                self._series_sampled.T, z, self._start_geom, Rc=rconv_lower
            )

            _, trunc_error_lower = self.phi_left(value_arr_0, z, trunc_error_lower)
            _, trunc_error_upper = self.phi_left(value_arr_0, z, trunc_error_upper)

            output = {
                "z": z,
                "value": np.mean(value_arr),
                "stat_error": stat_error,
                "trunc_error": trunc_error,
                "trunc_error_lower": np.median(trunc_error_lower),
                "trunc_error_upper": np.median(trunc_error_upper),
                "rconv": np.median(rconv),
                "rconv_stat_error": np.std(rconv),
                "rconv_lower": np.median(rconv_lower),
                "rconv_upper": np.median(rconv_upper),
            }
            if verbose:
                key_list = [
                    "z",
                    "value",
                    "stat_error",
                    "trunc_error",
                    "trunc_error_lower",
                    "trunc_error_upper",
                    "rconv",
                    "rconv_stat_error",
                    "rconv_lower",
                    "rconv_upper",
                ]
                for key in key_list:
                    print((key, ":\t", output[key]))
                print()
            if full_output:
                return output

        return np.mean(value_arr), stat_error, trunc_error

    def conformal_transform(self, phi):
        phi_rev_series = phi.rev_series(len(self._series_sampled))
        tr_series_sampled = np.empty_like(self._series_sampled)

        for i in range(len(self._series_sampled)):
            tr_series_sampled[i, :] = np.real(
                compose_series(self._series_sampled[i], phi_rev_series)
            )

        tr_series_obj = deepcopy(self)
        tr_series_obj._series_sampled = tr_series_sampled
        tr_series_obj._phi_right_list.append(np.vectorize(phi))

        return tr_series_obj

    def one_over_series(self):
        tr_series_sampled = one_over_series(self._series_sampled, axis=1)

        tr_series_obj = deepcopy(self)
        tr_series_obj._series_sampled = tr_series_sampled
        tr_series_obj._phi_left_list.append(
            np.vectorize(lambda x, z: 1.0 / x, excluded={1})
        )
        tr_series_obj._phi_left_der_list.append(
            np.vectorize(lambda x, z: old_div(-1.0, (x * x)), excluded={1})
        )

        return tr_series_obj

    def shift(self, offset):

        tr_series_obj = deepcopy(self)
        tr_series_obj._series_sampled[:, 0] += offset
        tr_series_obj._phi_left_list.append(
            np.vectorize(lambda x, z: x - offset, excluded={1})
        )
        tr_series_obj._phi_left_der_list.append(
            np.vectorize(lambda x, z: 1.0, excluded={1})
        )

        return tr_series_obj

    def shift_prop(self, factor):
        order_0 = np.mean(self._series_sampled[:, 0])
        return self.shift((factor - 1.0) * order_0)

    def eliminate_pole(self):
        tr_series_sampled = one_over_series(self._series_sampled, axis=1)
        order_0 = np.mean(tr_series_sampled[:, 0])
        tr_series_sampled = one_over_series(tr_series_sampled[:, 1:], axis=1)

        tr_series_obj = deepcopy(self)
        # TODO: adjust start_geom if necessary
        tr_series_obj._series_sampled = tr_series_sampled
        tr_series_obj._phi_left_list.append(
            np.vectorize(lambda x, z: old_div(x, (z + x * order_0)), excluded={1})
        )
        tr_series_obj._phi_left_der_list.append(
            np.vectorize(lambda x, z: old_div(z, (z + x * order_0) ** 2), excluded={1})
        )
        return tr_series_obj

    def eliminate_order0(self):

        tr_series_obj = deepcopy(self)
        # TODO: adjust start_geom if necessary
        order_0 = np.mean(tr_series_obj._series_sampled[:, 0])
        print(("Order 0 eliminated =", order_0))
        tr_series_obj._series_sampled = tr_series_obj._series_sampled[:, 1:]
        tr_series_obj._phi_left_list.append(
            np.vectorize(lambda x, z: (z * x + order_0), excluded={1})
        )
        tr_series_obj._phi_left_der_list.append(
            np.vectorize(lambda x, z: z, excluded={1})
        )

        return tr_series_obj

    def move_pole(self, z_target, value_target=None):
        """
        Construct an inverse shifted serie with a pole at `z_target` (in the
        series space).

        If `value_target` is not provided, the inverse series
        is summed at `z_target` to provide a value, but this sum may not
        converge. If provided, it should be the expected value of the series at
        `z_target`.
        """
        series_inv = self.one_over_series()  # this is not the inverse
        series_inv._phi_left_list = []
        series_inv._phi_left_der_list = []
        series_inv._phi_right = []
        if value_target is None:
            value_target, stat_error, trunc_error = series_inv.sum(z_target)

            if np.isinf(trunc_error):
                print("WARNING: target is out of convergence radius")
            elif np.abs(value_target) < stat_error + trunc_error:
                print("WARNING: target is very unprecise")

        if np.imag(z_target) == 0.0:
            alpha = np.real(value_target)
            beta = 0.0
        else:
            beta = old_div(np.imag(value_target), np.imag(z_target))
            alpha = np.real(value_target) - beta * np.real(z_target)

        print(("alpha =", alpha))
        print(("beta =", beta))

        tr_series_sampled = series_inv._series_sampled
        tr_series_sampled[:, 0] -= alpha
        tr_series_sampled[:, 1] -= beta
        tr_series_sampled = one_over_series(tr_series_sampled, axis=1)

        tr_series_obj = deepcopy(self)
        tr_series_obj._series_sampled = tr_series_sampled
        tr_series_obj._phi_left_list.append(
            np.vectorize(
                lambda x, z: old_div(1, (old_div(1, x) + alpha + z * beta)),
                excluded={1},
            )
        )
        tr_series_obj._phi_left_der_list.append(
            np.vectorize(
                lambda x, z: old_div(1, (x * (old_div(1, x) + alpha + z * beta)) ** 2),
                excluded={1},
            )
        )
        return tr_series_obj


if __name__ == "__main__":

    # ### tests prod_poly
    # assert np.array_equal(prod_poly([1, 0, 1], [1, 2]), np.array([1, 2, 1, 2], dtype=complex))

    # ### tests compose_poly
    # assert np.array_equal(compose_poly([1, 1, 1], [1j, 1]), np.array([1j, 1+2j, 1], dtype=complex))

    print("All tests passed.")
