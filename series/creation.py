"""
Creation of particular series.
"""
# from __future__ import division
# from __future__ import print_function
from .arithmetics import rescale_series, divide_series
from builtins import range
import numpy as np

########## Particular series ##########


def log_series(N, x0=1.0):
    """
    Taylor series of f(x) = ln(x0 + x), for x close to zero, up to order N (included).
    """
    s = np.empty(N + 1, dtype=complex)
    s[0] = np.log(x0)
    s[1:] = -1.0 / np.arange(1, N + 1, dtype=float)
    return rescale_series(s, -1.0 / x0)


def exp_series(N, x0=0.0):
    """
    Taylor series of f(x) = exp(x0 + x), for x close to zero, up to order N (included).
    """
    s = np.empty(N + 1, dtype=complex)
    s[0] = 1.0
    for k in range(1, N + 1):
        s[k] = s[k - 1] / float(k)
    return np.exp(x0) * s


def cos_series(N, x0=0.0):
    """
    Taylor series of f(x) = cos(x0 + x), for x close to zero, up to order N (included).
    Not tested !
    """
    s_plus = rescale_series(exp_series(N, x0=1.0j * x0), 1.0j)
    s_minus = rescale_series(exp_series(N, x0=-1.0j * x0), -1.0j)
    return 0.5 * (s_plus + s_minus)


def sin_series(N, x0=0.0):
    """
    Taylor series of f(x) = sin(x0 + x), for x close to zero, up to order N (included).
    Not tested !
    """
    s_plus = rescale_series(exp_series(N, x0=1.0j * x0), 1.0j)
    s_minus = rescale_series(exp_series(N, x0=-1.0j * x0), -1.0j)
    return -0.5j * (s_plus - s_minus)


def tan_series(N, x0=0.0):
    """
    Taylor series of f(x) = tan(x0 + x), for x close to zero, up to order N (included).
    Not tested !
    """
    s_sin = sin_series(N, x0=x0)
    s_cos = cos_series(N, x0=x0)
    try:
        return divide_series(s_sin, s_cos)
    except ValueError:
        raise ValueError(f"tan has a singularity at x0={x0}")


def cosh_series(N, x0=0.0):
    """
    Taylor series of f(x) = cosh(x0 + x), for x close to zero, up to order N (included).
    """
    e_p = exp_series(N)
    e_m = rescale_series(e_p, -1.0)
    return 0.5 * (np.exp(x0) * e_p + np.exp(-x0) * e_m)


def arctan_series(N, x0=0.0):
    """
    Taylor series of f(x) = arctan(x0 + x), for x close to zero, up to order N (included).
    """
    l1 = rescale_series(log_series(N, 1.0 + 1.0j * x0), 1.0j)
    l2 = rescale_series(log_series(N, 1.0 - 1.0j * x0), -1.0j)
    return (l1 - l2) / 2.0j


def power_law_series(alpha, N):
    """
    Taylor series of f(x) = (1 + x)**alpha, for x close to zero, up to order N (included).
    """
    output = np.zeros(N + 1, dtype=float)
    output[0] = 1
    for n in range(1, N + 1):
        output[n] = (alpha - n + 1) / float(n) * output[n - 1]
    return output


def sqrt_series(N, x0=1):
    """
    Taylor series of f(x) = sqrt(x0 + x), for x close to zero, up to order N (included).
    """
    x0 = complex(x0)
    output = np.zeros(N + 1, dtype=complex)
    output[0] = np.sqrt(x0)
    for n in range(1, N + 1):
        output[n] = (1.5 - n) / float(n) / x0 * output[n - 1]
    return output


def poly_to_series(coeffs, N):
    output = np.zeros(N + 1, dtype=complex)
    n = min(len(output), len(coeffs))
    output[:n] = coeffs[:n]
    return output
