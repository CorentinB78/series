"""
Arithmetics of series.
"""
import numpy as np


def _power_gen(U):
    """
    Generator yielding successive powers of `U` (litterally U**k)

    We compute the power each time to reduce rounding errors.
    """
    k = 0
    while True:
        yield U**k
        k += 1


def rescale_series(series, U, axis=-1):
    """
    `series` ND array whose axis `axis` represents the orders.
    `U` a scalar or array (broadcastable to `series` excluding axis `axis`)

    Returns a new series rescaled by `U` along axis `axis`:
    s_n' = s_n U^n
    """
    series_ = np.moveaxis(np.asarray(series), axis, 0)
    power_U = _power_gen(U)

    series_ = [x * next(power_U) for x in series_]

    return np.moveaxis(series_, 0, axis)


def prod_series(series1, series2, axis1=-1, axis2=-1):
    """
    Product of two truncated series, up to the smallest truncation order.
    """
    series1_ = np.moveaxis(np.asarray(series1), axis1, 0)
    series2_ = np.moveaxis(np.asarray(series2), axis2, 0)

    N = min(len(series1_), len(series2_))
    dtype = np.result_type(series1_, series2_)
    new_shape = (N,) + np.broadcast(series1_[0], series2_[0]).shape
    output = np.empty(new_shape, dtype=dtype)

    for n in range(N):
        output[n] = series1_[0] * series2_[n]
        for k in range(1, n + 1):
            output[n] += series1_[k] * series2_[n - k]
    return np.moveaxis(output, 0, axis1)


def one_over_series(series, axis=-1, one=1.0):
    """
    Inverse (regarding multiplication) of a series, truncated at the same order.
    """
    series_ = np.asarray(series)
    series_inv = np.empty_like(series_)  # output array has same ordering as input
    series_ = np.moveaxis(series_, axis, 0)
    view = np.moveaxis(series_inv, axis, 0)

    if np.any(series_[0] == 0):
        raise ZeroDivisionError("constant coefficient cannot be zero")

    view[0] = one / series_[0]

    for n in range(1, len(series_)):
        view[n] = -view[0] * series_[n]
        for k in range(1, n):
            view[n] -= view[k] * series_[n - k]
        view[n] /= series_[0]

    return series_inv


def divide_series(series1, series2, axis1=-1, axis2=-1):
    """
    Division of series series1/series2, up to the smallest truncation order.
    """
    series1_ = np.moveaxis(np.asarray(series1), axis1, 0)
    series2_ = np.moveaxis(np.asarray(series2), axis2, 0)

    if (series2_[0] == 0).any():
        raise ZeroDivisionError("denominator constant coefficient cannot be zero")

    N = min(len(series1_), len(series2_))
    dtype = np.result_type(series1_, series2_)
    new_shape = (N,) + np.broadcast(series1_[0], series2_[0]).shape
    output = np.zeros(new_shape, dtype=dtype)

    output[:] = series1_[:N] / series2_[0]

    for n in range(1, N):
        v = output[0] * series2_[n]
        for k in range(1, n):
            v += output[k] * series2_[n - k]
        output[n] -= v / series2_[0]

    return np.moveaxis(output, 0, axis1)


def compose_series(series1, series2, axis1=-1, axis2=-1):
    """
    Composition of two truncated series series1(series2), up to the smallest truncation order.
    It is necessary that series2(0) = 0 to ensure that the result is the truncated series of the composition.

    If series2 represent a function g(x) near x=x_0, and series1 represent a function f(y) near y=g(x_0), then the output series represent the function f(g(x)) near x=x_0.

    This is using Horner's method for polynomial evaluation.
    """
    series1_ = np.moveaxis(np.asarray(series1), axis1, 0)
    series2_ = np.moveaxis(np.asarray(series2), axis2, 0)

    if (series2_[0] != 0).any():
        raise ValueError(
            f"constant coefficient of series2 must be zero, but is {series2_[0]}"
        )

    N = min(len(series1_), len(series2_))
    dtype = np.result_type(series1_, series2_)
    new_shape = (N,) + np.broadcast(series1_[0], series2_[0]).shape
    output = np.zeros(new_shape, dtype=dtype)

    output[0] = series1_[N - 1]
    for j in range(N - 2, -1, -1):
        output = prod_series(output, series2_[:N], 0, 0)
        output[0] += series1_[j]

    return np.moveaxis(output, 0, axis1)


def reverse_series(series, axis=0):
    """
    Reversed (regarding composition) series, truncated at the same order, and along first axis.

    If the input series represent f(x) near x=x_0, then the output series represent the function g(y) near y=0 such that f^{-1}(y) = g(y - f(x_0)) + x_0. Note the constant coefficient f(x_0) is not used in the computation.

    Newton's method (R. Brent and H.T. Kung, Algorithms for composition and reversion of power series,
    Analytic Computational Complexity, Academic Press, New York, 1975, pp. 217-225.)

    http://repository.cmu.edu/cgi/viewcontent.cgi?article=2520&context=compsci
    """
    # TODO: don't change memory order between input and output
    series_ = np.moveaxis(np.asarray(series), axis, 0)

    if (series_[1] == 0).any():
        raise ValueError(
            "The series is not reversible, as its linear coefficient is zero."
        )

    output = np.zeros_like(series_)
    output[1] = 1.0 / series_[1]

    der_series = np.append(series_[1:] * np.arange(1, len(series_)), [0.0])

    k = 1
    while k < len(series):
        output[k + 1 :] = 0.0
        compo = compose_series(series, output, 0, 0)
        compo[0] = 0.0
        compo[1] -= 1.0
        compo_p = compose_series(der_series, output, 0, 0)

        output = output - divide_series(compo, compo_p, 0, 0)
        k = 2 * k + 1

    return np.moveaxis(output, 0, axis)
