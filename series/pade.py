# coding: UTF-8
"""
Padé and robust Padé for complex series.
"""
import numpy as _np
from scipy import linalg as _linalg
from .resummation import Rconv_1d, resum_series, sum_series, IdentityTransform
from .arithmetics import rescale_series


def robust_pade(taylor_series, m, n, tol, full_output=False, rescale=True):
    """
    Compute robust Padé via SVD, as defined and implemented in
    Gonnet, Guttel, Trefethen, "Robust Padé Approximation via SVD", SIAM review Vol.55, No.1, pp.101-117
    DOI 10.1137/110853236

    full output: a, b, chi, lambda, delta
    """
    m = int(m)
    n = int(n)
    taylor_series = _np.asarray(taylor_series)
    if taylor_series.dtype == complex:
        dtype = complex
    else:
        dtype = float 

    if m + n + 1 > len(taylor_series):
        raise ValueError('Not enough coefficients in Taylor series for this type of Padé.')

    if rescale:
        r = float(Rconv_1d(taylor_series)[0])

    c = _np.empty(m + n + 1, dtype=dtype)
    if len(taylor_series) < m + n + 1:
        c[:len(taylor_series)] = taylor_series
        c[len(taylor_series):] = 0.
    else:
        c[:] = taylor_series[:m + n + 1]

    if rescale:
        c = rescale_series(c, r)

    ts = tol * _linalg.norm(c)

    if _np.max(_np.abs(c[:m + 1])) <= tol * _np.max(_np.abs(c)):
        if full_output:
            return _np.zeros(1), _np.ones(1), None, None, n
        return _np.zeros(1), _np.ones(1)

    row = _np.r_[c[0], _np.zeros(n, dtype=dtype)]
    col = c
    chi = 0
    while True:
        if n == 0:
            a = c[:m + 1]
            b = _np.ones(1)
            break
        Z = _linalg.toeplitz(col[:m + n + 1], row[:n + 1])
        C = Z[m + 1:m + n + 1, :]
        assert(C.shape == (n, n+1))
        U, S, Vh = _linalg.svd(C, full_matrices=True)
        rho = _np.sum(S > ts)
        chi += n - rho
        if rho == n:
            break
        m = m - (n - rho)
        n = rho
        assert(m >= 0)

    lam = 0
    if n > 0:
        # V = np.conj(Vh.T)
        b = _np.conj(Vh[n, :])
        # assert(_np.all(_np.abs(_np.dot(C, b)) < ts))
        # D = _np.diag(_np.abs(b) + _np.sqrt(_np.spacing(1)))
        # Q, R = _linalg.qr(_np.dot(C, D).T)
        # b = _np.dot(D, Q[:, n]) ### out of range index!!
        b /= _linalg.norm(b)
        a = _np.dot(Z[:m + 1, :n + 1], b)
        idx = _np.nonzero(_np.abs(b) > tol)[0]
        lam = idx[0]
        b = b[lam:idx[-1] + 1]
        a = a[lam:]

    # a = a[:_np.nonzero(_np.abs(a) > ts)[0][-1] + 1]
    a /= b[0]
    b /= b[0]

    if rescale:
        a = rescale_series(a, 1 / r)
        b = rescale_series(b, 1 / r)

    if full_output:
        return a, b, chi, lam, chi + lam
    return a, b


def pade(taylor_series, m, n, rescale=True):
    """
    Compute the Padé interpolant of a Taylor series.

    Contrarily to mpmath.pade, complex series are handled.
    """
    return robust_pade(taylor_series, m, n, tol=0.0, rescale=rescale)


def eval_pade(num, denom, U):
    return sum_series(num, U) / sum_series(denom, U)


def find_poles_with_pade(series, max_ratio, transforms=None, tol=0.):
    """
    returns poles, zeros
    """
    if transforms is None:
        transforms = [IdentityTransform()]
    series_ = _np.asarray(series)
    assert(series_.ndim == 1)

    max_ratio = float(max_ratio)
    assert(max_ratio >= 1)
    order = len(series_) - 1
    zeros = []
    poles = []

    for phi in transforms:
        tr_series = resum_series(series_, phi)
        for l in range(order + 1):
            for m in range(order - l + 1):
                if m / max_ratio <= l and l / max_ratio <= m:
                    try:
    #                     p, q = pade(tr_series, l, m)
                        p, q = robust_pade(tr_series, l, m, tol=tol)
                    except ZeroDivisionError: # singular matrix
                        pass
                    else:
                        poles.extend([phi.rev(z) for z in _np.roots(q[::-1])])
                        zeros.extend([phi.rev(z) for z in _np.roots(p[::-1])])
    return poles, zeros
