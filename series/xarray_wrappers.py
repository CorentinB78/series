"""
xarray wrappers for some functions dealing with numpy arrays.
"""
# import numpy as np
import xarray as xr

def _check_array(a):
    if not(isinstance(a, xr.DataArray) or isinstance(a, xr.Dataset)):
        raise TypeError('Not an xarray instance')

############################ arithmetics ##########################
from . import arithmetics

# def rescale_series(series, U, dim):
    # _check_array(series)
    # _check_array(U)
#     return xr.apply_ufunc(arithmetics.rescale_series, series, U, input_core_dims=[[dim], []], output_core_dims=[[dim]], kwargs={'axis': -1})

def prod_series(series1, series2, dim):
    _check_array(series1)
    _check_array(series2)
    out = xr.apply_ufunc(arithmetics.prod_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis1': -1, 'axis2': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})

def one_over_series(series, dim):
    _check_array(series)
    return xr.apply_ufunc(arithmetics.one_over_series, series, input_core_dims=[[dim]], output_core_dims=[[dim]], kwargs={'axis': -1})

def divide_series(series1, series2, dim):
    _check_array(series1)
    _check_array(series2)
    out = xr.apply_ufunc(arithmetics.divide_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis1': -1, 'axis2': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})

def compose_series(series1, series2, dim):
    _check_array(series1)
    _check_array(series2)
    out = xr.apply_ufunc(arithmetics.compose_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis1': -1, 'axis2': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})


prod_series.__doc__ = arithmetics.prod_series.__doc__
one_over_series.__doc__ = arithmetics.one_over_series.__doc__
divide_series.__doc__ = arithmetics.divide_series.__doc__
compose_series.__doc__ = arithmetics.compose_series.__doc__

########################## resummation ############################
from . import resummation

def resum_series(series, conformal_trans, dim):
    _check_array(series)
    N = series.sizes[dim]
    conf_series = conformal_trans.rev_series(N - 1)
    conf_series = xr.DataArray(conf_series, dims=[dim])
    return compose_series(series, conf_series, dim=dim)

def sum_series(series, U, dim):
    _check_array(series)
    _check_array(U)
    # TODO: check dim is not in U
    return xr.apply_ufunc(resummation.sum_series, series, U, input_core_dims=[[dim], []], output_core_dims=[[]], exclude_dims={dim}, kwargs={'axis': -1})

resum_series.__doc__ = resummation.resum_series.__doc__
sum_series.__doc__ = resummation.sum_series.__doc__
