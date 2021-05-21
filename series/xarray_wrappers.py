"""
xarray wrappers for some functions dealing with numpy arrays.
"""
# import numpy as np
import xarray as xr
from . import arithmetics

# def rescale_series(series, U, dim):
#     return xr.apply_ufunc(arithmetics.rescale_series, series, U, input_core_dims=[[dim], []], output_core_dims=[[dim]], kwargs={'axis': -1})

def prod_series(series1, series2, dim):
    out = xr.apply_ufunc(arithmetics.prod_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})

def one_over_series(series, dim):
    return xr.apply_ufunc(arithmetics.one_over_series, series, input_core_dims=[[dim]], output_core_dims=[[dim]], kwargs={'axis': -1})

def divide_series(series1, series2, dim):
    out = xr.apply_ufunc(arithmetics.divide_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})

def compose_series(series1, series2, dim):
    out = xr.apply_ufunc(arithmetics.compose_series, series1, series2, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], exclude_dims={dim}, kwargs={'axis': -1})
    return out.assign_coords({dim: series1[dim][:out.sizes[dim]]})


prod_series.__doc__ = arithmetics.prod_series.__doc__
one_over_series.__doc__ = arithmetics.one_over_series.__doc__
divide_series.__doc__ = arithmetics.divide_series.__doc__
compose_series.__doc__ = arithmetics.compose_series.__doc__
