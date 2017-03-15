import inspect
import logging
from types import FunctionType
from typing import Dict

import numpy as np
from netCDF4 import Dataset, Variable

from utils import destagger_array_by_dim, DIM_SOUTH_NORTH_STAG, DIM_WEST_EAST_STAG
from vertical_interpolation import Interpolator

LOG = logging.getLogger(__name__)

# Mapping from derived variable name to function to calculate that variable.
# It is built from functions named like the cariables annotated with @calculator
CALCULATORS = dict()  # type: Dict[str, FunctionType]


class ChunkCalculator(object):
    """
    Instances will return variables from a Dataset in a time-range, destaggerd in the plane and vertically interpolated
    as needed.

    In addition to native variables, it can return derived variables by calling one of the calculating functions
    decorated with @derived and passing in the required input variables, which in turn can also either be native
    or derived.
    """

    def __init__(self, ds: Dataset, t_start: int, t_end: int, ipor_alig: Interpolator, ipor_stag: Interpolator):
        super().__init__()
        self.ds = ds
        self.vars = ds.variables  # type: Dict[str, Variable]
        self.t_start = t_start
        self.t_end = t_end
        self.ipor_alig = ipor_alig
        self.ipor_stag = ipor_stag

    def __call__(self, var_name: str):
        if var_name in self.vars:
            return self.get_chunk_native(var_name)
        else:
            raise ValueError('Unknown variable: ' + var_name)

    def get_chunk_native(self, var_name):
        var = self.vars[var_name]
        dims = var.dimensions

        if self.ipor_alig.dimension in dims:
            ipor = self.ipor_alig
        elif self.ipor_stag.dimension in dims:
            ipor = self.ipor_stag
        else:
            ipor = None

        if ipor:
            chunk = var[self.t_start:self.t_end, 0:ipor.max_k + 1]
        else:
            chunk = var[self.t_start:self.t_end]
        in_shape = chunk.shape

        # Destagger as needed
        chunk = destagger_array_by_dim(chunk, dims, DIM_SOUTH_NORTH_STAG, log_indent=8)
        chunk = destagger_array_by_dim(chunk, dims, DIM_WEST_EAST_STAG, log_indent=8)

        # Interpolate as needed
        if ipor:
            LOG.info('        interpolate on: %s', ipor.dimension)
            chunk = ipor(chunk)
        out_shape = chunk.shape
        if in_shape != out_shape:
            LOG.info('        shape: %s -> %s', in_shape, out_shape)

        return chunk


def derived(*dimensions: str):
    """
    Marks a function as a calculator for a derived variable and adds it to CALCULATORS.

    It adds these attributes:
    - `inputs` indicating the names of the input variables, which can in turn be derived variables.
    - `dimensions` indicating the dimensions of the result.
    """

    def wrapper(f):
        sig = inspect.signature(f)
        setattr(f, 'inputs', tuple(sig.parameters.keys()))
        setattr(f, 'dimensions', dimensions)
        CALCULATORS[f.__name__] = f
        return f

    return wrapper


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DERIVED VARIABLE DEFINITIONS

@derived('Time', 'bottom_top', 'south_north', 'west_east')
def U_true(U, V, COSALPHA, SINALPHA):
    return COSALPHA * U + SINALPHA * V


@derived('Time', 'bottom_top', 'south_north', 'west_east')
def V_true(U, V, COSALPHA, SINALPHA):
    return -SINALPHA * U + COSALPHA * V


@derived('Time', 'bottom_top', 'south_north', 'west_east')
def wind_speed(U, V):
    return np.sqrt(U ** 2 + V ** 2)


@derived('Time', 'bottom_top', 'south_north', 'west_east')
def wind_dir(U_true, V_true):
    return np.degrees(np.arctan2(V_true, U_true))
