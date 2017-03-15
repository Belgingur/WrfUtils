import inspect
import logging
from types import FunctionType
from typing import Dict, Iterable

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
        self.cache = {}  # type: Dict[str, np.ndarray]

    def __call__(self, var_name: str) -> np.ndarray:
        try:
            r = self.cache[var_name]
            LOG.info('CACHE HIT: %s', var_name)
        except KeyError:
            LOG.info('CACHE MISS: %s', var_name)
            if var_name in self.vars:
                r = self.get_chunk_native(var_name)
            elif var_name in CALCULATORS:
                r = self.get_chunk_derived(var_name)
            else:
                raise ValueError('Unknown variable: ' + var_name)
            self.cache[var_name] = r
        return r

    def get_chunk_native(self, var_name) -> np.ndarray:
        LOG.info('### get_chunk_native(%s)', var_name)
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
        LOG.info('chunk: %s', chunk)

        # Interpolate as needed
        if ipor:
            LOG.info('        interpolate on: %s', ipor.dimension)
            chunk = ipor(chunk)
        out_shape = chunk.shape
        if in_shape != out_shape:
            LOG.info('        shape: %s -> %s', in_shape, out_shape)

        return chunk

    def get_chunk_derived(self, var_name) -> np.ndarray:
        LOG.info('### get_chunk_derived(%s)', var_name)
        calc = CALCULATORS[var_name]
        LOG.info('calc.inputs: %s', calc.inputs)
        inputs = tuple(self(vn) for vn in calc.inputs)
        LOG.info('inputs: %s', inputs)
        output = calc(*inputs)
        LOG.info('output: %s', output)
        return output


def derived(
        dimensions: Iterable[str] = ('Time', 'bottom_top', 'south_north', 'west_east'),
        datatype: np.dtype = np.int16,
        description: str = None,
        scale_factor: float = 0.01,
        add_offset: float = None,
        FieldType: int = None,
        MemoryOrder: str = None,
        units: str = None,
        coordinates: str = None,
):
    """
    Marks a function as a calculator for a derived variable and adds it to CALCULATORS.

    It adds these attributes:
    - `inputs` indicating the names of the input variables, which can in turn be derived variables.
    - `dimensions` indicating the dimensions of the result.
    """

    def wrapper(f):

        sig = inspect.signature(f)
        setattr(f, 'inputs', tuple(sig.parameters.keys()))
        setattr(f, 'dimensions', tuple(dimensions))
        setattr(f, 'datatype', datatype)

        nonlocal description
        if description is None:
            description = f.__name__.replace('-', ' ').replace('_', ' ')
        setattr(f, 'description', description)

        setattr(f, 'scale_factor', scale_factor)
        setattr(f, 'add_offset', add_offset)
        setattr(f, 'FieldType', FieldType)
        setattr(f, 'MemoryOrder', MemoryOrder)
        setattr(f, 'units', units)
        setattr(f, 'coordinates', coordinates)

        CALCULATORS[f.__name__] = f
        return f

    return wrapper


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DERIVED VARIABLE DEFINITIONS

# The default attributes can be seen in the definition of `derived` above.
# Most notably variables by default:
#   - Use all the staggered dimensions
#   - Have a scaling factor of 0.01
#   - Use a 16-bit signed integer datatype
#   - Have a description derived from the variable nme
# Override for calculators with other values.

@derived(
    description='x-wind component in earth-coordinates',
    units='m s-1',
)
def U_true(U, V, COSALPHA, SINALPHA):
    ca = np.expand_dims(COSALPHA, 1)
    sa = np.expand_dims(SINALPHA, 1)
    return ca * U + sa * V


@derived(
    description='y-wind component in earth-coordinates',
    units='m s-1',
)
def V_true(U, V, COSALPHA, SINALPHA):
    ca = np.expand_dims(COSALPHA, 1)
    sa = np.expand_dims(SINALPHA, 1)
    return -sa * U + ca * V


@derived(
    units='m s-1',
    datatype=np.uint16,
)
def wind_speed(U, V):
    return np.sqrt(U ** 2 + V ** 2)


@derived(
    description='wind direction',
    units='degrees',
    datatype=np.uint16,
)
def wind_dir(U_true, V_true):
    return np.degrees(np.arctan2(V_true, U_true))


@derived(
    units='Pa',
    datatype=np.uint16,
)
def pressure(P, PB):
    return P + PB


@derived(
    datatype=np.uint16,
    units='K'
)
def potential_temperature(T):
    return T + 300.


@derived(
    datatype=np.uint16,
    units='K'
)
def temperature(potential_temperature, pressure):
    return potential_temperature * (pressure / 100000.0) ** 0.2856




@derived(
    datatype=np.uint16,
    units='kg m-3'
)
def density(ALT):
    return 1/ALT


