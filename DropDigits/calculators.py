import inspect
import logging
from enum import Enum
from types import FunctionType
from typing import Dict, Iterable
from typing import List
from typing import Set
from typing import Union

import numpy as np
from netCDF4 import Dataset, Variable

from utils import destagger_array_by_dim, DIM_SOUTH_NORTH_STAG, DIM_WEST_EAST_STAG, DIM_BOTTOM_TOP, DIM_BOTTOM_TOP_STAG, \
    g_inv, memoize, destagger_array
from vertical_interpolation import Interpolator, build_interpolator

LOG = logging.getLogger('belgingur.calculators')

# Mapping from derived variable name to function to calculate that variable.
# It is built from functions named like the cariables annotated with @calculator
CALCULATORS = dict()  # type: Dict[str, FunctionType]


class HeightType(Enum):
    above_ground = 1,
    above_sea = 2,
    pressure = 3,


class ChunkCalculator(object):
    """
    Instances will return variables from a Dataset in a time-range, destaggerd in the plane and vertically interpolated
    as needed.

    In addition to native variables, it can return derived variables by calling one of the calculating functions
    decorated with @derived and passing in the required input variables, which in turn can also either be native
    or derived.
    """

    def __init__(self, t_start: int, t_end: int, heights: List[int], height_type: HeightType):
        super().__init__()
        self.datasets = []  # type: List[(Dataset, int, Union[None, Set[str]])]
        self.static_vars = set()  # type: Set[str]

        self.t_start = t_start
        self.t_end = t_end
        self.heights = heights
        if height_type is True:
            self.height_type = HeightType.above_ground
        elif height_type is False:
            self.height_type = HeightType.above_sea
        else:
            self.height_type = height_type

        # Lazily crated objects
        self._ipor_stag = None
        self._ipor_alig = None

        # Cached variable data
        self.cache = {}  # type: Dict[str, np.ndarray]

    def add_dataset(self, ds: Dataset, margin: int = None, var_names: Iterable[str] = None):
        """
        Add dataset to take variables from and an optional list of dataset to allow reading.
        If the given dataset is, then do nothing.
        If var_names is None, then allow all variables to be read.
        """
        if ds is None:
            return
        if var_names is None:
            var_names = set(ds.variables.keys())
        else:
            var_names = {name for name in var_names if name in ds.variables}
        self.datasets.append((ds, margin, var_names))

    def make_vars_static(self, *var_names: str):
        """ Specify that the givne variables are static and we will always return only the first time step. """
        for var_name in var_names:
            self.static_vars.add(var_name)

    @memoize
    def z_stag(self):
        """ Return the heights of the input file's sigma levels as dictated by the height_type """
        LOG.info('        calculate z_stag')

        if self.height_type in (HeightType.above_sea, HeightType.above_ground):
            ph, m, n = self.get_var_native('PH')
            ph = ph[self.t_start:self.t_end, ..., m:n, m:n]
            phb, m, n = self.get_var_native('PHB')
            phb = phb[self.t_start:self.t_end, ..., m:n, m:n]

            z_stag = (ph + phb) * g_inv  # ph and phb are vertically staggered
            if self.height_type == HeightType.above_ground:
                hgt, m, n = self.get_var_native('HGT', 'HGT_M')
                hgt = hgt[0, m:n, m:n]
                z_stag -= hgt
            return z_stag

        else:
            return destagger_array(self.z_alig(), 1)

    @memoize
    def z_alig(self):
        LOG.info('        calculate z')
        if self.height_type in (HeightType.above_sea, HeightType.above_ground):
            return destagger_array(self.z_stag(), 1)

        else:
            p, m, n = self.get_var_native('P')
            p = p[self.t_start:self.t_end, ..., m:n, m:n]
            pb, m, n = self.get_var_native('PB')
            pb = pb[self.t_start:self.t_end, ..., m:n, m:n]
            return (p + pb) / 100

    def ipor_stag(self) -> Interpolator:
        if self._ipor_stag is None:
            self._ipor_stag = build_interpolator(self.z_stag(), self.heights, True)
        return self._ipor_stag

    def ipor_alig(self) -> Interpolator:
        if self._ipor_alig is None:
            self._ipor_alig = build_interpolator(self.z_alig(), self.heights, False)
        return self._ipor_alig

    def __call__(self, var_name: str) -> np.ndarray:
        # synthetic variable wanted by e.g. `height`
        if var_name == '_chunk_calculator':
            return self

        # Attempt to return variable from cache
        r = self.cache.get(var_name, None)
        if r is not None:
            LOG.info('        cached %s', var_name)
            return r

        # Attempt to read variable from a dataset
        var, m, n = self.get_var_native(var_name)
        if var is not None:
            r = self.get_chunk_native(var, m, n)

        # Attempt to calculate variable
        if r is None and var_name in CALCULATORS:
            r = self.get_chunk_derived(var_name)

        # it all failed.
        if r is None:
            raise ValueError('Unknown variable: ' + var_name)

        self.cache[var_name] = r
        return r

    def get_var_native(self, var_name: str, var_name2: str = None) -> (Variable, int):
        """ Finds a native variable and the associated margin by looking through the list of added datasets. """
        for ds, margin, var_names in self.datasets:
            if var_name in var_names:
                var = ds.variables[var_name]
            elif var_name2 in var_names:
                var = ds.variables[var_name2]
            else:
                continue
            if margin:
                return var, margin, -margin
            else:
                return var, None, None
        return None, None, None

    def get_chunk_native(self, var: Variable, m: int, n: int) -> np.ndarray:
        LOG.info('        read %s', var.name)
        dims = var.dimensions

        if DIM_BOTTOM_TOP in dims:
            ipor = self.ipor_alig()
        elif DIM_BOTTOM_TOP_STAG in dims:
            ipor = self.ipor_stag()
        else:
            ipor = None

        # Build indices and slices to read exactly what we need from the variable
        slices: List[Union[None, int, slice]] = []

        # Statics get a single time step. Others get the current chunk of time
        if var.name in self.static_vars:
            slices.append(0)
        else:
            slices.append(slice(self.t_start, self.t_end))
        # LOG.info('            %s[0,0]: %s %g .. %g', var.name, var.shape[2:], np.min(var[:]), np.max(var[:]))

        # Vertically interpolated variables only need sigma levels up to the highest used in the interpolation
        if ipor:
            slices.append(slice(0, ipor.max_k + 1))

        # Cut off any margin
        if m:
            slices.append(Ellipsis)
            slices.append(slice(m, n))
            slices.append(slice(m, n))

        chunk = var.__getitem__(tuple(slices))
        in_shape = chunk.shape

        # Destagger as needed
        chunk = destagger_array_by_dim(chunk, dims, DIM_SOUTH_NORTH_STAG, log_indent=12)
        chunk = destagger_array_by_dim(chunk, dims, DIM_WEST_EAST_STAG, log_indent=12)

        # Interpolate as needed
        if ipor:
            LOG.info('        interpolate on: %s', ipor.dimension)
            chunk = ipor(chunk)
        out_shape = chunk.shape
        if in_shape != out_shape:
            LOG.debug('        shape: %s -> %s', in_shape, out_shape)

        return chunk

    def get_chunk_derived(self, var_name) -> np.ndarray:
        calc = CALCULATORS[var_name]
        inputs = tuple(self(vn) for vn in calc.inputs)
        LOG.info('        calculate %s', var_name)
        output = calc(*inputs)
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

    # Avoid int scale & offset which can cause overflows in netCDF4 library
    if isinstance(scale_factor, int):
        scale_factor=float(scale_factor)
    if isinstance(add_offset, int):
        add_offset=float(add_offset)

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
#   - Use a 16-bit signed integer datatype   => range +- 327.68
#   - Have a description derived from the variable nme
# Override for calculators with other values.

@derived(
    dimensions=('bottom_top',),
    description='Height of vertical interpolation surfaces',
    units='m',
    datatype=np.int16,  # We interpolate to integer heights. ±32768
)
def height(_chunk_calculator: ChunkCalculator):
    """ :param _chunk_calculator: Super magic variable giving the ChunkCalculator which is calling us. """
    return np.array(_chunk_calculator.heights)

# for information on rotated winds, see http://www2.mmm.ucar.edu/wrf/users/FAQ_files/Miscellaneous.html
@derived(
    description='x-wind component in earth-coordinates',
    units='m s-1',
)
def U_true(U, V, COSALPHA, SINALPHA):
    return COSALPHA * U - SINALPHA * V


@derived(
    description='y-wind component in earth-coordinates',
    units='m s-1',
)
def V_true(U, V, COSALPHA, SINALPHA):
    return SINALPHA * U + COSALPHA * V


@derived(
    description='10m x-wind component in earth-coordinates',
    units='m s-1',
)
def U10_true(U10, V10, COSALPHA, SINALPHA):
    return COSALPHA * U10 - SINALPHA * V10


@derived(
    description='10m y-wind component in earth-coordinates',
    units='m s-1',
)
def V10_true(U10, V10, COSALPHA, SINALPHA):
    return SINALPHA * U10 + COSALPHA * V10


@derived(
    units='m s-1',
    datatype=np.int16,
    scale_factor=0.01,  # ±327.68
)
def wind_speed(U, V):
    return np.sqrt(U ** 2 + V ** 2)


@derived(
    units='m s-1',
    datatype=np.int16,
    scale_factor=0.01,  # ±327.68
)
def wind_speed_10(U10, V10):
    return np.sqrt(U10 ** 2 + V10 ** 2)


@derived(
    description='wind direction',
    units='degrees',
    datatype=np.int16,
    scale_factor=0.1,  # ±3276.8
)
def wind_dir(U_true, V_true):
    return (270 - np.degrees(np.arctan2(V_true, U_true))) % 360


@derived(
    description='wind direction',
    units='degrees',
    datatype=np.int16,
    scale_factor=0.1,  # ±3276.8
)
def wind_dir_10(U10_true, V10_true):
    return (270 - np.degrees(np.arctan2(V10_true, U10_true))) % 360


@derived(
    units='m2 s-2',
    datatype=np.int16,
    scale_factor=4,
)
def geopotential(PH, PHB):
    return PH + PHB


@derived(
    units='m',
    datatype=np.int16,
    add_offset=16384,
    scale_factor=0.5,  # 0..32768
)
def geopotential_height(geopotential):
    return geopotential * g_inv


@derived(
    units='m',
    datatype=np.int16,
    add_offset=16384,
    scale_factor=0.5,  # 0..32768
)
def height_(geopotential_height, HGT):  # There is already height without the _ which gives the heights we intend to interpolate to
    return geopotential_height - HGT


@derived(
    units='Pa',
    datatype=np.int16,
    add_offset=65536,
    scale_factor=2,  # 0..131064
)
def pressure(P, PB):
    return P + PB


@derived(
    units='K',
    datatype=np.int16,
    scale_factor=0.1,  # ±3276.8
)
def potential_temperature(T):
    return T + 300.


@derived(
    units='K',
    datatype=np.int16,
    scale_factor=0.1,  # ±3276.8
)
def temperature(potential_temperature, pressure):
    return potential_temperature * (pressure / 100000.0) ** 0.2856


@derived(
    units='kg m-3',
    datatype=np.int16,
    scale_factor=0.0001,  # ±3.2768
)
def density(ALT):
    return 1/ALT


@derived(
    dimensions=('south_north', 'west_east'),
    description='Terrain Height',
    units='m',
    datatype=np.float32,  # Constant over time and compresses well
)
def HGT(HGT_M):
    """ Called if HGT is not present in input and works by renaming HGT_M from geo file. """
    return HGT_M


@derived(
    dimensions=('south_north', 'west_east'),
    description='LATITUDE, SOUTH IS NEGATIVE',
    units='degree_north',
    datatype=np.float32,  # Constant over time and compresses well
)
def XLAT(XLAT_M):
    """ Called if XLAT is not present in input and works by renaming XLAT_M from geo file. """
    return XLAT_M


@derived(
    dimensions=('south_north', 'west_east'),
    description='LONGITUDE, WEST IS NEGATIVE',
    units='degree_east',
    datatype=np.float32,  # Constant over time and compresses well
)
def XLONG(XLONG_M):
    """ Called if XLAT is not present in input and works by renaming XLONG_M from geo file. """
    return XLONG_M
