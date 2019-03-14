import datetime
import getpass
import logging
import logging.config
import os
import socket
import sys
from pathlib import Path
from typing import Dict, Any, Iterator, Iterable
from typing import List, Tuple
from typing import Union

import numpy as np
import yaml
from netCDF4 import Dataset, Variable, Dimension

LOG = logging.getLogger('belgingur.utils')

g = 9.81  # [m/s²]
""" Gravitational acceleration on Earth """

g_inv = 1. / g  # [s²/m]
""" Inverse Gravity """

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

# Dimension names
DIM_TIME = 'Time'
DIM_BOTTOM_TOP = 'bottom_top'
DIM_BOTTOM_TOP_STAG = 'bottom_top_stag'
DIM_WEST_EAST_STAG = 'west_east_stag'
DIM_SOUTH_NORTH_STAG = 'south_north_stag'

# We work with temporal data in chunks of this many steps
CHUNK_SIZE_TIME = 128

# Mapping from number of dimensions to size of chunk of that many dimensions
CHUNK_SIZES: List[Tuple[int, ...]] = [
    (CHUNK_SIZE_TIME,),
    (CHUNK_SIZE_TIME, 19),
    (CHUNK_SIZE_TIME, 16, 16),
    (CHUNK_SIZE_TIME, 10, 16, 16)
]

SHORT_NAMES = dict(
    uint8='u1',
    uint16='u2',
    uint32='u4',
    uint64='u8',

    int8='i1',
    int16='i2',
    int32='i4',
    int64='i8',

    float32='f4',
    float64='f8',
)

# Range of values allowed by each netcdf type
TYPE_RANGE: Dict[Union[str, None], Tuple[int, int]] = dict(
    u1=(0, 2 ** 8 - 1),
    u2=(0, 2 ** 16 - 1),
    u4=(0, 2 ** 32 - 1),
    u8=(0, 2 ** 64 - 1),

    i1=(-2 ** 7, 2 ** 7 - 1),
    i2=(-2 ** 15, 2 ** 15 - 1),
    i4=(-2 ** 31, 2 ** 31 - 1),
    i8=(-2 ** 63, 2 ** 63 - 1),

    f4=(-3.4e38, +3.4e38),
    f8=(-1.79e308, +1.79e308),

    S1=None,
)
TYPE_RANGE[None] = None

# Names of types which are no smaller than the default f4
POINTLESS_TYPES = {'u4', 'u8', 'i4', 'i8', 'f8', }

# Names of types which are not supported by NETCDF4_CLASSIC
UNSUPPORTED_TYPES = {'u1', 'u2', 'u4', 'u8'}

# Types to avoid in high-dimension variables
LARGE_TYPES = POINTLESS_TYPES.union({'f4', 'float32'})


def memoize(f):
    """ Very simple memoization decorator """
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


def pick_chunk_sizes(out_ds: Dataset, dimensions: List[str], *, max_t: int=None, max_k: int = None) -> Tuple[int, ...]:
    """
    Given a variable, pick the appropriate chunk sizes to apply to it given it's shape
    """

    def adjust_sizes(unadjusted: Iterable[int]) -> Iterator[int]:
        for cs, dim_name in zip(unadjusted, dimensions):
            dim: Dimension = out_ds.dimensions[dim_name]
            if dim_name in (DIM_TIME,) and max_t is not None:
                cs = min(cs, max_t)
            elif dim.size is not None:
                cs = min(cs, dim.size)
            if dim_name in (DIM_BOTTOM_TOP, DIM_BOTTOM_TOP_STAG) and max_k is not None:
                cs = min(cs, max_k)
            yield cs

    if len(dimensions) > len(CHUNK_SIZES):
        raise IndexError('We can only deal with variables or {} dimensions or less', len(CHUNK_SIZES))
    unadjusted = CHUNK_SIZES[len(dimensions) - 1]
    adjusted = tuple(adjust_sizes(unadjusted))
    return adjusted


def read_wrf_dates(in_ds: Dataset) -> np.ndarray:
    """ Convert the WRF-style Times array from list of strings to a list of datetime objects. """
    times: Variable = in_ds.variables['Times']
    dates = []
    for b in times[:]:
        s = b.tostring().decode()
        d = datetime.datetime.strptime(s, '%Y-%m-%d_%H:%M:%S')
        dates.append(d)
    dates = np.array(dates)
    return dates


def setup_logging(config_path: Path = Path(__file__).parent / 'logging.yml'):
    with config_path.open() as configFile:
        logging_config = yaml.load(configFile)
        logging.config.dictConfig(logging_config)
        LOG.info('Configured logging from %s', config_path)


def out_file_name(in_file: str, outfile_pattern: str) -> str:
    """
    Creates an output file name from an input file name and a pattern. The pattern can contain these variables:
     - basename The name of the input file without extension or path
     - ext      The file extension excluding dot (ex: '.nc4') (ex: '')
     - path     The path to the input file (ex: '../runtime') (ex: '.')
    """
    nominal_in_file = in_file[0] if isinstance(in_file, list) else in_file
    in_path, in_base = os.path.split(nominal_in_file)
    in_base, in_ext = os.path.splitext(in_base)
    if in_ext: in_ext = in_ext[1:]  # Cut the dot
    if not in_path: in_path = '.'  # No path means current dir
    outfile = outfile_pattern.format(
        path=in_path,
        basename=in_base,
        ext=in_ext,
    )
    return outfile


def getuser():
    try:
        return getpass.getuser()
    except:
        return 'unknown user'


def gethostname():
    try:
        return socket.gethostname()
    except:
        return 'unknown host'


def create_output_dataset(out_file: str, in_file: str, in_ds: Dataset,
                          custom_attributes: Dict[str, str], app_name: str = None) -> (str, Dataset):
    """
    Creates a new dataset with the same attributes as an existing one plus additional
    attributes to trace the file's evolution. Copies the Times variable over verbatim
    if it exists.

    :param out_file: The path to the file to create
    :param in_file: The pth to the file to say the out_file is created from
    :param in_ds: The Dataset to copy attributes from
    :param custom_attributes: Dictionary of attributes to add to the file
    :param app_name: Name of application to say converted the file. Defaults to filename of root script.
    """
    LOG.info('Creating output file %s', out_file)
    if os.path.exists(out_file):
        logging.warning('Will overwrite existing file')
    out_ds = Dataset(
        out_file,
        mode='w',
        weakref=True,
        format='NETCDF4_CLASSIC'  # NC4C supports chunking and can be read by MFDataset
    )

    if not app_name:
        app_name = Path(sys.argv[0]).name

    # Add some file meta-data
    LOG.info('Setting/updating global file attributes for output file')
    LOG.info('Copy %s attributes', len(in_ds.ncattrs()))
    for attr in in_ds.ncattrs():
        v = getattr(in_ds, attr)
        setattr(out_ds, attr, v)
        LOG.debug('    %s = %s', attr, v)
    LOG.info('Add attributes:')
    date_str = datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%S')
    add_attr(out_ds, 'history', f'Converted with {app_name} at {date_str} by {getuser()} on {gethostname()}')
    add_attr(out_ds, 'source', in_file)
    for name, value in custom_attributes.items():
        add_attr(out_ds, name, value)
    out_ds.description = f'Reduced version of: {getattr(in_ds, "description", in_file)}'
    LOG.info('    description = %s', out_ds.description)

    # Flush to disk
    out_ds.sync()
    return out_ds


def add_attr(obj: Any, name: str, value: Any):
    """ Find the first unset attribute on `obj` called `name`, `name2`, etc. and set it to `value`. """
    n = 0
    name_n = name
    value_n = getattr(obj, name_n, None)
    while value_n is not None:
        LOG.info('    %s = %s', name_n, value_n)
        n += 1
        name_n = name + str(n)
        value_n = getattr(n, name_n, None)
    LOG.info('    %s = %s', name_n, value)
    setattr(obj, name_n, value)


def value_with_override(name: str, override, in_obj, default=None):
    """
    Returns the first of these found with a non-None value:
    - override.name
    - in_obj.name
    - default
    """
    value = getattr(override, name, None)
    if value is None:
        value = getattr(in_obj, name, None)
    if value is None:
        value = default
    return value


def override_field(out_obj, name, override, in_obj):
    """
    Overrides a named field in out_obj with the first value found in:
    - override.name
    - in_obj.name
    """
    value = value_with_override(name, override, in_obj, __EMPTY__)
    if value is not __EMPTY__:
        setattr(out_obj, name, value)


def destagger_array_by_dim(a: np.ndarray, dims: List[str], dim: str, *, log_indent=0) -> np.ndarray:
    """
    Destagger an array over a specified dimension if available
    :param a: The array to de-stagger
    :param dims: The list of dimensions that a spans
    :param dim: The name of the dimension to destagger along, if it is in dims
    """
    if dim not in dims:
        return a

    #LOG.info('%sdestagger on: %s', log_indent * ' ', dim)
    axis = dims.index(dim)
    return destagger_array(a, axis)


def destagger_array(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Staggers/destaggers an array along a given axis.
    """
    dims = len(a.shape)
    slices0 = tuple(
        slice(0, -1) if i == axis else slice(None)
        for i in range(dims)
    )
    slices1 = tuple(
        slice(1, None) if i == axis else slice(None)
        for i in range(dims)
    )
    a0 = a.__getitem__(slices0)
    a1 = a.__getitem__(slices1)
    aa = 0.5 * a0 + 0.5 * a1
    return aa
