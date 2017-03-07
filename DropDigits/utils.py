import datetime
import logging
import os
import sys
from typing import Dict
from typing import List, Tuple
from typing import Union

import numpy as np
import yaml
from netCDF4 import Variable

LOG = logging.getLogger('belgingur.utils')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

# We work with temporal data in chunks of this many steps
CHUNK_SIZE_TIME = 128

# Mapping from number of dimensions to size of chunk of that many dimensions
CHUNK_SIZES = [
    (CHUNK_SIZE_TIME,),
    (CHUNK_SIZE_TIME, 19),
    (CHUNK_SIZE_TIME, 16, 16),
    (CHUNK_SIZE_TIME, 10, 16, 16)
]  # type: List[Tuple[int]]

# Range of values allowed by each netcdf type
TYPE_RANGE = dict(
    u1=(0, 2 ** 8 - 1),
    u2=(0, 2 ** 16 - 1),
    u4=(0, 2 ** 32 - 1),
    u8=(0, 2 ** 64 - 1),

    i1=(-2 ** 7, 2 ** 7 - 1),
    i2=(-2 ** 15, 2 ** 15 - 1),
    i4=(-2 ** 31, 2 ** 31 - 1),
    i8=(-2 ** 63, 2 ** 63 - 1),

    f4=(-3.4e38, +3.4e38),
    f8=(-1.79e308, +1.79e308)
)  # type: Dict[Union[str, None],[int,int]]
TYPE_RANGE[None] = None


def pick_chunk_sizes(var: Variable, max_k: int = None) -> Tuple[int]:
    """
    Given a variable, pick the appropriate chunk sizes to apply to it given it's shape
    """
    in_shape = var.shape
    chunk_sizes = CHUNK_SIZES[len(in_shape) - 1]
    if max_k is not None and len(chunk_sizes) >= 4:
        # Don't make a chunk taller than the dimension
        chunk_sizes = tuple(s if i != 1 else max_k for i, s in enumerate(chunk_sizes))
    return chunk_sizes


def work_wrf_dates(times: List[str]) -> np.ndarray:
    """ Convert the WRF-style Times array from list of strings to a list of datetime objects. """
    dates = []
    for t in times[:]:
        tt = t.tostring()
        if sys.version_info >= (3, 0):
            tt = tt.decode()
        dates.append(datetime.datetime.strptime(tt, '%Y-%m-%d_%H:%M:%S'))
    dates = np.array(dates)
    return dates


def setup_logging(config_path: str = os.path.join(os.path.dirname(__file__), './logging.yml')):
    with open(config_path) as configFile:
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
    in_path, in_base = os.path.split(in_file)
    in_base, in_ext = os.path.splitext(in_base)
    if in_ext: in_ext = in_ext[1:]  # Cut the dot
    if not in_path: in_path = '.'  # No path means current dir
    outfile = outfile_pattern.format(
        path=in_path,
        basename=in_base,
        ext=in_ext,
    )
    return outfile


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
