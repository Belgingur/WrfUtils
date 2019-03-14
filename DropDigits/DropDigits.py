#!/usr/bin/env python3

"""
Shrink wrfout files by reducing the number of digits for variables.
"""

import argparse
import logging.config
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from math import log10, ceil
from typing import List, Dict, Any, Set
from typing import Union

import numpy as np
import yaml
from netCDF4 import Dataset, Variable, MFDataset

from utils import out_file_name, setup_logging, read_wrf_dates, TYPE_RANGE, CHUNK_SIZE_TIME, pick_chunk_sizes, \
    value_with_override, override_field, create_output_dataset, POINTLESS_TYPES, UNSUPPORTED_TYPES, LARGE_TYPES, SHORT_NAMES

LOG = logging.getLogger('belgingur.drop_digits')


@dataclass
class Override(object):
    datatype: str = None  # name must match Variable.datatype
    add_offset: float = None
    scale_factor: float = None
    is_default: bool = False

    DEFAULT = None

    def __post_init__(self):
        if self.datatype is not None:
            self.datatype = str(self.datatype)
        self.datatype = SHORT_NAMES.get(self.datatype, self.datatype)
        type_range = TYPE_RANGE.get(self.datatype)

        if type_range:
            sf = 1 if self.scale_factor is None else self.scale_factor
            ao = 0 if self.add_offset is None else self.add_offset
            self.range_min = type_range[0] * sf - ao
            self.range_max = type_range[1] * sf - ao
        else:
            self.range_min = None
            self.range_max = None

        # Calculate least significant digit for int/fixed point variables
        if self.datatype and self.datatype[0] in ('u', 'i') and self.scale_factor is not None:
            self.least_significant_digit = ceil(-log10(self.scale_factor))
        else:
            self.least_significant_digit = None

    def __repr__(self):
        s = self.datatype or 'unchanged'
        if self.scale_factor != 1 and self.scale_factor is not None:
            s += '*{:g}'.format(self.scale_factor)
        if self.add_offset != 0 and self.add_offset is not None:
            s += '{:s}{:g}'.format('+' if self.add_offset >= 0 else '', self.add_offset)
        # if self.range_min is not None:
        #    s += ' : {:g} … {:g}'.format(self.range_min, self.range_max)
        return s

    def of(self, other: Variable):
        return Override(
            datatype=value_with_override('datatype', self, other),
            add_offset=value_with_override('add_offset', self, other),
            scale_factor=value_with_override('scale_factor', self, other),
            is_default=self.is_default
        )


Override.DEFAULT = Override(is_default=True)
""" by default just pass variables through """


def configure() -> (argparse.Namespace, dict):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='DropDigits.yml',
                        help='Configuration to read (def: DropDigits.yml)')
    parser.add_argument('-f', '--fragments', default=False, action='store_true',
                        help='Process the input files as a single multi-file dataset')
    parser.add_argument('-s', '--sort-files', default=False, action='store_true',
                        help='Sort the input file names alphabetically before assembling')
    parser.add_argument('in_files', nargs='+',
                        help='wrf output files to process')
    args = parser.parse_args()

    LOG.info('Load config from %s', args.config)
    with open(args.config) as configFile:
        config = yaml.load(configFile)
        LOG.debug('Config: %s', config)

    return args, config


def arrange_in_files(args) -> List[Union[str, List[str]]]:
    # Creates a list of input data files, where each is either a single file or a list of files.
    files = args.in_files
    if args.sort_files:
        files = sorted(files)
    if args.fragments:
        files = [files]
    return files


def build_overrides(config: Dict) -> Dict[str, Override]:
    """
    Iterate through config['overrides'] and make each entry into a valid Override object with defaults taken from
    override_defaults
    """
    specs: Dict[str, Union[Dict[str], str]] = config['overrides']
    overrides = {}
    for var_name, spec in specs.items():
        try:
            if isinstance(spec, str) and spec[0] == '@':
                spec = specs[spec[1:]]
            override = Override(**spec)
            overrides[var_name] = override
            if override.datatype in POINTLESS_TYPES:
                LOG.warning('%s uses type %s which is as just large as f4', var_name, override.datatype)
            if override.datatype in UNSUPPORTED_TYPES:
                LOG.warning('%s uses type %s which is not supported by NETCDF4_CLASSIC', var_name, override.datatype)
        except:
            LOG.error('Failed to read override for %s', var_name)
            raise
    return overrides


def resolve_input_variables(in_ds: Dataset, config: Dict[str, Any]) -> List[str]:
    """
    Retrieves the names of variables from in_ds which we intend to copy to out_ds.
    """
    default_include = config.get('default_include', True)
    includes: List[str] = config.get('include', [])
    excludes: List[str] = config.get('exclude', [])

    if default_include:
        LOG.info('Include all variables except %s', excludes)
    else:
        LOG.info('Include selected variables %s', ', '.join(includes))

    included_names: List[str] = []
    excluded_names: List[str] = []
    for var_name, in_var in in_ds.variables.items():
        if (default_include and var_name not in excludes) or \
                (not default_include and var_name in includes):
            included_names.append(var_name)
        else:
            excluded_names.append(var_name)
    LOG.debug('Included variables: %s', ', '.join(included_names) or '<none>')
    LOG.info('Excluded variables: %s', ', '.join(excluded_names) or '<none>')

    if default_include:
        LOG.info('Included variables: %s', ', '.join(included_names))
    else:
        unseen_vars = [var_name for var_name in includes if var_name not in included_names]
        if unseen_vars:
            LOG.warning('Missing variables in include list: %s', ', '.join(unseen_vars))

    return included_names


def create_output_variables(
        in_ds_0: Dataset, out_ds: Dataset,
        var_names: List[str], overrides: Dict[str, Override],
        comp_level: int, chunking: bool, max_t: int, max_k: int,
        high_dimensional_floats: Set[str]
) -> List[Variable]:
    LOG.info('Create output variables with overrides:')
    out_vars = []
    for var_name in var_names:
        in_var_0 = in_ds_0.variables[var_name]
        override = overrides.get(var_name, Override.DEFAULT)
        combined = override.of(in_var_0)
        dims_string = ','.join(in_var_0.dimensions)

        expensive = (len(in_var_0.dimensions) >= 4) and (combined.datatype in LARGE_TYPES) and (combined.is_default)
        LOG.log(
            logging.WARNING if expensive else logging.INFO,
            '    %- 10s %- 10s %s … %s [%s]',
            var_name, combined, combined.range_min, combined.range_max, dims_string
        )
        if expensive:
            high_dimensional_floats.add(f'{var_name:10s} {combined.datatype:8s} [{dims_string}]')

        chunk_sizes = None
        if chunking:
            chunk_sizes = pick_chunk_sizes(out_ds, in_var_0.dimensions, max_t=max_t, max_k=max_k)
        out_var = out_ds.createVariable(var_name,
                                        combined.datatype,
                                        dimensions=in_var_0.dimensions,
                                        zlib=comp_level > 0,
                                        complevel=comp_level,
                                        shuffle=True,
                                        chunksizes=chunk_sizes)
        for field in (
                'description', 'least_significant_digit', 'scale_factor', 'add_offset',
                'FieldType', 'MemoryOrder', 'units', 'stagger', 'coordinates',
        ):
            override_field(out_var, field, combined, in_var_0)
        out_vars.append(out_var)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, out_vars)))
    return out_vars


def create_output_dimensions(in_ds: Dataset, var_names: List[str], out_ds: Dataset,
                             margin: int, max_t: int, max_k: int = None):
    LOG.info('Add output dimensions:')
    need_dims = set()
    for var_name in var_names:
        in_var = in_ds.variables[var_name]
        for dim_name in in_var.dimensions:
            need_dims.add(dim_name)

    included_names = []
    excluded_names = []
    for dim_name, in_dim in in_ds.dimensions.items():
        if dim_name in need_dims:
            if dim_name == 'Time':
                size = max_t
            elif in_dim.isunlimited():
                size = None
            else:
                size = in_dim.size
            if size and looks_planar(dim_name):
                size -= 2 * margin
            elif max_k and dim_name == 'bottom_top':
                size = min(max_k, size)
            elif max_k and dim_name == 'bottom_top_stag':
                size = min(max_k + 1, size)
            LOG.info('    %s (%s)', dim_name, size or 'unlimited')
            out_ds.createDimension(dim_name, size)
            included_names.append(dim_name)
        else:
            excluded_names.append(dim_name)

    # LOG.info('Included dimensions: %s', ', '.join(included_names))
    LOG.info('Excluded dimensions: %s', ', '.join(excluded_names) or '<none>')


def looks_planar(dim_name: str):
    """
    Determines whether a dimension with the given name is a planar dimension, i.e. east/west or north/south
    """
    return dim_name and \
           'south' in dim_name or 'north' in dim_name or \
           'west' in dim_name or 'east' in dim_name


# noinspection PyPep8Naming
def log_sigma_level_height(in_ds: Dataset, max_k: int = None):
    """ Logs the minimum height of the highest sigma level above sea level and above surface. """

    if max_k is None:
        # Nothing to do
        return

    PH_ = in_ds.variables.get('PH')
    PHB = in_ds.variables.get('PHB')
    HGT = in_ds.variables.get('HGT')
    if PH_ is not None and PHB is not None and HGT is not None:
        np.set_printoptions(edgeitems=5, precision=0, linewidth=220)

        # De-stagger PH and PHB (average adjacent surfaces)
        # and convert to height (add PH and PHB and divide by g)
        # Building the _l arrays and adding the two levels is much faster than adding them directly from PH and PHB
        PH__l = PH_[:, max_k - 1:max_k + 1, :, :]
        PHB_l = PHB[:, max_k - 1:max_k + 1, :, :]
        Z_HGT = (PH__l[:, 0, :, :] + PH__l[:, 1, :, :] +
                 PHB_l[:, 0, :, :] + PHB_l[:, 1, :, :]) / (2 * 9.81)
        HGT0 = np.maximum(HGT, 0)  # Ignore ocean depth
        # noinspection PyTypeChecker
        height_asl = np.min(Z_HGT)
        height_agl = np.min(Z_HGT - HGT0)
        LOG.info('    3D variables limited to %d levels which reach at least '
                 '%0.0fm above sea level and %0.0fm above surface level',
                 max_k, height_asl, height_agl)
    else:
        LOG.info('    3D variables limited to %d levels which reach an unknown height')


############################################################
# The main routine!

def main():
    setup_logging()

    args, config = configure()
    overrides = build_overrides(config)
    out_file_pattern = config.get('output_filename', './{filename}_reduced.nc4')
    in_files = arrange_in_files(args)
    total_start_time = time.time()
    total_errors = 0
    total_in_size = total_out_size = 0
    LOG.info('')
    for in_file in in_files:
        start_time = time.time()
        out_file = out_file_name(in_file, out_file_pattern)

        errors = process_file(in_file, out_file, config=config, overrides=overrides)

        # Print space saved and time used
        in_size = file_size(in_file)
        out_size = file_size(out_file)
        total_errors += errors
        total_in_size += in_size
        total_out_size += out_size
        out_percent = (100.0 * out_size / in_size)
        LOG.info('Size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:0.1f} s'.format(
            in_size / 1024.0,
            out_size / 1024.0,
            out_percent,
            time.time() - start_time
        ))
        if errors:
            LOG.error('%d errors in file', errors)
            sys.exit(1)
        LOG.info('')

    if len(in_files) > 1:
        total_out_percent = (100.0 * total_out_size / total_in_size)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_in_size / 1024.0,
            total_out_size / 1024.0,
            total_out_percent,
            time.time() - total_start_time,
        ))
        if total_errors:
            LOG.error('%d errors in total', total_errors)


def file_size(files: Union[str, List[str]]):
    """ Return the size of a file or cumulative size of a list of files. """
    if isinstance(files, list):
        return sum(os.path.getsize(file) for file in files)
    else:
        return os.path.getsize(files)


def open_dataset(in_file):
    # If we have multiple input files, make in_ds be a MFDataset.
    # and in_ds_0 be a Dataset for the first file to get more meta-data
    if isinstance(in_file, list):
        if len(in_file) > 1:
            in_ds = MFDataset(in_file, 'r')
            in_ds_0 = Dataset(in_file[0])
            return in_ds, in_ds_0, in_file[0]

        in_file = in_file[0]

    # If we have a single input file, make in_ds and in_ds_0 be the Dataset for that file
    in_ds = Dataset(in_file, 'r')
    return in_ds, in_ds, in_file


def count_time_steps(in_ds):
    var_t: Variable = in_ds.variables.get('Times')
    idx_t = var_t.dimensions.index('Time')
    max_t = var_t.shape[idx_t]
    return max_t


def process_file(
        in_file: Union[str, List[str]], out_file: str, *,
        config: Dict[str, Any], overrides: Dict[str, Override]
) -> int:
    LOG.info('Opening input dataset %s', in_file)
    errors = 0

    in_ds, in_ds_0, nominal_infile = open_dataset(in_file)

    var_names: List[str] = resolve_input_variables(in_ds, config)
    dates: List[datetime] = read_wrf_dates(in_ds)

    LOG.info('Dimensional limits')
    if len(dates) > 1:
        dt = int((dates[-1] - dates[0]).total_seconds() / (len(dates) - 1) + 0.5)
    else:
        dt = 3600
    spinup_hours = config.get('spinup_hours', 0)
    spinup = int(spinup_hours * 3600. / dt + 0.5)
    LOG.info('    Spinup is %dh = %d steps', spinup_hours, spinup)
    margin = int(config.get('margin_cells', 0))
    LOG.info('    Margin is %d cells', margin)
    max_k: Union[int, None] = config.get('sigma_limit', None)
    log_sigma_level_height(in_ds, max_k)
    max_t = count_time_steps(in_ds)

    custom_attributes = config.get('custom_attributes', dict())
    out_ds = create_output_dataset(out_file, nominal_infile, in_ds, custom_attributes)
    create_output_dimensions(in_ds, var_names, out_ds, margin, max_t, max_k)
    chunking = config.get('chunking', False)
    comp_level = config.get('complevel', 0)
    high_dimensional_floats: Set[str] = set()
    out_vars = create_output_variables(
        in_ds_0, out_ds,
        var_names, overrides,
        comp_level, chunking,
        max_t, max_k,
        high_dimensional_floats
    )

    LOG.info('Copying data in chunks of %s time steps', CHUNK_SIZE_TIME)
    for c_start in range(spinup, len(dates), CHUNK_SIZE_TIME):
        c_end = min(c_start + CHUNK_SIZE_TIME, len(dates))
        LOG.info('Chunk[%s:%s]: %s - %s', c_start, c_end, dates[c_start], dates[c_end - 1])
        LOG.info('    Variable            Min          Max')
        if c_start > spinup and c_end - c_start != CHUNK_SIZE_TIME:
            LOG.info('Last chunk is short')

        # Loop through variables
        for out_var in out_vars:
            var_name = out_var.name
            in_var = in_ds.variables[var_name]
            in_var_0 = in_ds_0.variables[var_name]

            # Decide whether to limit the 3rd dimension. We need to have a 3rd dimension and a limit
            var_max_k: int = None
            if max_k is not None:
                if 'bottom_top' in in_var.dimensions:
                    var_max_k = max_k
                elif 'bottom_top_stag' in in_var.dimensions:
                    var_max_k = (max_k or 0) + 1  # silly syntax to make PyCharm not complain about None

            # Carve out a chunk of input variable that we want to copy
            if var_max_k is not None:
                max_j, max_i = in_var.shape[-2:]
                in_chunk: np.ndarray = in_var[c_start:c_end, 0:var_max_k, margin:max_j - margin, margin:max_i - margin]
            elif len(in_var.shape) >= 3:
                max_j, max_i = in_var.shape[-2:]
                in_chunk = in_var[c_start:c_end, ..., margin:max_j - margin, margin:max_i - margin]
            else:
                in_chunk = in_var[c_start:c_end]
            out_var[c_start - spinup:c_end - spinup] = in_chunk
            out_ds.sync()

            # Log variable dimensions and sanity check
            override = overrides.get(var_name, Override.DEFAULT)
            if in_var_0.datatype == '|S1':
                # Text data
                LOG.info(f'    {var_name:10}          N/A          N/A')
            else:
                # Numeric data
                chunk_min, chunk_max = np.min(in_chunk), np.max(in_chunk)
                LOG.info(f'    {var_name:10} {chunk_min:12,.2f} {chunk_max:12,.2f}')
                if override.range_min is not None and override.range_max is not None:
                    sf = override.scale_factor or 0  # Allow overlap of 1 scale factor to be truncated away
                    if chunk_min < override.range_min - sf or chunk_max > override.range_max + sf:
                        LOG.error(
                            '%s[%s…%s] values are %g … %g outside valid range %g … %g for %s',
                            in_var.name, c_start, c_end,
                            chunk_min, chunk_max,
                            override.range_min, override.range_max,
                            override)
                        errors += 1

    # Close our datasets
    out_ds.close()
    in_ds.close()
    if in_ds_0 is not in_ds:
        in_ds_0.close()

    if high_dimensional_floats:
        LOG.warning('')
        LOG.warning('Implicit expensive high-dimensional variables encountered:')
        for var_name in sorted(high_dimensional_floats):
            LOG.warning('  %s', var_name)
        LOG.warning('Consider converting them to i2 or explicitly configure them for f4')
        LOG.warning('')

    return errors


if __name__ == '__main__':
    try:
        main()
    except:
        LOG.exception('Uncaught exception in main()')
        raise
