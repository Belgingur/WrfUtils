#!/usr/bin/env python
# encoding: utf-8

"""
Shrink wrfout files by reducing the number of digits for variables.
"""

import argparse
import datetime
import logging
import logging.config
import os
import socket
import sys
import time
from math import log10, ceil
from typing import List, Dict

import numpy as np
import yaml
from netCDF4 import Dataset, Variable

from utils import out_file_name, setup_logging, work_wrf_dates, TYPE_RANGE, CHUNK_SIZE_TIME, pick_chunk_sizes, \
    value_with_override, override_field

LOG = logging.getLogger('belgingur.drop_digits')


class Override(object):
    def __init__(self, datatype: str = None, scale_factor: float = None, add_offset: float = None):
        super(Override, self).__init__()
        self.add_offset = add_offset
        self.scale_factor = scale_factor
        self.data_type = datatype

        type_range = TYPE_RANGE[datatype]
        if type_range:
            sf = 1 if scale_factor is None else scale_factor
            ao = 0 if add_offset is None else add_offset
            self.range_min = type_range[0] * sf - ao
            self.range_max = type_range[1] * sf - ao
        else:
            self.range_min = None
            self.range_max = None

        # Calculate least significant digit for int/fixed point variables
        if self.data_type and self.data_type[0] in ('u', 'i') and self.scale_factor is not None:
            self.least_significant_digit = ceil(-log10(self.scale_factor))
        else:
            self.least_significant_digit = None

    def __repr__(self):
        s = self.data_type or 'unchanged'
        if self.scale_factor != 1 and self.scale_factor is not None:
            s += '*{:g}'.format(self.scale_factor)
        if self.add_offset != 0 and self.add_offset is not None:
            s += '{:s}{:g}'.format('+' if self.add_offset >= 0 else '', self.add_offset)
        # if self.range_min is not None:
        #    s += ' : {:g} .. {:g}'.format(self.range_min, self.range_max)
        return s


def configure() -> (argparse.Namespace, dict):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='DropDigits.yml',
                        help='Configuration to read (def: DropDigits.yml)')
    parser.add_argument('in_files', nargs="+",
                        help='wrf output files to process')
    args = parser.parse_args()

    LOG.info('Load config from %s', args.config)
    with open(args.config) as configFile:
        config = yaml.load(configFile)
        LOG.debug('Config: %s', config)

    return args, config


def build_overrides(config: Dict) -> Dict[str, Override]:
    """
    Iterate through config['overrides'] and make each entry into a valid Override object with defaults taken from
    override_defaults
    """
    specs = config['overrides']  # type: Dict[str, Dict[str]]
    overrides = {}
    for var_name, spec in specs.items():
        try:
            if isinstance(spec, str) and spec[0] == '@':
                spec = specs[spec[1:]]
            overrides[var_name] = Override(**spec)
        except:
            LOG.error('Failed to read override for %s', var_name)
            raise
    return overrides


def resolve_input_variables(in_ds: Dataset, config: Dict[str, None]):
    """
    Retrieves the variables from in_ds which we intend to copy to out_ds.
    """
    default_include = config.get('default_include', True)
    includes = config.get('include', [])  # type: List[str]
    excludes = config.get('exclude', [])  # type: List[str]

    if default_include:
        LOG.info('Include all variables except %s', excludes)
    else:
        LOG.info('Include selected variables %s', ', '.join(includes))

    in_vars = []  # type: List[Variable]
    included_names = []  # type: List[str]
    excluded_names = []  # type: List[str]
    for var_name, in_var in in_ds.variables.items():
        if (default_include and var_name not in excludes) or \
                (not default_include and var_name in includes):
            included_names.append(var_name)
            in_vars.append(in_var)
        else:
            excluded_names.append(var_name)
    LOG.debug('Included variables: \n%s', '\n'.join(map(str, in_vars)))
    LOG.info('Excluded variables: %s', ', '.join(excluded_names) or '<none>')

    if default_include:
        LOG.info('Included variables: %s', ', '.join(included_names))
    else:
        unseen_vars = [var_name for var_name in includes if var_name not in included_names]
        if unseen_vars:
            LOG.warning('Missing variables in include list: %s', ', '.join(unseen_vars))

    return in_vars


def create_output_variables(out_ds: Dataset, in_vars: List[Variable], overrides: Dict[str, Override],
                            comp_level: int, chunking: bool, max_k: int) -> List[Variable]:
    LOG.info('Create output variables with overrides:')
    out_vars = []
    default_override = overrides.get('default')
    for in_var in in_vars:
        var_name = in_var.name
        override = overrides.get(var_name, default_override)
        LOG.info('    %- 10s %- 10s %s..%s', var_name, override, override.range_min, override.range_max)

        data_type = value_with_override('datatype', override, in_var)
        chunk_sizes = pick_chunk_sizes(in_var, max_k) if chunking else None
        out_var = out_ds.createVariable(in_var.name,
                                        data_type,
                                        dimensions=in_var.dimensions,
                                        zlib=comp_level > 0,
                                        complevel=comp_level,
                                        shuffle=True,
                                        chunksizes=chunk_sizes)
        for field in (
                'description', 'least_significant_digit', 'scale_factor', 'add_offset',
                'FieldType', 'MemoryOrder', 'units', 'stagger', 'coordinates',
        ):
            override_field(out_var, field, override, in_var)
        out_vars.append(out_var)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, out_vars)))
    return out_vars


def add_attr(obj, name, value):
    """
    Find the first unset global attribute `name`, `name2`, etc. and set it to `value`
    """
    n = 0
    name_n = name
    value_n = getattr(obj, name_n, None)
    while value_n is not None:
        n += 1
        name_n = name + str(n)
        value_n = getattr(n, name_n, None)
    LOG.info('    %s = %s', name_n, value)
    setattr(obj, name_n, value)


def create_output_file(out_file_pattern: str, in_file: str, in_ds: Dataset,
                       custom_attributes: Dict[str, str]) -> (str, Dataset):
    """
    Creates a new dataset with the same attributes as an existing one plus additional
    attributes to trace the file's evolution. Copies the Times variable over verbatim
    if it exists.
    """
    out_file = out_file_name(in_file, out_file_pattern)
    LOG.info('Creating output file %s', out_file)
    if os.path.exists(out_file):
        logging.warning('Will overwrite existing file')
    out_ds = Dataset(out_file, mode='w', weakref=True)

    # Add some file meta-data
    LOG.info('Setting/updating global file attributes for output file')
    LOG.info('Copy %s attributes', len(in_ds.ncattrs()))
    for attr in in_ds.ncattrs():
        v = getattr(in_ds, attr)
        setattr(out_ds, attr, v)
        LOG.debug('    %s = %s', attr, v)
    LOG.info('Add attributes:')
    add_attr(out_ds, 'history', 'Converted with DropDigits.py at %s by %s on %s' % (
        datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%S'),
        os.getlogin(),
        socket.gethostname()
    ))
    add_attr(out_ds, 'source', in_file)
    for name, value in custom_attributes.items():
        add_attr(out_ds, name, value)
    out_ds.description = 'Reduced version of: %s' % (getattr(in_ds, 'description', in_file))
    LOG.info('    description = %s', out_ds.description)

    # Flush to disk
    out_ds.sync()
    return out_file, out_ds


def create_output_dimensions(in_ds:Dataset, in_vars:List[Variable], out_ds:Dataset, margin:int, sigma_limit:int):
    LOG.info('Add output dimensions:')
    need_dims = set()
    for in_var in in_vars:
        for dim_name in in_var.dimensions:
            need_dims.add(dim_name)

    included_names = []
    excluded_names = []
    for dim_name, in_dim in in_ds.dimensions.items():
        if dim_name in need_dims:
            size = None if in_dim.isunlimited() else in_dim.size
            if size and looks_planar(dim_name):
                size -= 2 * margin
            elif sigma_limit and dim_name == 'bottom_top':
                size = min(sigma_limit, size)
            elif sigma_limit and dim_name == 'bottom_top_stag':
                size = min(sigma_limit + 1, size)
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
def log_sigma_level_height(in_ds:Dataset, sigma_limit:int):
    """ Logs the minimum height of the highest sigma level above sea level and above surface. """
    if sigma_limit is not None:

        PH_ = in_ds.variables.get('PH')
        PHB = in_ds.variables.get('PHB')
        HGT = in_ds.variables.get('HGT')
        if PH_ is not None and PHB is not None and HGT is not None:
            np.set_printoptions(edgeitems=5, precision=0, linewidth=220)

            # De-stagger PH and PHB (average adjacent surfaces)
            # and convert to height (add PH and PHB and divide by g)
            # Building the _l arrays and adding the two levels is much faster than adding them directly from PH and PHB
            PH__l = PH_[:, sigma_limit - 1:sigma_limit + 1, :, :]
            PHB_l = PHB[:, sigma_limit - 1:sigma_limit + 1, :, :]
            Z_HGT = (PH__l[:, 0, :, :] + PH__l[:, 1, :, :] +
                     PHB_l[:, 0, :, :] + PHB_l[:, 1, :, :]) / (2 * 9.81)
            HGT0 = np.maximum(HGT, 0)  # Ignore ocean depth
            # noinspection PyTypeChecker
            height_asl = np.min(Z_HGT)
            height_agl = np.min(Z_HGT - HGT0)
            LOG.info('    3D variables limited to %d levels which reach at least '
                     '%0.0fm above sea level and %0.0fm above surface level',
                     sigma_limit, height_asl, height_agl)
        else:
            LOG.info('    3D variables limited to %d levels which reach an unknown height')


############################################################
# The main routine!

def main():
    setup_logging()

    args, config = configure()
    overrides = build_overrides(config)
    out_file_pattern = config.get('output_filename', './{filename}_reduced.nc4')
    total_start_time = time.time()
    total_errors = 0
    total_in_size = total_out_size = 0
    LOG.info('')
    for in_file in args.in_files:
        start_time = time.time()
        errors = 0

        # Open input datasets
        LOG.info('Opening input dataset %s', in_file)
        in_ds = Dataset(in_file, 'r')
        in_vars = resolve_input_variables(in_ds, config)

        # Convert time vector
        dates = work_wrf_dates(in_ds.variables['Times'])

        # Calculate spinup and margin
        LOG.info('Dimensional limits')
        dt = int((dates[-1] - dates[0]).total_seconds() / (len(dates) - 1) + 0.5)
        spinup_hours = config.get('spinup_hours', 0)
        spinup = int(spinup_hours * 3600. / dt + 0.5)
        LOG.info('    Spinup is %dh = %d steps', spinup_hours, spinup)
        margin = int(config.get('margin_cells', 0))
        LOG.info('    Margin is %d cells', margin)
        sigma_limit = config.get('sigma_limit', None)
        log_sigma_level_height(in_ds, sigma_limit)

        # Create empty output file
        custom_attributes = config.get('custom_attributes', dict())
        out_file, out_ds = create_output_file(out_file_pattern, in_file, in_ds, custom_attributes)
        create_output_dimensions(in_ds, in_vars, out_ds, margin, sigma_limit)
        chunking = config.get('chunking', False)
        comp_level = config.get('complevel', 0)
        out_vars = create_output_variables(out_ds, in_vars, overrides, comp_level, chunking, sigma_limit)

        # Start the loop through time
        LOG.info('Copying data in chunks of %s time steps', CHUNK_SIZE_TIME)
        for c_start in range(spinup, len(dates), CHUNK_SIZE_TIME):
            c_end = min(c_start + CHUNK_SIZE_TIME, len(dates))
            LOG.info('Chunk[%s..%s]: %s - %s', c_start, c_end - 1, dates[c_start], dates[c_end - 1])
            LOG.info('    Variable            Min          Max  Dimensions')
            if c_start > spinup and c_end - c_start != CHUNK_SIZE_TIME:
                LOG.info('Last chunk is short')
            for in_var, out_var in zip(in_vars, out_vars):
                in_chunk = in_var[c_start:c_end]
                dim_str = ', '.join(map(lambda x: '%s[%s]' % x, zip(in_var.dimensions, in_var.shape)))
                override = overrides.get(in_var.name) or overrides.get('default')

                if in_var.datatype == '|S1':
                    # Text data
                    LOG.info('    {:10}          N/A          N/A  {}'.format(in_var.name, dim_str))
                else:
                    # Numeric data
                    chunk_min, chunk_max = np.min(in_chunk), np.max(in_chunk)
                    LOG.info('    {:10} {:12,.2f} {:12,.2f}  {}'.format(in_var.name, chunk_min, chunk_max, dim_str))
                    if override.range_min is not None and override.range_max is not None:
                        sf = override.scale_factor  # Allow overlap of 1 scale factor to be truncated away
                        if chunk_min < override.range_min - sf or chunk_max > override.range_max + sf:
                            LOG.error(
                                '%s[%s..%s] values are %g .. %g outside valid range %g .. %g for %s',
                                in_var.name, c_start, c_end,
                                chunk_min, chunk_max,
                                override.range_min, override.range_max,
                                override)
                            errors += 1

                # Decide whether to limit the 3rd dimension. We need to have a 3rd dimension and a limit
                max_k = None
                if sigma_limit is not None:
                    if 'bottom_top' in in_var.dimensions:
                        max_k = sigma_limit
                    elif 'bottom_top_stag' in in_var.dimensions:
                        max_k = sigma_limit + 1

                # Copy chunk, but shift by spinup steps and cut off margin and sigma levels as appropriate
                if max_k is not None:
                    out_var[c_start - spinup:c_end - spinup] = \
                        in_chunk[..., 0:max_k, margin:-margin, margin:-margin]
                else:
                    out_var[c_start - spinup:c_end - spinup] = \
                        in_chunk[..., margin:-margin, margin:-margin]
                out_ds.sync()

        # Close our datasets
        in_ds.close()
        out_ds.close()

        # Print space saved and time used
        in_size = os.path.getsize(in_file)
        out_size = os.path.getsize(out_file)
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

    if len(args.in_files) > 1:
        total_out_percent = (100.0 * total_out_size / total_in_size)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_in_size / 1024.0,
            total_out_size / 1024.0,
            total_out_percent,
            time.time() - total_start_time,
        ))
        if total_errors:
            LOG.error('%d errors in total', total_errors)


if __name__ == '__main__':
    main()
