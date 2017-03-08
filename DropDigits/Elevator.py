#!/usr/bin/env python
# encoding: utf-8

"""
Shrink wrfout files by reducing the number of digits for variables.
"""

import argparse
import logging.config
import os
import time
from typing import List, Dict, Any

import numpy as np
import yaml
from netCDF4 import Dataset, Variable

from utils import out_file_name, setup_logging, work_wrf_dates, CHUNK_SIZE_TIME, pick_chunk_sizes, create_output_dataset

LOG = logging.getLogger('belgingur.drop_digits')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """


def configure() -> (argparse.Namespace, dict):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='Elevator.yml',
                        help='Configuration to read (def: Elevator.yml)')
    parser.add_argument('in_files', nargs="+",
                        help='wrf output files to process')
    args = parser.parse_args()

    LOG.info('Load config from %s', args.config)
    with open(args.config) as configFile:
        config = yaml.load(configFile)
        LOG.debug('Config: %s', config)

    return args, config


def resolve_input_variables(in_ds: Dataset, config: Dict[str, Any]):
    """
    Retrieves the variables from in_ds which we intend to copy to out_ds.
    """
    includes = config.get('variables', ['T', 'wind_speed', 'wind_dir'])
    in_vars = []
    included_names = []
    excluded_names = []
    for var_name, in_var in in_ds.variables.items():
        if var_name in includes:
            included_names.append(var_name)
            in_vars.append(in_var)
        else:
            excluded_names.append(var_name)
    LOG.debug('Included variables: \n%s', '\n'.join(map(str, in_vars)))
    LOG.info('Excluded variables: %s', ', '.join(excluded_names) or '<none>')

    unseen_vars = [var_name for var_name in includes if var_name not in included_names]
    if unseen_vars:
        LOG.warning('Missing variables in include list: %s', ', '.join(unseen_vars))

    return in_vars


def create_output_variables(out_ds: Dataset, in_vars: List[Variable],
                            comp_level: int, chunking: bool, max_k: int) -> List[Variable]:
    LOG.info('Create output variables with overrides:')
    out_vars = []
    for in_var in in_vars:
        var_name = in_var.name
        LOG.info('    %- 10s', var_name)

        data_type = in_var.datatype
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
            value = getattr(in_var, field, None)
            if value is not None:
                setattr(out_var, field, value)
        out_vars.append(out_var)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, out_vars)))
    return out_vars


def create_output_dimensions(in_ds: Dataset, in_vars: List[Variable], out_ds: Dataset, heights: List[int]):
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
            # TODO: if dim_name == 'bottom_top':
            #    # Reduce vertical dimension to the number of elevation levels
            #    size = len(heights)
            if dim_name == 'bottom_top_stag':
                # We will interpolate variables on this dimension to 'bottom_top'
                LOG.info('Dropping staggered vertical dimension %s', dim_name)
                continue
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


############################################################
# The main routine!

def main():
    setup_logging()

    args, config = configure()
    total_start_time = time.time()
    total_in_size = total_out_size = 0
    heights = config.get('heights', None)
    if not heights:
        LOG.error('No heights configured in config file')
        exit(1)
    LOG.info('Interpolate variables to %s', heights)
    LOG.info('')
    for in_file in args.in_files:
        start_time = time.time()
        out_file_pattern = config.get('output_filename', './{filename}_reduced.nc4')
        out_file = out_file_name(in_file, out_file_pattern)

        process_file(in_file, out_file, config=config)

        # Print space saved and time used
        in_size = os.path.getsize(in_file)
        out_size = os.path.getsize(out_file)
        total_in_size += in_size
        total_out_size += out_size
        out_percent = (100.0 * out_size / in_size)
        LOG.info('Size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:0.1f} s'.format(
            in_size / 1024.0,
            out_size / 1024.0,
            out_percent,
            time.time() - start_time
        ))
        LOG.info('')

    if len(args.in_files) > 1:
        total_out_percent = (100.0 * total_out_size / total_in_size)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_in_size / 1024.0,
            total_out_size / 1024.0,
            total_out_percent,
            time.time() - total_start_time,
        ))


def process_file(in_file: str, out_file: str, *, config: Dict[str, Any]):
    LOG.info('Opening input dataset %s', in_file)
    in_ds = Dataset(in_file, 'r')
    in_vars = resolve_input_variables(in_ds, config)
    dates = work_wrf_dates(in_ds.variables['Times'])
    heights = config.get('heights')  # type: List[int]

    custom_attributes = config.get('custom_attributes', dict())
    out_ds = create_output_dataset(out_file, in_file, in_ds, custom_attributes)
    create_output_dimensions(in_ds, in_vars, out_ds, heights)

    chunking = config.get('chunking', False)
    comp_level = config.get('complevel', 0)
    out_vars = create_output_variables(out_ds, in_vars, comp_level, chunking, len(heights))

    LOG.info('Copying data in chunks of %s time steps', CHUNK_SIZE_TIME)
    for c_start in range(0, len(dates), CHUNK_SIZE_TIME):
        c_end = min(c_start + CHUNK_SIZE_TIME, len(dates))
        LOG.info('Chunk[%s..%s]: %s - %s', c_start, c_end - 1, dates[c_start], dates[c_end - 1])
        LOG.info('    Variable            Min          Max  Dimensions')
        if c_start > 0 and c_end - c_start != CHUNK_SIZE_TIME:
            LOG.info('Last chunk is short')
        for in_var, out_var in zip(in_vars, out_vars):
            in_chunk = in_var[c_start:c_end]
            dim_str = ', '.join(map(lambda x: '%s[%s]' % x, zip(in_var.dimensions, in_var.shape)))

            if in_var.datatype == '|S1':
                # Text data
                LOG.info('    {:10}          N/A          N/A  {}'.format(in_var.name, dim_str))
            else:
                # Numeric data
                chunk_min, chunk_max = np.min(in_chunk), np.max(in_chunk)
                LOG.info('    {:10} {:12,.2f} {:12,.2f}  {}'.format(in_var.name, chunk_min, chunk_max, dim_str))

            out_var[c_start:c_end] = in_chunk[:]
            out_ds.sync()

    # Close our datasets
    in_ds.close()
    out_ds.close()
    return out_file


if __name__ == '__main__':
    main()
