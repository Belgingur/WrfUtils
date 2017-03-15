#!/usr/bin/env python
# encoding: utf-8

"""
Interpolate 3D variables to certain elevations above ground or sea-level.
Also de-staggers all variables horizontally.
"""

import argparse
import logging.config
import os
import time
from typing import List, Dict, Any

import numpy as np
import yaml
from netCDF4 import Dataset, Variable

from calculators import ChunkCalculator
from utils import out_file_name, setup_logging, read_wrf_dates, CHUNK_SIZE_TIME, pick_chunk_sizes, \
    create_output_dataset, g_inv, DIM_BOTTOM_TOP, DIM_BOTTOM_TOP_STAG
from vertical_interpolation import build_interpolators

LOG = logging.getLogger('belgingur.elevator')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

np.set_printoptions(4, edgeitems=3, linewidth=200)


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


def resolve_input_variables(in_ds: Dataset, out_var_names: List[str]) -> List[Variable]:
    """
    Retrieves the variables from in_ds which we intend to copy to out_ds or use to calculate derived variables.
    """
    in_vars = []
    ds_vars = in_ds.variables  # type: Dict[str, Variable]
    for in_var_name in out_var_names:
        in_var = ds_vars[in_var_name]
        in_vars.append(in_var)
    return in_vars


def resolve_input_dimensions(in_vars: List[Variable]) -> List[str]:
    """ Return the list of names of dimensions used by the given list of dimensions """
    in_dim_names = []  # type: List[str]
    for in_var in in_vars:
        for in_dim in in_var.dimensions:
            if in_dim not in in_dim_names:
                in_dim_names.append(in_dim)
    LOG.info('Included dimensions: %s', ', '.join(map(str, in_dim_names)))
    return in_dim_names


def resolve_output_dimensions(in_dim_names: List[str]) -> List[str]:
    """ Given a list of dimensions, create the list of matching de-staggered dimensions with duplicates removed. """
    out_dim_names = []  # type: List[str]
    for in_dim_name in in_dim_names:
        out_dim_name = destagger_dim_name(in_dim_name)
        if out_dim_name not in out_dim_names:
            out_dim_names.append(out_dim_name)
    return out_dim_names


def destagger_dim_name(in_dim_name: str):
    """ Given a dimension name, returns the name of the un-staggered dimension. """
    if in_dim_name.endswith('_stag'):
        return in_dim_name[:-len('_stag')]
    else:
        return in_dim_name


def create_output_variables(in_ds: Dataset, out_ds: Dataset, out_var_names: List[str],
                            comp_level: int, chunking: bool, elevation_limit: int) -> List[Variable]:
    LOG.info('Create output variables with:')
    out_vars = []
    for var_name in out_var_names:
        LOG.info('    %- 10s', var_name)
        in_var = in_ds.variables[var_name]  # type: Variable
        in_dims = in_var.dimensions  # type: List[str]
        out_dims = [destagger_dim_name(d) for d in in_dims]
        chunk_sizes = pick_chunk_sizes(in_var, elevation_limit) if chunking else None
        out_var = out_ds.createVariable(in_var.name,
                                        in_var.datatype,
                                        dimensions=out_dims,
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


def create_output_dimensions(in_ds: Dataset, out_ds: Dataset, out_dim_names: List[str], max_k: int):
    LOG.info('Add output dimensions:')
    for dim_name in out_dim_names:
        in_dim = in_ds.dimensions[dim_name]
        size = None if in_dim.isunlimited() else in_dim.size
        if dim_name == DIM_BOTTOM_TOP:
            # Reduce vertical dimension to the number of elevation levels
            size = max_k
        LOG.info('    %s (%s)', dim_name, size or 'unlimited')
        out_ds.createDimension(dim_name, size)

############################################################
# The main routine!

def main():
    setup_logging()

    args, config = configure()
    total_start_time = time.time()
    total_in_size = total_out_size = 0
    heights = config.get('heights')
    above_ground = bool(config.get('above_ground'))
    LOG.info('Interpolate variables to %sm above %s',
             'm, '.join(map(str, heights)),
             'ground' if above_ground else 'sea-level')
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
        out_percent = (100 * out_size / in_size)
        LOG.info('Size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:0.1f} s'.format(
            in_size / 1024,
            out_size / 1024,
            out_percent,
            time.time() - start_time
        ))
        LOG.info('')

    if len(args.in_files) > 1:
        total_out_percent = (100 * total_out_size / total_in_size)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_in_size / 1024,
            total_out_size / 1024,
            total_out_percent,
            time.time() - total_start_time,
        ))


def process_file(in_file: str, out_file: str, *, config: Dict[str, Any]):
    LOG.info('Opening input dataset %s', in_file)
    in_ds = Dataset(in_file, 'r')
    out_var_names = config.get('variables')
    in_vars = resolve_input_variables(in_ds, out_var_names)
    in_dim_names = resolve_input_dimensions(in_vars)
    out_dim_names = resolve_output_dimensions(in_dim_names)
    dates = read_wrf_dates(in_ds)
    heights = config.get('heights')  # type: List[int]
    above_ground = bool(config.get('above_ground'))
    custom_attributes = config.get('custom_attributes', dict())

    out_ds = create_output_dataset(out_file, in_file, in_ds, custom_attributes)
    create_output_dimensions(in_ds, out_ds, out_dim_names, len(heights))

    chunking = config.get('chunking', False)
    comp_level = config.get('complevel', 0)
    create_output_variables(in_ds, out_ds, out_var_names, comp_level, chunking, len(heights))

    chunk_size = CHUNK_SIZE_TIME // 4
    LOG.info('Processing data in chunks of %s time steps', chunk_size)
    for t_start in range(0, len(dates), chunk_size):
        t_end = min(t_start + chunk_size, len(dates))
        LOG.info('Chunk[%s:%s]: %s - %s', t_start, t_end, dates[t_start], dates[t_end - 1])

        need_aligned = DIM_BOTTOM_TOP in in_dim_names
        need_staggered = DIM_BOTTOM_TOP_STAG in in_dim_names
        z_stag = calc_z_stag(in_ds, t_start, t_end, above_ground)
        ipor_alig, ipor_stag = build_interpolators(z_stag, heights, need_aligned, need_staggered)
        cc = ChunkCalculator(in_ds, t_start, t_end, ipor_alig, ipor_stag)

        LOG.info('Processing Variable     Input Dimensions')
        for out_var_name in out_var_names:
            LOG.info('    %s', out_var_name)
            out_ds.variables[out_var_name][t_start:t_end] = cc(out_var_name)
            out_ds.sync()

    # Close our datasets
    in_ds.close()
    out_ds.close()
    return out_file


def calc_z_stag(ds: Dataset, t_start: int, t_end: int, above_ground: bool):
    PH = ds.variables['PH'][t_start:t_end]
    PHB = ds.variables['PHB'][t_start:t_end]
    z_stag = (PH + PHB) * g_inv  # PH and PHB are staggered

    if above_ground:
        HGT = ds.variables['HGT']
        hgt = HGT[0]
        z_stag -= hgt

    return z_stag



if __name__ == '__main__':
    main()
