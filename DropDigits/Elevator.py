#!/usr/bin/env python
# encoding: utf-8

"""
Interpolate 3D variables to certain elevations above ground or sea-level.
Also de-staggers all variables horizontally.
"""

import argparse
import logging.config
import os
import re
from collections import OrderedDict
from time import time
from types import FunctionType
from typing import List, Dict, Any, Union

import numpy as np
import yaml
from netCDF4 import Dataset, Variable

from calculators import ChunkCalculator, CALCULATORS, HeightType
from utils import out_file_name, setup_logging, read_wrf_dates, CHUNK_SIZE_TIME, pick_chunk_sizes, \
    create_output_dataset, DIM_BOTTOM_TOP, DIM_TIME

LOG = logging.getLogger('belgingur.elevator')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

np.set_printoptions(4, edgeitems=3, linewidth=200)

""" Names of static dimensions which can be read from another file from the same model config """
DIM_NAMES_GEO = ('XLAT', 'XLAT_M', 'XLONG', 'XLONG_M', 'HGT', 'HGT_M', 'COSALPHA', 'SINALPHA')

""" Variables where we drop the Time dimension. """
VAR_NAMES_STATIC = (
    'SINALPHA', 'COSALPHA',
    'XLAT', 'XLAT_M', 'XLONG', 'XLONG_M',
    'HGT', 'HGT_M', 'HGT_SHAD',
    'MAPFAC', 'MAPFAC_M', 'MF_VX_INV',
    'E', 'F',
)


HEIGHT_TYPE_DESCRIPTION = {
    HeightType.above_ground: 'above ground',
    HeightType.above_sea: 'above sea-level',
    HeightType.pressure: 'pressure levels'
}


def configure() -> (argparse.Namespace, dict):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='Elevator.yml',
                        help='Configuration to read (def: Elevator.yml)')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Write more progress data')
    parser.add_argument('--geo-fallback',
                        help='Read XLAT, XLONG, HGT from this file if missing in an input file')
    parser.add_argument('--geo-margin', default=0, type=int,
                        help='Margin to discard on geo-fallback file to match input files')
    parser.add_argument('in_files', nargs="+",
                        help='wrf output files to process')
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug('Enabled DEBUG logging')

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


def create_output_variables(
        in_ds: Dataset, geo_ds: Union[None, Dataset], calculators:
        Dict[str, Any], out_ds: Dataset, out_var_names: List[str],
        comp_level: int, chunking: bool, elevation_limit: int,
        height_unit: str
) -> List[Variable]:
    LOG.info('Create output variables with:')
    out_vars = []
    for var_name in out_var_names:
        # Pick either an input ariable or a calculator function to create a variable for
        # We have given the calculator functions just enough attributes that either will work below
        if var_name in in_ds.variables:
            source = in_ds.variables[var_name]  # type: Variable
        elif var_name in DIM_NAMES_GEO and geo_ds and var_name in geo_ds.variables:
            source = geo_ds.variables[var_name]  # type: Variable
        elif var_name in calculators:
            source = calculators[var_name]
        else:
            raise ValueError('Unknown variable %s', var_name)

        dimensions = (destagger_dim_name(d) for d in source.dimensions)
        if var_name in VAR_NAMES_STATIC:
            dimensions = filter(lambda s: s != DIM_TIME, dimensions)
        dimensions = list(dimensions)

        data_type_name = datatype_name(source.datatype)
        scale_factor = getattr(source, 'scale_factor', 1)
        add_offset = getattr(source, 'add_offset', 0)
        data_type_name, add_offset, scale_factor = avoid_signed_types(data_type_name, add_offset, scale_factor)
        LOG.info('    %- 15s(%s): %s * %s + %s', var_name, ','.join(dimensions), data_type_name, scale_factor, add_offset)

        chunk_sizes = pick_chunk_sizes(in_ds, dimensions, max_k=elevation_limit) if chunking else None
        out_var = out_ds.createVariable(
            var_name,
            data_type_name,
            dimensions=dimensions,
            zlib=comp_level > 0,
            complevel=comp_level,
            shuffle=True,
            chunksizes=chunk_sizes
        )
        for field in (
                'description', 'least_significant_digit',
                'FieldType', 'MemoryOrder', 'units', 'coordinates',
        ):
            if field == 'units' and var_name == 'height':
                value = height_unit
            else:
                value = getattr(source, field, None)
            if value is not None:
                setattr(out_var, field, value)
        if scale_factor not in (None, 1):
            out_var.scale_factor = scale_factor
        if add_offset not in (None, 0):
            out_var.add_offset = add_offset
        out_vars.append(out_var)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, out_vars)))
    return out_vars


def avoid_signed_types(name: str, offset: float, scale_factor: float):
    if name == 'uint8':
        bits = 8
    elif name == 'uint16':
        bits = 16
    elif name == 'uint32':
        bits = 32
    else:
        return name, offset, scale_factor

    LOG.info('    Avoid NC4C incompatible type %s * %s + %s', name, scale_factor, offset)
    if offset is None:
        offset = 0
    u_name = f'int{bits}'
    u_offset = offset + 2 ** (bits - 1) * scale_factor
    return u_name, u_offset, scale_factor


def datatype_name(datatype):
    """ String representation of np.dtype """
    dts = str(datatype)
    m = re.compile("<.*\.(\w+)'>").match(dts)
    if m:
        return m.group(1)
    return dts


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
    total_start_time = time()
    total_in_size = total_out_size = 0
    heights = config.get('heights')
    height_type = HeightType[config.get('height_type')]
    height_unit = 'hPa' if height_type == HeightType.pressure else 'm'
    LOG.info(
        'Interpolate variables to %s%s %s',
        (height_unit + ', ').join(map(str, heights)),
        height_unit,
        HEIGHT_TYPE_DESCRIPTION[height_type]
    )
    LOG.info('')
    geo_ds = Dataset(args.geo_fallback) if args.geo_fallback else None
    geo_margin = args.geo_margin

    for in_file in args.in_files:
        start_time = time()
        out_file_pattern = config.get('output_filename', './{filename}_reduced.nc4')
        out_file = out_file_name(in_file, out_file_pattern)

        process_file(geo_ds, geo_margin, in_file, out_file, config=config)

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
            time() - start_time
        ))
        LOG.info('')

    if len(args.in_files) > 1:
        total_out_percent = (100 * total_out_size / total_in_size)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_in_size / 1024,
            total_out_size / 1024,
            total_out_percent,
            time() - total_start_time,
        ))


def process_file(geo_ds: Dataset, geo_margin: int, in_file: str, out_file: str, *, config: Dict[str, Any]):
    LOG.info('Opening input dataset %s', in_file)
    in_ds = Dataset(in_file, 'r')
    out_var_names = config.get('variables')

    in_dim_names, out_dim_names = resolve_dimensions(in_ds, CALCULATORS, out_var_names)

    dates = read_wrf_dates(in_ds)
    heights = config.get('heights')  # type: List[int]
    height_type = HeightType[config.get('height_type')]
    height_unit = 'hPa' if height_type == HeightType.pressure else 'm'
    custom_attributes = config.get('custom_attributes', dict())
    custom_attributes['interpolation'] = f'Interpolated to {(height_unit + ", ").join(map(str, heights))}{height_unit} {HEIGHT_TYPE_DESCRIPTION[height_type]}'

    out_ds = create_output_dataset(out_file, in_file, in_ds, custom_attributes)
    create_output_dimensions(in_ds, out_ds, out_dim_names, len(heights))

    chunking = config.get('chunking', False)
    comp_level = config.get('complevel', 0)
    create_output_variables(in_ds, geo_ds, CALCULATORS, out_ds, out_var_names, comp_level, chunking, len(heights), height_unit)

    chunk_size = CHUNK_SIZE_TIME // 4
    LOG.info('Processing data in chunks of %s time steps', chunk_size)
    for t_start in range(0, len(dates), chunk_size):
        t_end = min(t_start + chunk_size, len(dates))
        LOG.info('Chunk[%s:%s]: %s - %s', t_start, t_end, dates[t_start], dates[t_end - 1])

        cc = ChunkCalculator(t_start, t_end, heights, height_type)
        cc.add_dataset(in_ds)
        cc.add_dataset(geo_ds, geo_margin, DIM_NAMES_GEO)
        cc.make_vars_static(*VAR_NAMES_STATIC)

        LOG.info('Processing Variable')
        for out_var_name in out_var_names:
            LOG.info('    %s', out_var_name)
            before_var = time()
            out_var = out_ds.variables[out_var_name]
            out_chunk = cc(out_var_name)

            try:
                if DIM_TIME in out_var.dimensions:
                    out_var[t_start:t_end] = out_chunk
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug('        range: %g .. %g', np.min(out_chunk), np.max(out_chunk))

                # If this is not a time series, copy the data as we do the first chunk.
                elif t_start == 0:
                    out_var[:] = out_chunk

            except IndexError:
                LOG.error(
                    'Unable to write out_chunk with shape %s to output variable of shape %s',
                    out_chunk.shape, out_var.shape)
                exit(1)

            out_ds.sync()
            LOG.info('        %.3fs', time() - before_var)

    # Close our datasets
    in_ds.close()
    out_ds.close()
    return out_file


def resolve_dimensions(
        in_ds: Dataset, calculators: Dict[str, FunctionType], out_var_names: List[str]
) -> (List[str], List[str]):
    in_dim_names = OrderedDict()  # type: OrderedDict[str, str]
    out_dim_names = OrderedDict()  # type: OrderedDict[str, str]
    vars = in_ds.variables  # type: Dict[str, Variable]
    _ = 'X'

    def add_var(var_name, add_out, indent=0):
        if var_name in vars:
            for in_dim_name in vars[var_name].dimensions:
                in_dim_names[in_dim_name] = _
                if add_out:
                    out_dim_name = destagger_dim_name(in_dim_name)
                    out_dim_names[out_dim_name] = _

        elif var_name in calculators:
            calc = calculators[var_name]
            if add_out:
                for out_dim_name in calc.dimensions:
                    out_dim_names[out_dim_name] = _
            for in_var_name in calc.inputs:
                add_var(in_var_name, False, indent + 1)

    for out_var_name in out_var_names:
        add_var(out_var_name, True)

    return list(in_dim_names.keys()), list(out_dim_names.keys())


if __name__ == '__main__':
    main()
