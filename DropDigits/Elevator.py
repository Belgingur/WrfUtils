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
from collections import namedtuple
from functools import reduce
from typing import List, Dict, Any

import numpy as np
import yaml
from netCDF4 import Dataset, Variable

from utils import out_file_name, setup_logging, read_wrf_dates, CHUNK_SIZE_TIME, pick_chunk_sizes, \
    create_output_dataset, g_inv

DIM_BOTTOM_TOP = 'bottom_top'
DIM_BOTTOM_TOP_STAG = 'bottom_top_stag'

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
        out_dim_name = destagger_name(in_dim_name)
        if out_dim_name not in out_dim_names:
            out_dim_names.append(out_dim_name)
    return out_dim_names


def destagger_name(in_dim_name):
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
        out_dims = [destagger_name(d) for d in in_dims]
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
    var_names = config.get('variables')
    in_vars = resolve_input_variables(in_ds, var_names)
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
    out_vars = create_output_variables(in_ds, out_ds, var_names, comp_level, chunking, len(heights))

    LOG.info('Processing data in chunks of %s time steps', CHUNK_SIZE_TIME)
    for c_start in range(0, len(dates), CHUNK_SIZE_TIME):
        c_end = min(c_start + CHUNK_SIZE_TIME, len(dates))
        LOG.info('Chunk[%s:%s]: %s - %s', c_start, c_end, dates[c_start], dates[c_end - 1])

        inerpolator, interpolator_stag = build_interpolators(
            heights, in_dim_names, in_ds, c_start, c_end, above_ground
        )

        LOG.info('Processing Variable     Input Dimensions')
        if c_start > 0 and c_end - c_start != CHUNK_SIZE_TIME:
            LOG.info('Last chunk is short')
        for in_var, out_var in zip(in_vars, out_vars):
            in_chunk = in_var[c_start:c_end]
            dim_str = ', '.join(map(lambda x: '%s[%s]' % x, zip(in_var.dimensions, in_var.shape)))

            if in_var.datatype == '|S1':
                # Text data
                LOG.info('    {:10}          {}'.format(in_var.name, dim_str))
            else:
                # Numeric data
                if DIM_BOTTOM_TOP in in_var.dimensions:
                    in_chunk = inerpolator(in_chunk)
                elif DIM_BOTTOM_TOP_STAG in in_var.dimensions:
                    in_chunk = interpolator_stag(in_chunk)
                LOG.info('    {:10} {}'.format(in_var.name, dim_str))

            out_var[c_start:c_end] = in_chunk
            out_ds.sync()

    # Close our datasets
    in_ds.close()
    out_ds.close()
    return out_file


# Vertical Interpolation Constants
VIC = namedtuple('VIC', 't_grid j_grid i_grid k_ce k_fl w_ce w_fl mask')


class Interpolator(object):
    # TODO: replace with a closure
    def __init__(self, heights: List[float], vics: List[VIC]):
        super().__init__()
        if len(heights) != len(vics):
            raise ValueError('heights and vics lists must be of same length')
        self.heights = heights
        self.vics = vics

    def __call__(self, var: np.ndarray) -> np.ndarray:
        return apply_vics(self.vics, var)


def build_interpolators(heights: List[float], in_dims: List[str],
                        ds: Dataset, t_start: int, t_end: int, above_ground: bool) -> (Interpolator, Interpolator):
    """ Builds Interpolators for bottom_top and bottom_top_stag as needed according to in_dims. """
    LOG.info('Generate Vertical Interpolation Constants')

    PH = ds.variables['PH'][t_start:t_end]
    PHB = ds.variables['PHB'][t_start:t_end]
    z_stag = (PH + PHB) * g_inv  # PH and PHB are staggered

    if above_ground:
        HGT = ds.variables['HGT'][0, :, :]
        z_stag -= HGT

    LOG.info('heights: %s', heights)
    interpolator_stag = None
    if DIM_BOTTOM_TOP_STAG in in_dims:
        LOG.info('    for %s', DIM_BOTTOM_TOP_STAG)
        vics_stag = list(build_vic(h, z_stag) for h in heights)
        interpolator_stag = Interpolator(heights, vics_stag)
        # z_stag_heights = interpolator_stag(z_stag)  # Should be similar to heights

    interpolator = None
    if DIM_BOTTOM_TOP in in_dims:
        LOG.info('    for %s', DIM_BOTTOM_TOP)
        z = 0.5 * (z_stag[:, 0:-1, :, :] + z_stag[:, 1:, :, :])  # de-stagger along k-axis
        vics = list(build_vic(h, z) for h in heights)
        interpolator = Interpolator(heights, vics)
        #z_heights = interpolator(z)  # Should be similar to heights

    return interpolator, interpolator_stag


def build_vic(target: float, z: np.ndarray) -> 4 * (np.ndarray,):
    """
    Builds interpolation indices and coefficients to vertically interpolate any variable for chunk of time-steps in a
    wrfout file.

    :param: target (m) Desired vertical height
    :param: z[t,k,j,i] (m) geo-potential height of sigma fields
    :param: terrain[j,i] (m) optional height of terrain. If given, interpolate above surface instead of sea level.

    :return: 2D arrays for floor and ceiling indexes, 2D scalar fields for floor and ceiling weights
    """

    # Expected shape of variables with a flattened k-dimension
    flatshape = z.shape[0:1] + (1,) + z.shape[2:]
    flatsize = reduce(lambda x, y: x * y, flatshape)

    # Build the complete indexing grids for dimensions t,j,i but flatten dimension k
    t_size, k_size, j_size, i_size = z.shape
    t_grid, k_grid, j_grid, i_grid = np.meshgrid(range(t_size), range(1), range(j_size), range(i_size), indexing='ij')
    assert t_grid.shape == flatshape

    # The ceiling index at [t,_,j,i] is the number of values in z[t,:,j,i] below or at target.
    # We use <= so that if target is 0 then k_ce is 1 and not 0 so k_fl is not negative.
    # We cap k_ce at t_size to avoid indexing errors. This will be masked out in w_ce later
    k_ce = np.sum(z <= target, axis=1, keepdims=True)
    k_ce = np.minimum(k_ce, t_size)
    k_fl = k_ce - 1
    assert k_ce.shape == flatshape
    assert k_fl.shape == flatshape

    # Retrieve the height of the sigma surface at the ceiling and floor indexes
    z_ce = z[t_grid, k_ce, j_grid, i_grid]
    z_fl = z[t_grid, k_fl, j_grid, i_grid]
    assert z_ce.shape == flatshape
    assert z_fl.shape == flatshape
    # Interpolate ceiling weights and calculate floor weights
    w_ce = (target - z_fl) / (z_ce - z_fl)
    w_fl = 1 - w_ce
    assert w_ce.shape == flatshape
    assert w_fl.shape == flatshape

    # z should now interpolate to exactly target.
    # z_target = z_ce * w_ce + z_fl * w_fl
    # assert z_target.shape == flatshape
    # below = np.min(z_target - target)
    # above = np.max(z_target - target)
    # LOG.info('below, above: %s, %s', below, above)
    # assert below > -0.01 or isinstance(below, np.ma.core.MaskedConstant)
    # assert above < +0.01 or isinstance(above, np.ma.core.MaskedConstant)

    # We mask out extrapolated point
    mask = np.ma.mask_or(w_ce < 0, w_ce > 1, shrink=False)
    if isinstance(mask, np.ndarray):
        mask = mask[:, 0, :, :]
    trues = np.count_nonzero(mask)
    if trues:
        LOG.warning('        %sm %0.0f%% masked', target, 100 * trues / flatsize)
    else:
        LOG.info('        %sm', target)

    return VIC(t_grid, j_grid, i_grid, k_ce, k_fl, w_ce, w_fl, mask)


def apply_vic(vic: VIC, var: np.ndarray) -> np.ndarray:
    # Get ceiling and floor values for all [t,j,i]
    var_ce = var[vic.t_grid, vic.k_ce, vic.j_grid, vic.i_grid]
    var_fl = var[vic.t_grid, vic.k_fl, vic.j_grid, vic.i_grid]

    # Calculate weighed average and drop the empty k-dimension
    var_lvl = var_ce * vic.w_ce + var_fl * vic.w_fl
    var_lvl = np.squeeze(var_lvl, axis=1)

    # Fill with NaN according to the mask
    var_lvl[vic.mask] = np.nan
    return var_lvl


def apply_vics(vics, var) -> np.ndarray:
    var_lvl_list = []
    for vic in vics:
        var_lvl = apply_vic(vic, var)
        var_lvl_list.append(var_lvl)
    var_lvls = np.stack(var_lvl_list, axis=1)
    return var_lvls


if __name__ == '__main__':
    main()
