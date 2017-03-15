"""
Code for vertical interpolation of 3D time-series to constant elevation levels
"""

import logging.config
from collections import namedtuple
from typing import List

import numpy as np

from utils import destagger_array, DIM_BOTTOM_TOP_STAG, DIM_BOTTOM_TOP

LOG = logging.getLogger('belgingur.elevator')

# Vertical Interpolation Constants
VIC = namedtuple('VIC', 't_grid j_grid i_grid k_ce k_fl w_ce w_fl mask')


class Interpolator(object):
    def __init__(self, heights: List[float], vics: List[VIC], dimension: str):
        super().__init__()
        if len(heights) != len(vics):
            raise ValueError('heights and vics lists must be of same length')
        self.heights = tuple(heights)
        self.vics = vics
        self.dimension = dimension

        self.max_k = max(np.max(v.k_ce) for v in vics)  # type: int
        """ The highest k-index used for any interpolation level """

    def __repr__(self, *args, **kwargs):
        return 'Interpolator[{}: {}]'.format(self.dimension, self.heights)

    def __call__(self, var: np.ndarray) -> np.ndarray:
        return apply_vics(self.vics, var)


def build_interpolators(
        z_stag: np.ndarray, targets: List[float], need_aligned: bool, need_staggered: bool
) -> (Interpolator, Interpolator):
    """ Builds Interpolators for bottom_top and bottom_top_stag as needed according to in_dims. """
    LOG.info('Generate Vertical Interpolation Constants')

    LOG.info('targets: %s', targets)
    interpolator_stag = None
    if need_staggered:
        LOG.info('    for vertically staggered')
        vics_stag = list(build_vic(tgt, z_stag) for tgt in targets)
        interpolator_stag = Interpolator(targets, vics_stag, DIM_BOTTOM_TOP_STAG)
        # z_stag_heights = interpolator_stag(z_stag)  # Should be similar to heights

    interpolator = None
    if need_aligned:
        LOG.info('    for vertically aligned')
        z = destagger_array(z_stag, 1)
        vics = list(build_vic(tgt, z) for tgt in targets)
        interpolator = Interpolator(targets, vics, DIM_BOTTOM_TOP)
        # z_heights = interpolator(z)  # Should be similar to heights

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
    flatsize = z.size / z.shape[1]

    # Build the complete indexing grids for dimensions t,j,i but flatten dimension k
    t_size, k_size, j_size, i_size = z.shape
    t_grid, k_grid, j_grid, i_grid = np.meshgrid(range(t_size), range(1), range(j_size), range(i_size), indexing='ij')
    assert t_grid.shape == flatshape

    # The ceiling index at [t,_,j,i] is the number of values in z[t,:,j,i] below or at target.
    # We use <= so that if target is 0 then k_ce is 1 and not 0 so k_fl is not negative.
    # We cap k_ce at t_size to avoid indexing errors. This will be masked out in w_ce later
    k_ce = np.sum(z <= target, axis=1, keepdims=True)
    k_ce = np.minimum(k_ce, k_size - 1)
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
