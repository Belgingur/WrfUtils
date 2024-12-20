#!/usr/bin/env python3
# encoding: utf-8

"""
Builds data files for wiski by applying masks from make_masks to WRF model output
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from gzip import GzipFile
from itertools import groupby
from typing import Tuple, Dict, List, Optional, Iterable

import netCDF4 as nc
import numpy as np
from pytz import UTC

from make_masks import ConfigGetter

np.set_printoptions(precision=3, threshold=10000, linewidth=125)


def regex_matching_arg(pattern, description=None):
    """
    Returns a custom type for use with ArgumentParser which passes the value through unchanged but validates it against
    a regular expression. On error it either shows the failing regex or an alternative description supplied in the
    argument.

    :param str or unicode pattern:
    :param str or unicode description:
    """
    if description is None:
        description = '"%s" does not match "%s"' % ('%s', pattern)
    pattern = '^' + pattern + '$'

    def type(v):
        try:
            return re.match(pattern, v, ).group(0)
        except:
            raise argparse.ArgumentTypeError(description % v)

    return type


def read_config() -> ConfigGetter:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=None
    )
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Write more progress data')
    parser.add_argument('-c', '--config', default='wiski.yml',
                        help='Read configuration from this file (def: wiski.yml)')
    parser.add_argument('-s', '--simulation',
                        help='Configured simulation to work with.')
    parser.add_argument('-p', '--perturb-forecast', default='N0E0',
                        type=regex_matching_arg(
                            '[NS][0-9]+[WE][0-9]+(,[NS][0-9]+[WE][0-9]+)*',
                            '%s is not a comma-separated list of [NS]number[EW]number values'
                        ),
                        help='Move weight grid by this many cells within the input wrfout files. '
                             'Can be a comma-separated list such as N2E2,N0W1,S2W0. '
                             'If multiple values are given, the process is run once for each perturbation '
                             'and the corresponding string value made available as {perturbation} in the '
                             'output file pattern')
    parser.add_argument('wrfouts', nargs='*',
                        help='WRF model output to calculate from. Overrides pattern in config.')
    args = parser.parse_args()

    return ConfigGetter(parser)


def is_digit(c: str):
    return c in '0123456789'


def parse_perturb(perturb_pretty: str) -> List[Tuple[int, int]]:
    perturb_pretties = perturb_pretty.split(',')
    perturb_idxs = []
    for pp in perturb_pretties:
        pp = tuple(''.join(x) for _, x in groupby(pp, key=is_digit))
        perturb_idxs.append((
            int(pp[1]) if pp[0] == 'N' else -int(pp[1]),
            int(pp[3]) if pp[2] == 'E' else -int(pp[3])
        ))
    return perturb_idxs


# PREPARE

@dataclass
class RegionAndWeights:
    region: str

    key: str
    """ Output key with heights (if any) """

    min_height: Optional[float]
    """ the bottom of the height range we're looking at (inclusive) """

    max_height: Optional[float]
    """ the top of the height range we're looking at (exclusive) """

    offset: Tuple[int, int]
    """ Offset of weight grid in domain """

    weight_grid: np.ndarray
    """ Weight grid, on a scale from 0..levels per cell"""

    total_weight: float
    """ Total weight of the grid """


def read_weights(weight_file_pattern: str, simulation: str, levels, *, only_all_heights=False):
    weight_file = weight_file_pattern.format(simulation=simulation)
    print('Read weights from', weight_file)
    weight_map: Dict[str, np.ndarray] = np.load(weight_file)
    keys_and_weights: List[RegionAndWeights] = []

    for key, weight_grid in sorted(weight_map.items()):
        total_weight = np.sum(weight_grid) / levels

        if ':' in key:
            key, offset = key.split(':')
            offset = tuple(map(int, offset.split('_')))
            assert len(offset) == 2
        else:
            offset = (0, 0)

        if '_' in key:
            region, min_height, max_height = key.split('_')
            if only_all_heights:
                continue
            min_height = float(min_height)
            max_height = float(max_height)
        else:
            min_height = max_height = None
            region = key

        print('  {:<25s} {:5.2f} cl'.format(key, total_weight))
        keys_and_weights.append(RegionAndWeights(region, key, min_height, max_height, offset, weight_grid, total_weight))

    return keys_and_weights


# VARIABLE FETCHERS

WRF_TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'


def parse_timestamp(b: Iterable[bytes]) -> datetime:
    """ Parse a WRF time value """
    return parse_wrf_time_string(b''.join(bb for bb in b).decode())


def parse_wrf_time_string(s: str) -> datetime:
    """ Parse a WRF time string """
    d = datetime.strptime(s, '%Y-%m-%d_%H:%M:%S')
    d = UTC.localize(d)
    return d


def read_timestamps(wrfout):
    return [parse_timestamp(t) for t in wrfout.variables['Times'][:]]


def read_data_direct(wrfout, var_key):
    return wrfout.variables[var_key][:]


def read_data_T2(wrfout, var_key):
    return wrfout.variables[var_key][:] - 272.15


def read_data_WSPEED(wrfout, var_key):
    V10 = wrfout.variables['V10'][:]
    U10 = wrfout.variables['U10'][:]
    WS = np.sqrt(U10 ** 2 + V10 ** 2)
    return WS


def read_data_PRECIP(wrfout, var_key):
    """
    Returns precipitation [t,i,j] as the change in RAINC+RAINCC between time
    steps. Since this is not defined for t==0 we return i*j NaNs in the first
    step.
    """
    dsv = wrfout.variables
    p = dsv['RAINC'][:] + dsv['RAINNC'][:]
    dp = np.empty_like(p)
    dp[0, :, :] = np.nan
    dp[1:, :, :] = p[1:, :, :] - p[:-1, :, :]
    # print(' p[t,10,10]', p[1:11, 10, 10])
    # print('*p[t,10,10]', p[0:10, 10, 10])
    # print('dp[t,10,10]', dp[:10, 10, 10])
    return dp


def read_data_RAINFALL(wrfout, var_key):
    rv = read_data_PRECIP(wrfout, 'PRECIP')
    sr = wrfout.variables['SR'][:]
    rv[sr >= 0.4] = 0
    return rv


def read_data_SLEETFALL(wrfout, var_key):
    rv = read_data_PRECIP(wrfout, 'PRECIP')
    sr = wrfout.variables['SR'][:]
    rv[(sr < 0.4) | (sr >= 0.7)] = 0
    return rv


def read_data_SNOWFALL(wrfout, var_key):
    rv = read_data_PRECIP(wrfout, 'PRECIP')
    sr = wrfout.variables['SR'][:]
    rv[sr < 0.7] = 0
    return rv


DEFAULT_PRECISION = 2
PRECISON = dict(
    PSFC=0
)
""" Output-precision of variables if other than the default """


def read_data(wrfout, var_key):
    """
    Finds the specific read_data function for var_key, such as read_data_T2
    for var_key==T2 and returns the data and precision returned. If only the
    data is returned, supplies a default precision of 2

    :type wrfout: ndarray
    :type var_key: str
    :rtype: (ndarray, int)
    """
    getter = globals().get('read_data_' + var_key, read_data_direct)
    data = getter(wrfout, var_key)  # [t,i,j], []
    precision = PRECISON.get(var_key, DEFAULT_PRECISION)
    return data, precision


# MAIN FUNCTION


def rround(x, p):
    return round(x, p) if p > 0 else int(x)


def main():
    cfg = read_config()
    perturb_idxs = parse_perturb(cfg.perturb_forecast)
    levels: int = cfg.sub_sampling ** 2
    spinup_steps = int(cfg.spinup_steps)

    regions_and_weights = read_weights(cfg.weight_file_pattern, cfg.simulation, levels)
    output_line_pattern: str = cfg.output_line_pattern
    output_file_pattern: str = cfg.output_file_pattern

    for wrfout_name in cfg.wrfouts:
        print('Read', wrfout_name)

        for perturb_idx, perturb_pretty in zip(perturb_idxs, cfg.perturb_forecast):
            with nc.Dataset(wrfout_name) as wrfout:
                timestamps = read_timestamps(wrfout)
                start_time = timestamps[spinup_steps]
                output_name = output_file_pattern.format(
                    start_time=start_time,
                    perturb_pretty=perturb_pretty,
                    perturb_idx=perturb_idx
                )
                print('Write', output_name)
                if len(perturb_idxs) > 1 and perturb_idxs != (0, 0):
                    print('Perturb', perturb_pretty, perturb_idx)

                with GzipFile(output_name, 'w') as output_file:
                    for var_key in cfg.variables:
                        print('  ', var_key)
                        data, precision = read_data(wrfout, var_key)
                        # print(data.shape, np.min(data), np.average(data), np.max(data))

                        for raw in regions_and_weights:
                            # print(weight_grid.shape, np.min(weight_grid), np.average(weight_grid), np.max(weight_grid))

                            # Crop data to the size of weight_grid at the requested offset
                            wgo = (raw.offset[0] + perturb_idx[0], raw.offset[1] + perturb_idx[1])
                            wgs = raw.weight_grid.shape
                            ds = data.shape[1:]
                            if wgo[0] + wgs[0] > ds[0] or wgo[1] + wgs[1] > ds[1] or wgo[0] < 0 or wgo[1] < 0:
                                raise Exception(
                                    'Can\'t perturb weight grid for %s of shape %s at offset %s by %s %s and fit in data of shape %s' %
                                    (raw.key, wgs, raw.offset, perturb_pretty, perturb_idx, ds)
                                )
                            cropped_data = data[:, wgo[0]:wgo[0] + wgs[0], wgo[1]:wgo[1] + wgs[1]]

                            # Weigh data and accumulate over grid, leaving time axis
                            weighed = cropped_data * raw.weight_grid / levels  # [i,j,t]
                            sum_over_area = np.sum(weighed, axis=(1, 2))  # t
                            avg_over_area = sum_over_area / raw.total_weight

                            if cfg.verbose:
                                avg = np.average(avg_over_area[spinup_steps:])
                                print('    {:<25}{:8.{:}f}'.format(raw.key, avg, precision))

                            # output the averaged data time series along with identifiers
                            for time, value in zip(timestamps[spinup_steps:], avg_over_area[spinup_steps:]):
                                line = output_line_pattern.format(
                                    region_key=raw.key,
                                    variable=var_key,
                                    time=time,
                                    value=rround(value, precision)
                                )
                                output_file.write(line.encode())
                                output_file.write(b'\n')


if __name__ == '__main__':
    main()
