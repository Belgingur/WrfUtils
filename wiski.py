#!/usr/bin/env python
# encoding: utf-8

"""
Builds data files for wiski by applying masks from make_masks wrf model output
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Python library imports
import argparse
from gzip import GzipFile
import netCDF4
from datetime import datetime

import numpy as np
import sys
import yaml


# SETUP

np.set_printoptions(precision=3, threshold=10000, linewidth=125)


def read_config():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        epilog=None
    )
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Write more progress data')
    parser.add_argument('--config', default='wiski.yml',
                        help='Read configuration from this file (def: wiski.yml)')
    parser.add_argument('wrfout', nargs='+',
                        help='WRF model output to calculate from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    return config, args.wrfout, args.verbose


# PREPARE

def read_weights(weight_file, levels):
    print('Read weights from', weight_file)
    weight_map = np.load(weight_file)
    keys_and_weights = []
    for key, weight_grid in sorted(weight_map.items()):
        total_weight = np.sum(weight_grid) / levels
        print('  {:<25s} {:5.2f} cl'.format(key, total_weight))
        keys_and_weights.append((key, weight_grid, total_weight))
    return keys_and_weights


# VARIABLE FETCHERS

WRF_TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'


def parse_timestamp(t):
    return datetime.strptime(t.tostring(), WRF_TIME_FORMAT)


def read_timestamps(wrfout):
    return [parse_timestamp(t) for t in wrfout.variables['Times']]


def read_data_direct(wrfout, var_key):
    return wrfout.variables[var_key][:]


def read_data_T2(wrfout, var_key):
    return wrfout.variables[var_key][:] - 272.15


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
    config, wrfouts, verbose = read_config()
    levels = config['sub_sampling'] ** 2
    spinup_steps = int(config['spinup_steps'])

    region_keys_and_weights = read_weights(config['weight_file'], levels)
    output_line_pattern = config['output_line_pattern']
    output_file_pattern = config['output_file_pattern']
    for wrfout_name in wrfouts:
        print('Read', wrfout_name)
        with netCDF4.Dataset(wrfout_name) as wrfout:
            timestamps = read_timestamps(wrfout)
            start_time = timestamps[spinup_steps]
            output_name = output_file_pattern.format(start_time=start_time)
            print('Write', output_name)
            with GzipFile(output_name, 'w') as output_file:
                for var_key in config['variables']:
                    print('  ', var_key)
                    data, precision = read_data(wrfout, var_key)
                    # print(data.shape, np.min(data), np.average(data), np.max(data))
                    for region_key, weight_grid, total_weight in region_keys_and_weights:
                        # print(weight_grid.shape, np.min(weight_grid), np.average(weight_grid), np.max(weight_grid))
                        # weight_grid = weight_grid.reshape(data.shape[1:3])  # [i,j]
                        weighed = data * weight_grid / levels  # [i,j,t]
                        sum_over_time = np.sum(weighed, axis=(1, 2))  # t
                        avg_over_time = sum_over_time / total_weight
                        avg = np.average(avg_over_time[spinup_steps:])
                        if verbose:
                            print('    {:<25}{:8.{:}f}'.format(region_key, avg, precision))

                        for time, value in zip(timestamps[spinup_steps:], avg_over_time[spinup_steps:]):
                            line = output_line_pattern.format(
                                region_key=region_key,
                                variable=var_key,
                                time=time,
                                value=rround(value, precision)
                            )
                            output_file.write(line)
                            output_file.write('\n')


if __name__ == '__main__':
    main()
