#!/usr/bin/env python
# encoding: utf-8

"""
Show the min/max values for all variables in a list of wrfout files and picks the most precise 16-bit fixed-point type for each.
"""
from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import logging
import logging.config
from math import log, floor
from collections import OrderedDict

import netCDF4
import numpy as np
import yaml

LOG = logging.getLogger('belgingur.show_ranges')


def setup_logging(config_path='./logging.yml'):
    with open(config_path) as configFile:
        logging_config = yaml.load(configFile)
        logging.config.dictConfig(logging_config)
        LOG.info('Configured logging from %s', config_path)


def configure():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infiles', nargs="+",
                        help='input file from wrf to to process')
    return parser.parse_args()


def pick_type(vmin, vmax):
    if vmin < -(vmax - vmin) / 1000:
        sign = 'i'
        maxmax = max(abs(vmax), abs(vmin))
        total_bits = 15
    else:
        vmin = max(vmin, 0)  # Round away tiny negatives
        sign = 'u'
        maxmax = vmax
        total_bits = 16
    bits = log(maxmax, 2) if maxmax > 0 else 0
    spare_bits = total_bits - bits
    digits = floor(spare_bits * log(2, 10))
    digits = int(digits)

    if digits > 0:
        scale = '0.' + ('0' * (digits - 1)) + '1'
    else:
        scale = '1' + '0' * (1 - digits)

    rmax = 2 ** total_bits / 10 ** digits
    rmin = -rmax if sign == 'i' else 0
    assert rmin <= vmin <= vmax <= rmax

    return sign + '2', scale


def main():
    setup_logging()
    args = configure()

    ranges = OrderedDict()

    for infile in args.infiles:
        LOG.info('')
        LOG.info('Opening input dataset %s', infile)
        LOG.info('Variable             Min    Max')
        inds = netCDF4.Dataset(infile, 'r')
        for var_name, invar in inds.variables.items():
            if var_name == 'Times':
                continue
            try:
                var_min = np.amin(invar)
                var_max = np.amax(invar)
                range_min, range_max = ranges.get(var_name, (var_min, var_max))
                range_min, range_max = (min(range_min, var_min), max(range_max, var_max))
                ranges[var_name] = (range_min, range_max)
                LOG.info('%- 10s% 14s .. %- 14s', var_name, var_min, var_max)

            except Exception as e:
                LOG.info('%s failed: %s', var_name, e)
                LOG.exception('')

    print('#VARIABLE      OVERRIDE                                      #          OBSERVED RANGE')
    for var_name, (range_min, range_max) in ranges.items():
        datatype, scale_factor = pick_type(range_min, range_max)
        print("%-12s : { 'datatype':'%s', 'scale_factor':%- 9s } # % 14s .. %- 14s" % (
            var_name, datatype, scale_factor, range_min, range_max
        ))


if __name__ == '__main__':
    main()
