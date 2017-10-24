#!/usr/bin/env python
# encoding: utf-8

"""

Generate a point forecast using wrfout file and station information file and save it to disk.
Use bilinear interpolation to create weights.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
from collections import OrderedDict
import netCDF4
import numpy as np

from wrfout_reader import WRFReader
from bilinear_interpolation import bilinear_on_demand, NoClosePointError
from utilities import load_config, configure_logging, parse_iso_date
from data_utils import save_timeseries, templated_filename, load_point_metadata


LOG = logging.getLogger('belgingur.point_forecast')


class TargetOutsideGridError(ValueError):
    """ Raised when the target point is outside the borders of the forecast grid. """


class NotEnoughDataError(ValueError):
    """ Raised when there is no data about weights, and --no-adhoc-interpolation prevented from interpolating on-demand. """


def weighted_avg(data, weights):

    """ Calculate weighted average from the values and weights list. """

    return sum([d * w for d, w in zip(data, weights)])


def circular_weighted_avg(m, weights):

    """ Calculate weighted average of circular quantities such as wind direction. """

    s = [np.sin(np.deg2rad(x)) for x in m]
    weighted_s = weighted_avg(s, weights)
    c = [np.cos(np.deg2rad(x)) for x in m]
    weighted_c = weighted_avg(c, weights)

    value = np.rad2deg(np.arctan2(weighted_s, weighted_c))
    if value < 0:
        value += 360
    return value


def make_pf(data, weights, constant=0, circular=False):

    """ Apply weights to data taken from a WRF out file and return a timeseries of point forecasts """

    keys = weights.keys()
    weights_order = [weights[k] for k in keys]

    pf_series = []
    for time_slice in data:
        ts_values = [time_slice[j, i] for (i, j) in keys]
        avg = circular_weighted_avg(ts_values, weights_order) if circular else weighted_avg(ts_values, weights_order)
        avg += constant
        pf_series.append(avg)

    return pf_series


def make_forecast_timeseries(station, var, args, data_gridded, data_gridded_lt):

    """ Get the weights and produce point forecast timeseries. """

    LOG.info('Processing station %s, variable %s', station['ref'], var)

    try:
        weights = bilinear_on_demand(station, args.wrfout, args.margin)
    except NoClosePointError as e:
        LOG.debug(e)
        LOG.warning('Location of station %s outside wrfout grid, ignoring further processing for that station.', station['ref'])
        raise TargetOutsideGridError()

    pf = make_pf(data_gridded[var], weights, circular=(var == 'wind_dir'))

    if args.wrfout_long_term:
        try:
            weights = bilinear_on_demand(station, args.wrfout_long_term)
        except NoClosePointError as e:
            LOG.debug(e)
            LOG.warning('Station location outside wrfout-long-term grid, ignoring further processing for that station.')
            LOG.warning("The main wrfout file covered the grid, while the long term wrfout didn't. Are you sure the grids were properly defined?")

            raise TargetOutsideGridError()
        pf.extend(make_pf(data_gridded_lt[var], weights, circular=(var == 'wind_dir')))

    return pf


def load_forecast_data(wrfout, wrfout_long_term, components, spinup):

    """ Load forecast from a wrfout file with optional second dataset for a long term forecast. """

    LOG.info('Reading forecast from %s', wrfout)
    with netCDF4.Dataset(wrfout, mode='r') as nc_data:
        reader = WRFReader(nc_data)
        data_gridded = {var: reader.get_variable(var, spinup) for var in components}
        timestamps = reader.get_timestamps(spinup)

    data_gridded_lt = None
    if wrfout_long_term:
        LOG.info('A long term forecast from %s is being read and merged with the high-resolution short-term forecast.',
                 wrfout_long_term)
        LOG.info('The weights for long term forecast would be generated ad hoc by bilinear interpolation.')
        with netCDF4.Dataset(wrfout_long_term, mode='r') as nc_data:
            reader = WRFReader(nc_data)
            timestamps_lt = reader.get_timestamps(0)
            first_lt = min([ts for ts in timestamps_lt if ts > max(timestamps)])
            spinup = timestamps_lt.index(first_lt)

            data_gridded_lt = {var: reader.get_variable(var, spinup) for var in components}
            timestamps_lt = reader.get_timestamps(spinup)
            timestamps.extend(timestamps_lt)

    return data_gridded, data_gridded_lt, timestamps


def parse_args():
    parser = argparse.ArgumentParser(description=netCDF4.sys.modules[__name__].__doc__)
    parser.add_argument('--config', required=True, help='Config file for program. Required')
    parser.add_argument('--log-config', default=os.path.join(os.path.dirname(__file__), 'logging.yml'),
                        help='Config file for logging. Defaults to logging.yml in code folder')

    parser.add_argument('--analysis', type=parse_iso_date, required=True,
                        help='Analysis date for the forecast. Required')  # TODO not required? this is only an output
    parser.add_argument('--components', nargs='+', required=True,
                        help='One or more meteorological variables to produce point forecast (use temp, wind_speed, '
                             'wind_dir, prec_rate, snow_ratio, pressure, mslp, humidity, rel_hum, total_clouds '
                             'or any name of a two-dimensional non-staggered variable from wrfout)')
    parser.add_argument('--wrfout', required=True, help='Grid forecast file to use for point forecast generation.')
    parser.add_argument('--wrfout-long-term', help='Additional wrfout can be added with lower resolution and '
                                                   'times ranging farther than the basic wrfout.')
    parser.add_argument('--margin', type=int, default=0,
                        help='When using on-demand bilinear interpolation, describes the number of cells '
                             'from the border of the domain that we want to discard from processing')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_config)
    config = load_config(args.config)

    stations_pf = load_point_metadata(config)
    data_gridded, data_gridded_lt, timestamps = load_forecast_data(args.wrfout, args.wrfout_long_term, args.components, config.get('spinup', 0))

    for location in stations_pf:
        pf = {}
        try:
            for var in args.components:
                pf[var] = make_forecast_timeseries(location, var, args, data_gridded, data_gridded_lt)
        except (TargetOutsideGridError, NotEnoughDataError):
            # skip this location
            continue

        pf_file = templated_filename(config, analysis_date=args.analysis, ref=location['ref'], create_dirs=True)

        metadata = OrderedDict([
            ('ref', location['ref']),
            ('name', location['name']),
            ('analysis_date', args.analysis.strftime('%Y-%m-%dT%H:%M:%S')),
            ('longitude', '{:.4f}'.format(location['lon'])),
            ('latitude', '{:.4f}'.format(location['lat']))
        ])

        save_timeseries(timestamps, pf, pf_file, metadata)


if __name__ == '__main__':
    main()
