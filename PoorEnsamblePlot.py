#!/usr/bin/env python
# encoding: utf-8

"""
Plot a sequence of point forecasts on the same plot, with older data fading into the distance
"""

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import logging
import logging.config
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import netCDF4
import numpy as np
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from glob import glob
from itertools import count
from math import log, ceil, floor
from sqlalchemy import create_engine, MetaData

LOG = logging.getLogger('belgingur.poor_ensamble_plot')

# Make basestring available in python3.5 so we can test if isinstance(var, basestring)
try:
    basestring
except NameError:
    basestring = str
    unicode = str


def setup_logging(config_path='./logging.yml'):
    """
    Configure logging from the given YAML file.

    :param unicode config_path: path to file to take logging config from
    """
    with open(config_path) as configFile:
        logging_config = yaml.load(configFile)
        logging.config.dictConfig(logging_config)
        LOG.info('Configured logging from %s', config_path)


def parse_args():
    """
    Parse command-line arguments.

    :return: the resulting args object
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='PoorEnsamblePlot.yml',
                        help='Configuration to read (def: PoorEnsamblePlot.yml)')
    return parser.parse_args()


def load_config(path):
    """
    Load program configuration from the given YAML file.

    :param unicode path: path to file to take config from
    :rtype: dict
    """
    LOG.info('Load config from %s', path)
    with open(path) as configFile:
        config = yaml.load(configFile)
    return config


def init_database(db_url):
    """
    Initialise a database engin and load table metadata.

    :param unicode db_url: Connection URL for database. Something like
    """
    engine = create_engine(db_url)
    # insp = inspect(engine)  # If we need an inspector
    meta = MetaData()
    meta.reflect(bind=engine)
    LOG.info('Database Engine: %s', engine)
    for name, table in meta.tables.items():
        LOG.debug('  %s: %s', name, table)
    return engine, meta


def list_forecast_files(engine, schedule, domain, wrf_path_template):
    """
    List the forecast of the given schedule which have data on-disk.

    :param Engine engine:
    :param unicode schedule:
    :rtype: list[tuple]
    """

    # Select completed and unscrubbed jobs in the given schedule.
    # We might want to try to plot partially completed jobs as well
    sql = '''
        SELECT job.ref, job.submitted, job.analysis
        FROM job
        JOIN schedule sch ON sch.id=job.schedule_id
        WHERE sch.ref = %(schedule_ref)s
          AND NOT job.scrubbed
          AND job.completed IS NOT NULL
        ORDER BY job.analysis DESC
        LIMIT 20
    '''
    results = engine.execute(sql, schedule_ref=schedule)

    analyses = []
    files = []
    for r in results:
        wrf_file_glob = wrf_path_template.format(
            ref=r[0],
            submitted=r[1],
            analysis=r[2],
            domain=domain
        )
        matches = glob(wrf_file_glob)
        if len(matches) == 0:
            LOG.warn('No file found matching %s', wrf_file_glob)
            continue
        if len(matches) > 1:
            LOG.warn('Multiple files found matching %s', wrf_file_glob)
        LOG.info('  %s: %s', r[2], matches[0])
        analyses.append(r[2])
        files.append(matches[0])

    return analyses, files


def closest_point(file, lat, lon):
    ds = netCDF4.Dataset(file)
    lats = ds.variables['XLAT']
    lons = ds.variables['XLONG']
    while len(lats.shape) > 2:
        lats = lats[0]
        lons = lons[0]

    LOG.info('Find index of% 7.3f, % 7.3f', lat, lon)
    j0, i0 = -1, -1
    j = lats.shape[0] // 2
    i = lats.shape[1] // 2
    while j != j0 or i != i0:
        j0, i0 = j, i
        LOG.info('% 4d, % 4d -> % 7.3f, % 7.3f', j, i, lats[j, i], lons[j, i])
        i = np.searchsorted(lons[j, :], lon)
        LOG.info('% 4d, % 4d -> % 7.3f, % 7.3f', j, i, lats[j, i], lons[j, i])
        j = np.searchsorted(lats[:, i], lat)

    return j, i


def read_dates(ds):
    times = ds.variables['Times']
    dates = []
    for t in times[:]:
        tt = t.tostring()
        if sys.version_info >= (3, 0):
            tt = tt.decode()
        dates.append(datetime.strptime(tt, '%Y-%m-%d_%H:%M:%S'))
    return dates


def cutoff_at(values, cutoff):
    for i, v in enumerate(values):
        if v > cutoff:
            return values[i:], i
    return values[-1:-1], 0


def read_series(filename, cutoff_date, j, i, *var_names):
    ds = netCDF4.Dataset(filename)
    dates = read_dates(ds)
    dates, cutoff_index = cutoff_at(dates, cutoff_date)
    LOG.info('%s .. %s: %s', cutoff_index, len(dates), filename)
    if len(dates) < 2:
        raise ValueError('No usable data in ' + filename)

    vars = []
    for var_name in var_names:
        var = ds.variables[var_name]
        var = var[cutoff_index:, j, i]
        vars.append(var)
    return [dates] + vars


def make_minor_locator(days, max_ticks):
    per_day = max_ticks / days
    interval = int(ceil(24 / per_day))
    while 24 % interval != 0:
        interval += 1
    minorLocator = matplotlib.dates.HourLocator(range(0, 24, interval))
    return minorLocator


def plot_data(plot_path, plot_ref, component, analyses, dates_ens, var_ens):
    LOG.info('Plot %s', plot_path)

    fig, ax = plt.subplots()

    curve_count = len(dates_ens)
    LOG.info('len(analyses): %s', len(analyses))
    LOG.info('len(dates_ens): %s', len(dates_ens))
    LOG.info('len(var_ens): %s', len(var_ens))
    period = dates_ens[0][-1] - dates_ens[-1][0]
    days = period.total_seconds() / 60 / 60 / 24

    for i, analysis, dates, var in zip(count(), analyses, dates_ens, var_ens):
        if i == 0:
            width = 3
            color = (0.7, 0.1, 0)
        else:
            width = 1
            c = log(i + 2) / log(curve_count + 2)
            color = (1, c, c)
        ax.plot(dates[0:len(var)], var, zorder=1000 - i, linewidth=width, color=color)

    # Fromat grid and axis
    #ax.xaxis.grid(b=True, which='minor', color='0.9', linestyle='-', linewidth=0.5)
    ax.xaxis.grid(b=True, which='major', color='0.8', linestyle='-')
    ax.yaxis.grid(b=True, which='major', color='0.8', linestyle='-')

    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator([12]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('\n%a %d.%m'))
    ax.xaxis.set_minor_locator(make_minor_locator(days, 30))
    ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%H'))

    # labels = ax.get_xticklabels()
    # plt.setp(labels, rotation=30)

    plt.title(
        '{plot_ref} {component}\nAnalysis {analysis:%Y-%m-%d %H}'.format(
            plot_ref=plot_ref,
            component=component,
            analysis=analyses[0],
        ))
    fig.set_size_inches(12, 6)
    plt.savefig(plot_path, dpi=120)
    plt.close()


############################################################
# The main routine!

def main():
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    engine = create_engine(config['database_url'])
    wrf_path_template = config['wrf_path_template']
    plot_path_template = unicode(config['plot_path_template'])

    for plot in config['plots']:
        LOG.info('{ref}: {schedule}.*.d{domain:02d} ({lat},{lon})'.format(**plot))
        analyses, files = list_forecast_files(engine, plot['schedule'], plot['domain'], wrf_path_template)

        first_analysis = analyses[0]
        cutoff_date = first_analysis + netCDF4.timedelta(hours=plot['spinup'] - 0.5)
        j, i = closest_point(files[0], plot['lat'], plot['lon'])

        dates_ens = []
        data = defaultdict(list)
        for analysis, filename in zip(analyses, files):
            try:
                dates, t2, u10, v10, rainc, rainnc = read_series(
                    filename, cutoff_date, j, i,
                    'T2', 'U10', 'V10', 'RAINC', 'RAINNC'
                )

                # Get average length of time step in seconds
                interval = (dates[-1] - dates[0]).total_seconds() / (len(dates) - 1)

                # Convert accumulated precipitation to rate per hour
                precip = rainc + rainnc
                precip = precip[1:] - precip[:-1]
                precip = precip / interval * 3600

                dates_ens.append(dates)
                data['temperature'].append(t2 - 273.15)
                data['wind_speed'].append((u10 ** 2 + v10 ** 2) ** 0.5)
                data['precipitation'].append(precip)
            except Exception as e:
                LOG.warning(str(e))
                break

        for var_name, var_ens in data.items():
            plot_path = plot_path_template.format(
                component=var_name,
                analysis=first_analysis,
                **plot
            )
            plot_data(
                plot_path,
                plot['ref'],
                var_name,
                analyses,
                dates_ens,
                var_ens
            )


if __name__ == '__main__':
    main()
