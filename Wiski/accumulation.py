#!/usr/bin/env python3
# encoding: utf-8

"""
Applies masks from make_masks to wrf model output and returns the difference in accumulated precipitation between the two
"""

from __future__ import annotations

import argparse
import os
import sys
from builtins import reversed
from datetime import datetime, timedelta
from glob import glob
from typing import List, Tuple, Callable

import netCDF4 as nc
import numpy as np
import pylab
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import SEC_PER_HOUR
from mpl_toolkits.basemap import Basemap
from pytz import UTC

from make_masks import ConfigGetter
from wiski import read_weights, read_timestamps, RegionAndWeights

np.set_printoptions(precision=3, threshold=10000, linewidth=125)


def parse_time(s: str) -> datetime:
    last_ex = None
    for DF in ('%Y-%m-%d', '%Y-%m-%dT%H', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(s, DF)
            dt = UTC.localize(dt)
            return dt
        except Exception as e:
            last_ex = e
    raise last_ex


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

    parser.add_argument('-f', '--from_time', type=parse_time,
                        help='Start of accumulation period')
    parser.add_argument('-t-', '--to_time', type=parse_time,
                        help='End of accumulation period')
    return ConfigGetter(parser)


def pick_period(cfg, region, key: str):
    dt: datetime = getattr(cfg._args, key)
    if dt is not None:
        return dt

    period_conf = cfg.get('periods', {}).get(region, {})
    dt = period_conf.get(key)
    dt = parse_time(dt)
    return dt


def read_geo_shape(cfg: ConfigGetter):
    path = cfg.geo
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    print("geo path:", path)
    with nc.Dataset(path) as ds:
        lats = ds.variables['XLAT_M'][0]
        lons = ds.variables['XLONG_M'][0]
        hgts = ds.variables['HGT_M'][0]
        print('shape', lats.shape)

        return lats, lons, hgts


def read_accumulation_time_to_time(path: str, to_time: datetime, from_time: datetime = None, verbose=False):
    """ Read accumulation between timestamps in the given data file. If to_time is not given, return accumulation from start of simulation. """
    with nc.Dataset(path) as ds:
        times = read_timestamps(ds)
        to_step = times.index(to_time)
        print(f'read step {to_step} ({to_time:%Y-%m-%dT%H:%M}) of {path}')
        to_accumulation = ds.variables['RAINC'][to_step] + ds.variables['RAINNC'][to_step]
        # print("to_accumulation:", np.round(np.sum(to_accumulation)))  # total mm*cells

        if from_time is None:
            return to_accumulation

        from_step = times.index(from_time)
        print(f'read step {from_step} ({from_time:%Y-%m-%dT%H:%M}) of {path}')
        from_accumulation = ds.variables['RAINC'][from_step] + ds.variables['RAINNC'][from_step]
        # print("from_accumulation:", np.round(np.sum(from_accumulation)))  # total mm*cells
        accumulation = to_accumulation - from_accumulation
        return accumulation


def find_nearest(a, v):
    return min(range(len(a)), key=lambda x: abs(a[x] - v))


def read_accumulation_idx_to_time(path: str, min_from_time: datetime, from_step: int, to_time: datetime, verbose: bool) -> Tuple[np.ndarray, datetime]:
    """
    Read accumulation in file between from_time and to_time where from_time is the time at from_step
    unless this is before min_from_time, then we start there instead.
    """
    with nc.Dataset(path) as ds:
        times = read_timestamps(ds)
        from_time = times[from_step]
        if from_time < min_from_time:
            from_step = find_nearest(times, min_from_time)
            from_time = times[from_step]

        to_step = find_nearest(times, to_time)

        hours = (to_time - from_time).total_seconds() / SEC_PER_HOUR
        hours = int(hours + 0.5)

        to_accumulation = ds.variables['RAINC'][to_step] + ds.variables['RAINNC'][to_step]
        from_accumulation = ds.variables['RAINC'][from_step] + ds.variables['RAINNC'][from_step]
        accumulation = to_accumulation - from_accumulation

        print(f'read {np.average(accumulation): 4.1f} mm at {from_time:%Y-%m-%dT%H:%M} +{hours:2d}h from {path}')
        return accumulation, from_time


def read_accumulation(cfg: ConfigGetter, to_time: datetime, from_time: datetime, verbose: bool) -> np.ndarray:
    """ Read accumulation between timestamps in the configured accumulation """
    if from_time >= to_time:
        cfg.error(f'Empty accumulation period {from_time :%Y-%m-%dT%H:%M} … {to_time :%Y-%m-%dT%H:%M}')

    files = set()

    paths = cfg.wrfouts
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        files_for_path: List[str] = glob(path, recursive=True)
        files.update(files_for_path)
    files = sorted(files)

    if len(files) == 0:
        print('No wrfout files found')
        sys.exit(1)
    if verbose:
        print(f'\nFound {len(files)} wrfout files')
        print(f'  to time: {to_time:%Y-%m-%dT%H:%M}')
        print(f'from time: {from_time:%Y-%m-%dT%H:%M}')

    from_step, step_to = cfg.get('file_steps', (None, None))
    if from_step:
        from_time_file = from_time - timedelta(minutes=from_step * cfg.step_length)
        print(f'from time: {from_time_file:%Y-%m-%dT%H:%M} with spinup')
    else:
        from_time_file = from_time

    wrfout_tpl = cfg.wrfout_tpl
    wrfout_tpl = os.path.expandvars(wrfout_tpl)
    wrfout_tpl = os.path.expanduser(wrfout_tpl)

    # We need the last files that start on or before from_time and to_time
    from_idx, from_file = last_before(cfg.error, wrfout_tpl.format(start_time=from_time_file), files)
    to_idx, to_file = last_before(cfg.error, wrfout_tpl.format(start_time=to_time), files)

    if cfg.verbose:
        print('files:')
        for i, file in enumerate(files):
            print(f'    {file}', '← from' if i == from_idx else '← to' if i == to_idx else '')
        print()

    if from_step is None:
        # All files are from one simulation
        if from_file == to_file:
            accumulation = read_accumulation_time_to_time(from_file, from_time, to_time, verbose)
            print("accumulation:", np.sum(accumulation))
            return accumulation
        else:
            to_accumulation = read_accumulation_time_to_time(to_file, to_time, None, verbose)
            from_accumulation = read_accumulation_time_to_time(from_file, from_time, None, verbose)
            accumulation = to_accumulation - from_accumulation
            if cfg.verbose:
                print("from_accumulation:", np.round(np.sum(from_accumulation), 2))
                print("  to_accumulation:", np.round(np.sum(to_accumulation), 2))
                print("     accumulation:", np.round(np.sum(accumulation), 2))
            return accumulation

    else:
        # Each file is from a different simulation so we need to accumulate over each and sum
        ptr_time = to_time
        accumulation = 0
        for file in reversed(files[from_idx:to_idx + 1]):
            accumulation_from_file, ptr_time = read_accumulation_idx_to_time(file, from_time, from_step, ptr_time, verbose)
            accumulation += accumulation_from_file
        return accumulation


def last_before(error: Callable[[str], None], target: str, files: List[str]) -> Tuple[int, str]:
    try:
        idx, file = next((i, f) for i, f in enumerate(reversed(files)) if f <= target)
        idx = len(files) - idx - 1
        return idx, file
    except StopIteration:
        error(f'No wrfout file found before\n\t\t{target}\n\tFirst file is:\n\t\t{files and files[0]}')
        sys.exit(1)


def setup_basemap(lat00: float, lon00: float, lat11: float, lon11: float):
    latc = (lat11 + lat00) / 2
    lonc = (lon11 + lon00) / 2
    return Basemap(projection='stere', lat_ts=latc, lat_0=latc, lon_0=lonc,
                   llcrnrlat=lat00, llcrnrlon=lon00,
                   urcrnrlat=lat11, urcrnrlon=lon11,
                   resolution='i')


A, B, C, D = (0.00, 0.25, 0.75, 1.00)
CMAP = LinearSegmentedColormap('white-yellow-red', dict(
    red=[
        (A, 1.0, 1.0),
        (B, 1.0, 1.0),
        (C, 1.0, 1.0),
        (D, 1.0, 0.0),
    ],
    green=[
        (A, 1.0, 1.0),
        (B, 1.0, 1.0),
        (C, 0.0, 0.0),
        (D, 0.0, 0.0),
    ],
    blue=[
        (A, 1.0, 1.0),
        (B, 0.0, 0.0),
        (C, 0.0, 0.0),
        (D, 0.5, 0.5),
    ],

), N=1024)


def pre_plot(m: Basemap, lat00: float, lon00: float, lat11: float, lon11: float, xs: np.ndarray, ys: np.ndarray, hgts: np.ndarray):
    pylab.figure(figsize=(12, 10), dpi=100, frameon=True)
    m.drawcoastlines(linewidth=1.5, color='black')
    m.readshapefile('joklar/joklar', 'joklar', linewidth=1.5, color='black')

    # m.drawparallels(np.arange(floor(lat00), ceil(lat11), 0.1), labels=[1, 0, 0, 0], color=(0, 0, 0, 0.4))
    # m.drawmeridians(np.arange(floor(lon00), ceil(lon11), 0.2), labels=[0, 0, 0, 1], color=(0, 0, 0, 0.4))

    iso = np.arange(0, 2300, 100)
    cs = m.contour(xs, ys, hgts, iso, colors='#808080', linewidths=0.5)
    pylab.clabel(cs, fmt='%1.0f', colors='#808080', inline=1, fontsize=10)


def plot(cfg, raw: RegionAndWeights, lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray, weighed: np.ndarray):
    # Crop domain shape vars to slightly larger than the weight matrix
    pad = 1
    j0 = raw.offset[0] - pad
    i0 = raw.offset[1] - pad
    j1 = j0 + raw.weight_grid.shape[0] + 2 * pad
    i1 = i0 + raw.weight_grid.shape[1] + 2 * pad
    crop_lats = lats[j0:j1, i0:i1]
    crop_lons = lons[j0:j1, i0:i1]
    crop_hgts = hgts[j0:j1, i0:i1]

    # Pad weighted data to same size as above
    padded = np.zeros(shape=crop_lats.shape)
    padded[pad:-pad, pad:-pad] = weighed

    lat0, lat1 = crop_lats[0, 0], crop_lats[-1, -1]
    lon0, lon1 = crop_lons[0, 0], crop_lons[-1, -1]
    m = setup_basemap(lat0, lon0, lat1, lon1)
    xs, ys = m(crop_lons, crop_lats)

    pre_plot(m, lat0, lon0, lat1, lon1, xs, ys, crop_hgts)

    m.contourf(xs, ys, padded, cmap=CMAP, levels=100)
    m.plot(xs, ys, '.', ms=2, color='k')

    # Add title and save
    plot_title = cfg.accumulation_plot_title_pattern.format(simulation=cfg.simulation, region=raw.region)
    plot_file = cfg.accumulation_plot_file_pattern.format(simulation=cfg.simulation, region=raw.region)
    print('\nsave', plot_file)
    pylab.title(plot_title)
    pylab.savefig(plot_file, pad_inches=0.2, bbox_inches='tight')
    pylab.clf()
    pylab.close()


def main():
    cfg = read_config()
    sub_levels: int = cfg.sub_sampling ** 2

    print('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print('SETUP:', cfg.simulation)
    regions_and_weights = read_weights(cfg.weight_file_pattern, cfg.simulation, sub_levels, only_all_heights=True)
    lats, lons, hgts = read_geo_shape(cfg)
    for raw in regions_and_weights:
        # print(weight_grid.shape, np.min(weight_grid), np.average(weight_grid), np.max(weight_grid))
        print('\n────────────────────────────────────────────────────────────────────────────────')
        print('REGION:', raw.key, '\n')
        from_time = pick_period(cfg, raw.region, 'from_time')
        to_time = pick_period(cfg, raw.region, 'to_time')
        days = (to_time - from_time).total_seconds() / (60 * 60 * 24)
        print(f'period: {from_time :%Y-%m-%dT%H:%M} … {to_time :%Y-%m-%dT%H:%M} is {days:g} days')
        accumulation = read_accumulation(cfg, to_time, from_time, cfg.verbose)
        print(f'\naccumulation on forecast domain: {np.average(accumulation):0.1f} mm')

        # For plotting crop accumulation to the size of weight_grid and multiply
        j0 = raw.offset[0]
        i0 = raw.offset[1]
        j1 = j0 + raw.weight_grid.shape[0]
        i1 = i0 + raw.weight_grid.shape[1]
        crop_accn = accumulation[j0:j1, i0:i1]
        weighed = crop_accn * raw.weight_grid / sub_levels  # [j,i]
        avg_over_area = np.sum(weighed) / raw.total_weight
        print(f'accumulation on {raw.region} {avg_over_area:0.1f} mm')

        plot(cfg, raw, lats, lons, hgts, weighed)


if __name__ == '__main__':
    main()
