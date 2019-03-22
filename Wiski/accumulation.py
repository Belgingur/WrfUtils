#!/usr/bin/env python3
# encoding: utf-8

"""
Applies masks from make_masks to wrf model output and returns the difference in accumulated precipitation between the two
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from glob import glob
from typing import List, Tuple, Callable

import netCDF4 as nc
import numpy as np
import pylab
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
from pytz import UTC

from make_masks import ConfigGetter
from wiski import read_weights, read_timestamps

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
    dt = getattr(cfg._args, key)
    if dt is not None:
        return dt

    period_conf = cfg.get('periods', {}).get(region, {})
    dt = period_conf.get(key)
    dt = parse_time(dt)
    dt = UTC.localize(dt)
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

        return lats, lons, hgts


def read_accumulation_in_file(path: str, to_time: datetime, from_time: datetime = None):
    """ Read accumulation between timestamps in the given data file. If to_time is not given, return accumulation from start of simulation. """
    with nc.Dataset(path) as ds:
        times = read_timestamps(ds)
        to_idx = times.index(to_time)
        print(f'read step {to_idx} ({to_time:%Y-%m-%dT%H:%M}) of {path}')
        to_accumulation = ds.variables['RAINC'][to_idx] + ds.variables['RAINNC'][to_idx]
        # print("to_accumulation:", np.round(np.sum(to_accumulation)))  # total mm*cells

        if from_time is None:
            return to_accumulation

        from_idx = times.index(from_time)
        print(f'read step {from_idx} ({from_time:%Y-%m-%dT%H:%M}) of {path}')
        from_accumulation = ds.variables['RAINC'][from_idx] + ds.variables['RAINNC'][from_idx]
        # print("from_accumulation:", np.round(np.sum(from_accumulation)))  # total mm*cells
        accumulation = to_accumulation - from_accumulation
        return accumulation


def read_accumulation(cfg: ConfigGetter, to_time: datetime, from_time: datetime) -> np.ndarray:
    """ Read accumulation between timestamps in the configured data """
    if from_time > to_time:
        cfg.error(f'Empty accumulation period {from_time :%Y-%m-%dT%H:%M:%S} … {to_time :%Y-%m-%dT%H:%M:%S}')

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
    print(f'\nFound {len(files)} wrfout files')
    print(f'from time: {from_time:%Y-%m-%dT%H:%M}')
    print(f'  to time: {to_time:%Y-%m-%dT%H:%M}')

    wrfout_tpl = cfg.wrfout_tpl
    wrfout_tpl = os.path.expandvars(wrfout_tpl)
    wrfout_tpl = os.path.expanduser(wrfout_tpl)

    # We need the last files that start on or before from_time and to_time
    from_idx, from_file = last_before(cfg.error, wrfout_tpl.format(start_time=from_time), files)
    to_idx, to_file = last_before(cfg.error, wrfout_tpl.format(start_time=to_time), files)

    if cfg.verbose:
        print('files:')
        for i, file in enumerate(files):
            print(f'    {file}', '← from' if i == from_idx else '← to' if i == to_idx else '')
        print()

    file_steps = cfg.get('file_steps', None)
    if file_steps is None:
        if from_file == to_file:
            data = read_accumulation_in_file(from_file, from_time, to_time)
            print("data:", np.sum(data))
            return data
        else:
            to_data = read_accumulation_in_file(to_file, to_time)
            from_data = read_accumulation_in_file(from_file, from_time)
            data = to_data - from_data
            if cfg.verbose:
                print("  to_data:", np.round(np.sum(to_data), 2))
                print("from_data:", np.round(np.sum(from_data), 2))
                print("     data:", np.round(np.sum(data), 2))
            return data

    else:
        print('TODO: Implement reading a range of steps from each file')
        sys.exit(1)


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


def plot(
        cfg: ConfigGetter,
        region: str,
        lats: np.ndarray, lons: np.ndarray, hgts: np.ndarray,
        accumulation: np.ndarray
):
    lat0, lat1 = lats[0, 0], lats[-1, -1]
    lon0, lon1 = lons[0, 0], lons[-1, -1]
    m = setup_basemap(lat0, lon0, lat1, lon1)
    xs, ys = m(lons, lats)

    pre_plot(m, lat0, lon0, lat1, lon1, xs, ys, hgts)

    m.contourf(xs, ys, accumulation, cmap=CMAP, levels=100)
    m.plot(xs, ys, '.', ms=2, color='k')

    plot_title = cfg.accumulation_plot_title_pattern.format(simulation=cfg.simulation, region=region)
    plot_file = cfg.accumulation_plot_file_pattern.format(simulation=cfg.simulation, region=region)
    print('Save', plot_file)
    pylab.title(plot_title)
    pylab.savefig(plot_file, pad_inches=0.2, bbox_inches='tight')
    pylab.clf()
    pylab.close()


def main():
    cfg = read_config()
    sub_levels: int = cfg.sub_sampling ** 2

    regions_and_weights = read_weights(cfg.weight_file_pattern, cfg.simulation, sub_levels, only_all_heights=True)
    lats, lons, hgts = read_geo_shape(cfg)
    for raw in regions_and_weights:
        # print(weight_grid.shape, np.min(weight_grid), np.average(weight_grid), np.max(weight_grid))
        print('\n___________________________________________________________________')
        print('REGION:', raw.key)
        from_time = pick_period(cfg, raw.region, 'from_time')
        to_time = pick_period(cfg, raw.region, 'to_time')
        accumulation = read_accumulation(cfg, to_time, from_time)
        print("accumulation:", np.round(np.sum(accumulation)), accumulation.shape)

        # For plotting crop accumulation to the size of weight_grid and multiply
        j0 = raw.offset[0]
        i0 = raw.offset[1]
        j1 = j0 + raw.weight_grid.shape[0]
        i1 = i0 + raw.weight_grid.shape[1]
        crop_accn = accumulation[j0:j1, i0:i1]
        weighed = crop_accn * raw.weight_grid / sub_levels  # [j,i]

        pad = 1
        j0p = j0 - pad
        i0p = i0 - pad
        i1p = i1 + pad
        j1p = j1 + pad
        crop_lats = lats[j0p:j1p:, i0p:i1p:]
        crop_lons = lons[j0p:j1p:, i0p:i1p:]
        crop_hgts = hgts[j0p:j1p:, i0p:i1p:]
        padded = np.zeros(shape=crop_lats.shape)
        padded[pad:-pad, pad:-pad] = weighed
        plot(
            cfg, raw.region,
            crop_lats, crop_lons, crop_hgts,
            padded
        )
        sum_over_area = np.sum(weighed)
        print("sum_over_area:", round(sum_over_area, 2))


if __name__ == '__main__':
    main()
