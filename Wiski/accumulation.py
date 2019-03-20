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

import numpy as np
from pytz import UTC

from make_masks import ConfigGetter
from wiski import read_weights

np.set_printoptions(precision=3, threshold=10000, linewidth=125)


def parse_date(s: str) -> datetime:
    last_ex = None
    for DF in ('%Y-%m-%d', '%Y-%m-%dT%H', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s, DF)
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

    parser.add_argument('from_date', type=parse_date, default=datetime(1970, 1, 1),
                        help='Start of accumulation period')
    parser.add_argument('to_date', type=parse_date, default=datetime.now(UTC),
                        help='End of accumulation period')
    return ConfigGetter(parser)


# MAIN FUNCTION

def read_data(cfg: ConfigGetter) -> np.ndarray:
    if cfg.from_date > cfg.to_date:
        cfg.error(f'Empty accumulation period {cfg.from_date:%Y-%m-%dT%H:%M:%S} … {cfg.to_date:%Y-%m-%dT%H:%M:%S}')

    files = set()

    paths = cfg.wrfouts
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        files_for_path: List[str] = glob(path, recursive=True)
        files.update(files_for_path)
    files = sorted(files)

    if len(files) == 0:
        print('No wrfout files found')
        sys.exit(1)
    print(f'Found {len(files)} wrfout files')

    wrfout_tpl = cfg.wrfout_tpl
    wrfout_tpl = os.path.expanduser(wrfout_tpl)
    wrfout_tpl = os.path.expandvars(wrfout_tpl)

    # We need the last files that start on or before from_date and to_date
    from_idx, from_file = last_before(cfg.error, wrfout_tpl.format(start_date=cfg.from_date), files)
    to_idx, to_file = last_before(cfg.error, wrfout_tpl.format(start_date=cfg.to_date), files)
    print("from_file:", from_file, from_idx)
    print("to_file: ", to_file, to_idx)

    if cfg.verbose:
        print('\nfiles:')
        for i, file in enumerate(files):
            print(f'    {file}', '← from' if i == from_idx else '← to' if i == to_idx else '')

    file_steps = cfg.get('file_steps', None)
    if file_steps is None:
        ...

    else:
        print('TODO: Implement reading a range of steps from each file')

    sys.exit(42)


def last_before(error: Callable[[str], None], target: str, files: List[str]) -> Tuple[int, str]:
    try:
        idx, file = next((i, f) for i, f in enumerate(reversed(files)) if f <= target)
        idx = len(files) - idx - 1
        return idx, file
    except StopIteration:
        error(f'No wrfout file found before\n\t\t{target}\n\tFirst file is:\n\t\t{files and files[0]}')
        sys.exit(1)


def main():
    cfg = read_config()
    sub_levels: int = cfg.sub_sampling ** 2

    regions_and_weights = read_weights(cfg.weight_file_pattern, cfg.simulation, sub_levels, only_all_heights=True)

    data = read_data(cfg)

    for raw in regions_and_weights:
        # print(weight_grid.shape, np.min(weight_grid), np.average(weight_grid), np.max(weight_grid))

        # Crop data to the size of weight_grid at the requested offset
        wgo = raw.offset
        wgs = raw.weight_grid.shape
        cropped_data = data[wgo[0]:wgo[0] + wgs[0], wgo[1]:wgo[1] + wgs[1]]
        print("cropped_data:", cropped_data)

        # Weigh data and accumulate over grid, leaving time axis
        weighed = cropped_data * raw.weight_grid / sub_levels  # [i,j,t]
        print("weighed:", weighed)
        sum_over_area = np.sum(weighed)  # t
        print("sum_over_area:", sum_over_area)


if __name__ == '__main__':
    main()
