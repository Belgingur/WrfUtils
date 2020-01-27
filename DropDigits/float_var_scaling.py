#!/usr/bin/env python3
from __future__ import annotations

import sys

import netCDF4 as nc
import numpy as np

if len(sys.argv) == 1:
    print()
    print('    Run: ', sys.argv[0], 'files...')
    print()
    print('    to make sure that scale_factor and add_offset in the files are floating')
    print('    point values and avoid overflows in netCDF scaling the values back from')
    print('    integer representation')
    print()
    sys.exit(1)

for fn in sys.argv[1:]:
    print('Fix', fn)
    ds = None
    try:
        ds = nc.Dataset(fn, mode='r+', diskless=True, persist=False)

        for name, var in ds.variables.items():
            scale_factor = getattr(var, 'scale_factor', None)
            add_offset = getattr(var, 'add_offset', None)
            int_scale_factor = np.issubdtype(type(scale_factor), np.integer)
            int_add_offset = np.issubdtype(type(add_offset), np.integer)
            if int_scale_factor or int_add_offset:
                print('    ', name, '*', scale_factor, '+', add_offset)
                if int_scale_factor:
                    var.scale_factor = float(scale_factor)
                if int_add_offset:
                    var.add_offset = float(add_offset)

    finally:
        if ds is not None:
            ds.close()
