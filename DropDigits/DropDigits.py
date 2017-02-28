#!/usr/bin/env python
# encoding: utf-8

"""
Shrink wrfout files by reducing the number of digits for variables.
"""

from __future__ import print_function, unicode_literals, division, absolute_import

import argparse
import datetime
import logging
import logging.config
import os
import sys
import time
from math import log10, ceil

import netCDF4
import numpy as np
import yaml

LOG = logging.getLogger('belgingur.drop_digits')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

# Make basestring available in python3.5 so we can test if isinstance(var, basestring)
try:
    basestring
except NameError:
    basestring = str

#############################################################
# Parameters/constants for input data

CHUNK_SIZE_TIME = 128

CHUNK_SIZES = [None,
               (CHUNK_SIZE_TIME, 19),
               (CHUNK_SIZE_TIME, 16, 16),
               (CHUNK_SIZE_TIME, 10, 16, 16)]

TYPE_RANGE = dict(
    u1=(0, 2 ** 8 - 1),
    u2=(0, 2 ** 16 - 1),
    u4=(0, 2 ** 32 - 1),
    u8=(0, 2 ** 64 - 1),

    i1=(-2 ** 7, 2 ** 7 - 1),
    i2=(-2 ** 15, 2 ** 15 - 1),
    i4=(-2 ** 31, 2 ** 31 - 1),
    i8=(-2 ** 63, 2 ** 63 - 1),

    f4=(-3.4e38, +3.4e38),
    f8=(-1.79e308, +1.79e308)
)
TYPE_RANGE[None] = None


class Override(object):
    def __init__(self, datatype=None, scale_factor=None, add_offset=None, least_significant_digit=None):
        super(Override, self).__init__()
        self.least_significant_digit = least_significant_digit
        self.add_offset = add_offset
        self.scale_factor = scale_factor
        self.datatype = datatype

        range = TYPE_RANGE[datatype]
        if range:
            sf = 1 if scale_factor is None else scale_factor
            ao = 0 if add_offset is None else add_offset
            self.range_min = range[0] * sf - ao
            self.range_max = range[1] * sf - ao
        else:
            self.range_min = None
            self.range_max = None

        # Calculate least significant digit for int/fixed point variables
        if self.least_significant_digit is None:
            if self.datatype and self.datatype[0] in ('u', 'i') and self.scale_factor is not None:
                self.least_significant_digit = ceil(-log10(self.scale_factor))

    def __repr__(self):
        s = self.datatype or 'unchanged'
        if self.scale_factor != 1 and self.scale_factor is not None:
            s += '*{:g}'.format(self.scale_factor)
        if self.add_offset != 0 and self.add_offset is not None:
            s += '{:s}{:g}'.format('+' if self.add_offset >= 0 else '', self.add_offset)
        # if self.range_min is not None:
        #    s += ' : {:g} .. {:g}'.format(self.range_min, self.range_max)
        return s


def setup_logging(config_path=os.path.join(os.path.dirname(__file__), './logging.yml')):
    with open(config_path) as configFile:
        logging_config = yaml.load(configFile)
        logging.config.dictConfig(logging_config)
        LOG.info('Configured logging from %s', config_path)


def configure():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='DropDigits.yml',
                        help='Configuration to read (def: DropDigits.yml)')
    parser.add_argument('infiles', nargs="+",
                        help='input file from wrf to to process')
    args = parser.parse_args()

    LOG.info('Load config from %s', args.config)
    with open(args.config) as configFile:
        config = yaml.load(configFile)
        LOG.debug('Config: %s', config)

    return args, config


def build_overrides(config):
    """
    Iterate through config['overrides'] and make each entry into a valid Override object with defaults taken from
    override_defaults
    """
    specs = config['overrides']
    overrides = {}
    for var_name, spec in specs.items():
        try:
            if isinstance(spec, basestring) and spec[0] == '@':
                spec = specs[spec[1:]]
            overrides[var_name] = Override(**spec)
        except:
            LOG.error('Failed to read override for %s', var_name)
            raise
    return overrides


def resolve_input_variables(inds, config):
    """
    Retrieves the variables from inds which we intend to copy to outds.
    """
    default_include = config.get('default_include', True)
    includes = config.get('include', [])
    excludes = config.get('exclude', [])

    if default_include:
        LOG.info('Include all variables except %s', excludes)
    else:
        LOG.info('Include selected variables %s', ', '.join(includes))

    invars = []
    included_names = []
    excluded_names = []
    for var_name, invar in inds.variables.items():
        if (default_include and var_name not in excludes) or \
                (not default_include and var_name in includes):
            included_names.append(var_name)
            invars.append(invar)
        else:
            excluded_names.append(var_name)
    LOG.debug('Included variables: \n%s', '\n'.join(map(str, invars)))
    LOG.info('Excluded variables: %s', ', '.join(excluded_names) or '<none>')

    if default_include:
        LOG.info('Included variables: %s', ', '.join(included_names))
    else:
        unseen_vars = [var_name for var_name in includes if var_name not in included_names]
        if unseen_vars:
            LOG.warn('Missing variables in include list: %s', ', '.join(unseen_vars))

    return invars


def value_with_override(name, override, invar, default=None):
    value = getattr(override, name, None)
    if value is None:
        value = getattr(invar, name, None)
    if value is None:
        value = default
    return value


def override_field(outvar, name, override, invar, default=None):
    """
    Overrides a named field in outvar with the first value found in:

    - override dict
    - invar attribute
    - default
    """
    value = value_with_override(name, override, invar, __EMPTY__)
    if value is not __EMPTY__:
        setattr(outvar, name, value)


def create_output_variables(outds, invars, overrides, complevel):
    LOG.info('Create output variables with overrides:')
    outvars = []
    for invar in invars:
        var_name = invar.name
        override = overrides.get(var_name, None) or overrides.get('default')
        LOG.info('    %- 10s %- 10s %s..%s', var_name, override, override.range_min, override.range_max)

        datatype = value_with_override('datatype', override, invar)

        outvar = outds.createVariable(invar.name,
                                      datatype,
                                      dimensions=invar.dimensions,
                                      zlib=complevel > 0, complevel=complevel,
                                      shuffle=True)
        for field in ('description', 'least_significant_digit', 'scale_factor', 'add_offset',):
            override_field(outvar, field, override, invar)
        outvars.append(outvar)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, outvars)))
    return outvars


def add_attr(obj, name, value):
    """
    Find the first unset global attribute `name`, `name2`, etc. and set it to `value`
    """
    n = 0
    name_n = name
    value_n = getattr(obj, name_n, None)
    while value_n is not None:
        n += 1
        name_n = name + str(n)
        value_n = getattr(n, name_n, None)
    LOG.info('    %s = %s', name_n, value)
    setattr(obj, name_n, value)


def create_output_file(outfile_pattern, infile, inds):
    """
    Creates a new dataset with the same attributes as an existing one plus additional
    attributes to trace the file's evolution. Copies the Times variable over verbatim
    if it exists.

    Args:
        outfile (string):
        infile (string):
        inds (netCDF4.Dataset):
    """
    inpath, inbase = os.path.split(infile)
    inbase, inext = os.path.splitext(inbase)
    outfile = outfile_pattern.format(
        path=inpath or '.',
        basename=inbase,
        ext=inext,
    )

    LOG.info('Creating output file %s', outfile)
    if os.path.exists(outfile):
        logging.warning('Will overwrite existing file')
    outds = netCDF4.Dataset(outfile, mode='w', weakref=True)

    # Add some file meta-data
    LOG.info('Setting/updating global file attributes for output file')
    LOG.info('Copy %s attributes', len(inds.ncattrs()))
    for attr in inds.ncattrs():
        v = getattr(inds, attr)
        setattr(outds, attr, v)
        LOG.debug('    %s = %s', attr, v)
    LOG.info('Add attributes:')
    add_attr(outds, 'history', 'Created with python at %s by %s' % (
        datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%S'),
        os.getlogin()
    ))
    add_attr(outds, 'institution', 'Belgingur')
    add_attr(outds, 'source', infile)
    outds.description = 'Reduced version of: %s' % (getattr(inds, 'description', infile))
    LOG.info('    description = %s', outds.description)

    # Flush to disk
    outds.sync()
    return outfile, outds


def create_output_dimensions(inds, invars, outds, margin, sigma_limit):
    LOG.info('Add output dimensions:')
    needdims = set()
    for invar in invars:
        for dimname in invar.dimensions:
            needdims.add(dimname)

    included_names = []
    excluded_names = []
    for dimname, indim in inds.dimensions.items():
        if dimname in needdims:
            size = None if indim.isunlimited() else indim.size
            if size and looks_planar(dimname):
                size -= 2 * margin
            elif sigma_limit and dimname == 'bottom_top':
                size = min(sigma_limit, size)
            elif sigma_limit and dimname == 'bottom_top_stag':
                size = min(sigma_limit + 1, size)
            LOG.info('    %s (%s)', dimname, size or 'unlimited')
            outds.createDimension(dimname, size)
            included_names.append(dimname)
        else:
            excluded_names.append(dimname)

    # LOG.info('Included dimensions: %s', ', '.join(included_names))
    LOG.info('Excluded dimensions: %s', ', '.join(excluded_names) or '<none>')


def looks_planar(dimname):
    """
    Determines whether a dimension with the given name is a planar dimension, i.e. east/west or north/south

    :param unicode dimname: Name of dimension
    """
    return dimname and \
           'south' in dimname or 'north' in dimname or \
           'west' in dimname or 'east' in dimname


def work_wrf_dates(times):
    """
    Convert the WRF-style Times array from list of strings to a list of datetime objects

    Args:
        list[string] times:

    Returns:
        list[datetime]: Python-friendly dates
    """
    dates = []
    for t in times[:]:
        tt = t.tostring()
        if sys.version_info >= (3, 0):
            tt = tt.decode()
        dates.append(datetime.datetime.strptime(tt, '%Y-%m-%d_%H:%M:%S'))
    dates = np.array(dates)
    return dates


def log_sigma_level_height(inds, sigma_limit):
    """ Logs the minimum height of the highest sigma level above sea level and above surface. """
    if sigma_limit is not None:

        PH_ = inds.variables.get('PH')
        PHB = inds.variables.get('PHB')
        HGT = inds.variables.get('HGT')
        if PH_ is not None and PHB is not None and HGT is not None:
            np.set_printoptions(edgeitems=5, precision=0, linewidth=220)

            # De-stagger PH and PHB (average adjacent surfaces)
            # and convert to height (add PH and PHB and divide by g)
            # Bulding the _l arrays and adding the two levels is much faster than adding them directly from PH and PHB
            PH__l = PH_[:, sigma_limit - 1:sigma_limit + 1, :, :]
            PHB_l = PHB[:, sigma_limit - 1:sigma_limit + 1, :, :]
            Z_HGT = (PH__l[:, 0, :, :] + PH__l[:, 1, :, :] +
                     PHB_l[:, 0, :, :] + PHB_l[:, 1, :, :]) / (2 * 9.81)
            HGT0 = np.maximum(HGT, 0)  # Ignore ocean depth
            height_asl = np.min(Z_HGT)
            height_agl = np.min(Z_HGT - HGT0)
            LOG.info(
                '    3D variables limited to %d levels which reach at least %0.0fm above sea level and %0.0fm above surface level',
                sigma_limit, height_asl, height_agl
            )
        else:
            LOG.info('    3D variables limited to %d levels which reach an unknown height')


############################################################
# The main routine!

def main():
    setup_logging()

    args, config = configure()
    overrides = build_overrides(config)
    outfile_pattern = config.get('output_filename', './{filename}_reduced.nc4')
    total_start_time = time.time()
    total_errors = 0
    total_insize = total_outsize = 0
    LOG.info('')
    for infile in args.infiles:
        start_time = time.time()
        errors = 0

        # Open input datasets
        LOG.info('Opening input dataset %s', infile)
        inds = netCDF4.Dataset(infile, 'r')
        invars = resolve_input_variables(inds, config)

        # Convert time vector
        dates = work_wrf_dates(inds.variables['Times'])

        # Calculate spinup and margin
        LOG.info('Dimensional limits')
        dt = int((dates[-1] - dates[0]).total_seconds() / (len(dates) - 1) + 0.5)
        spinup_hours = config.get('spinup_hours', 0)
        spinup = int(spinup_hours * 3600. / dt + 0.5)
        LOG.info('    Spinup is %dh = %d steps', spinup_hours, spinup)
        margin = int(config.get('margin_cells', 0))
        LOG.info('    Margin is %d cells', margin)
        sigma_limit = config.get('sigma_limit', None)
        log_sigma_level_height(inds, sigma_limit)

        # Create empty output file
        outfile, outds = create_output_file(outfile_pattern, infile, inds)
        create_output_dimensions(inds, invars, outds, margin, sigma_limit)
        outvars = create_output_variables(outds, invars, overrides, config.get('complevel', 0))

        # Start the loop through time
        LOG.info('Copying data in chunks of %s time steps', CHUNK_SIZE_TIME)
        for c_start in range(spinup, len(dates), CHUNK_SIZE_TIME):
            c_end = min(c_start + CHUNK_SIZE_TIME, len(dates))
            LOG.info('Chunk[%s..%s]: %s - %s', c_start, c_end - 1, dates[c_start], dates[c_end - 1])
            LOG.info('    Variable            Min          Max  Dimensions')
            if c_start > spinup and c_end - c_start != CHUNK_SIZE_TIME:
                LOG.info('Last chunk is short')
            for invar, outvar in zip(invars, outvars):
                inchunk = invar[c_start:c_end]
                dim_str = ', '.join(map(lambda x: '%s[%s]' % x, zip(invar.dimensions, invar.shape)))
                override = overrides.get(invar.name) or overrides.get('default')

                if invar.datatype == '|S1':
                    # Text data
                    LOG.info('    {:10}          N/A          N/A  {}'.format(invar.name, dim_str))
                else:
                    # Numeric data
                    chunk_min, chunk_max = np.min(inchunk), np.max(inchunk)
                    LOG.info('    {:10} {:12,.2f} {:12,.2f}  {}'.format(invar.name, chunk_min, chunk_max, dim_str))
                    if override.range_min is not None and override.range_max is not None:
                        if chunk_min < override.range_min - override.scale_factor or chunk_max > override.range_max + override.scale_factor:
                            LOG.error(
                                '%s[%s..%s] values are %g .. %g outside valid range %g .. %g for %s',
                                invar.name, c_start, c_end,
                                chunk_min, chunk_max,
                                override.range_min, override.range_max,
                                override)
                            errors += 1

                # Decide whether to limit the 3rd dimension. We need to have a 3rd dimension and a limit
                max_k = None
                if sigma_limit is not None:
                    if 'bottom_top' in invar.dimensions:
                        max_k = sigma_limit
                    elif 'bottom_top_stag' in invar.dimensions:
                        max_k = sigma_limit + 1

                # Copy chunk, but shift by spinup steps and cut off margin and sigma levels as appropriate
                if max_k is not None:
                    outvar[c_start - spinup:c_end - spinup] = \
                        inchunk[..., 0:max_k, margin:-margin, margin:-margin]
                else:
                    outvar[c_start - spinup:c_end - spinup] = \
                        inchunk[..., margin:-margin, margin:-margin]
            outds.sync()

        # Close our datasets
        inds.close()
        outds.close()

        # Print space saved and time used
        insize = os.path.getsize(infile)
        outsize = os.path.getsize(outfile)
        total_errors += errors
        total_insize += insize
        total_outsize += outsize
        outpercent = (100.0 * outsize / insize)
        LOG.info('Size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:0.1f} s'.format(
            insize / 1024.0,
            outsize / 1024.0,
            outpercent,
            time.time() - start_time
        ))
        if errors:
            LOG.error('%d errors in file', errors)
            sys.exit(1)
        LOG.info('')

    if len(args.infiles) > 1:
        total_outpercent = (100.0 * total_outsize / total_insize)
        LOG.info('Total size: {:,.0f} MB -> {:,.0f} MB, reduced to {:,.2g}% in {:.1f} s'.format(
            total_insize / 1024.0,
            total_outsize / 1024.0,
            total_outpercent,
            time.time() - total_start_time,
        ))
        if total_errors:
            LOG.error('%d errors in total', total_errors)


if __name__ == '__main__':
    main()
