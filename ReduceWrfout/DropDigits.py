#!/usr/bin/env python
# encoding: utf-8

"""
Shrink wrfout files by reducing the number of digits for variables.
"""

import os
import sys

import netCDF4
import numpy
import logging
import logging.config
import time
import subprocess
import datetime
import argparse

import yaml

LOG = logging.getLogger('belgingur.drop_digits')

__EMPTY__ = '__EMPTY__'
""" Magic empty value distinct from None. """

#############################################################
# Parameters/constants for input data

chunkSize_days = 10


def setup_logging(config_path='./logging.yml'):
    with open(config_path) as configFile:
        logging_config = yaml.load(configFile)
        logging.config.dictConfig(logging_config)
        LOG.info('Configured logging from %s', config_path)


def configure():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='DropDigits.yml',
                        help='Configuration to read (def: DropDigits.yml)')
    parser.add_argument('infile',
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
    overrides = config['overrides']
    for var_name, value in overrides.items():
        if isinstance(value, basestring) and value[0] == '@':
            value = overrides[value[1:]]
        overrides[var_name] = value


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
        LOG.info('Exclude all variables except %s', includes)

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
    LOG.info('Included variables: %s', ', '.join(included_names))
    LOG.debug('Included variables: \n%s', '\n'.join(map(str, invars)))
    LOG.info('Excluded variables: %s', ', '.join(excluded_names))

    if not default_include:
        unseen_vars = [var_name for var_name in includes if var_name not in included_names]
        if unseen_vars:
            LOG.warn('Missing variables in include list: %s', ', '.join(unseen_vars))

    return invars


def value_with_override(name, override, invar, default=None):
    value = override.get(name)
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


def create_output_variables(outds, invars, config):
    overrides = config.get('overrides', {})
    deflatelevel = config.get('deflatelevel', 0)
    outvars = []
    for invar in invars:
        var_name = invar.name
        override = overrides.get(var_name, {})
        LOG.info('%s override: %s', var_name, override)

        datatype = value_with_override('datatype', override, invar)

        outvar = outds.createVariable(invar.name,
                                      datatype,
                                      dimensions=invar.dimensions,
                                      zlib=deflatelevel > 0, complevel=deflatelevel,
                                      shuffle=True)
        # TODO: fillValue = False for speed?
        # TODO: individual compression levels per variable?
        for field in ('description', 'least_significant_digit', 'scale_factor', 'add_offset',):
            override_field(outvar, field, override, invar)
        outvars.append(outvar)

    LOG.debug('Converted variables: \n%s', '\n'.join(map(str, outvars)))
    return outvars


def add_attr(outobj, inobj, name, value=None):
    """
    Copy all attributes name, name1, name2, etc from inobj to outobj and
    add name or nameN to the end with the given new value. if value_new
    is None then no new value is appended.
    """
    n = 0
    name_n = name
    value_n = getattr(inobj, name_n, None)
    while value_n is not None:
        LOG.info('Copy attribute %s = %s', name_n, value_n)
        setattr(outobj, name_n, value_n)
        n += 1
        name_n = name + str(n)
        value_n = getattr(n, name_n, None)
    if value is not None:
        LOG.info('Add attribute %s = %s', name_n, value)
        setattr(outobj, name_n, value)


def create_output_file(outfile, infile, inds):
    """
    Creeates an empty netcdf file with the same dimensions as an existing one.

    Args:
        string netcdf_in: The netcdf whose dimensions to copy
        string netcdf_out: The name of a netcdf file to create
        dates:

    Returns:

    """

    LOG.info('Creating output file %s', outfile)
    outds = netCDF4.Dataset(outfile, mode='w', weakref=True)

    # Add some file meta-data
    LOG.info('Setting/updating global file attributes for output file')
    outds.description = 'Reduced version of: %s' % (getattr(inds, 'description', infile))

    strnow = datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%S')

    add_attr(outds, inds, 'TITLE')
    add_attr(outds, inds, 'history', 'Created with python at %s by %s' % (strnow, os.getlogin()))
    add_attr(outds, inds, 'institution', 'Belgingur')
    add_attr(outds, inds, 'source', infile)

    # Flush to disk
    outds.sync()
    return outds


def create_output_dimensions(inds, invars, outds):
    needdims = set()
    for invar in invars:
        LOG.info('%s: %s', invar.name, invar.dimensions)
        for dimname in invar.dimensions:
            needdims.add(dimname)

    LOG.info('needdims: %s', needdims)
    included_names = []
    excluded_names = []
    for dimname, indim in inds.dimensions.items():
        if dimname in needdims:
            size = None if indim.isunlimited() else indim.size
            LOG.info('Add dimension %s (%s)', dimname, size or 'unlimited')
            outds.createDimension(dimname, size)
            included_names.append(dimname)
        else:
            excluded_names.append(dimname)

    LOG.info('Included dimensions: %s', ', '.join(included_names))
    LOG.info('Excluded dimensions: %s', ', '.join(excluded_names))


def work_wrf_dates(times):
    """
    Convert the WRF-style Times array from list of strings to a list of datetime objects

    Args:
        list[string] times:

    Returns:
        list[datetime]: Python-friendly dates
    """
    LOG.info('Working dates %s to %s', times[0].tostring(), times[-1].tostring())
    dates = []
    for t in times[:]:
        dates.append(datetime.datetime.strptime(t.tostring(), '%Y-%m-%d_%H:%M:%S'))
    dates = numpy.array(dates)
    return dates


def calc_wind_dir(ut, vt, sina, cosa):
    # Calculate true wind direction
    u_true = cosa * ut + sina * vt
    v_true = -sina * ut + cosa * vt
    wdir = numpy.mod(270. - (numpy.arctan2(v_true, u_true) * 180 / 3.14159), 360.)
    return wdir


############################################################
# The main routine!

def main():
    start_time = time.time()
    setup_logging()

    args, config = configure()
    build_overrides(config)
    infile = args.infile
    outfile = infile + '_reduced.nc4'

    # Open input datasets
    LOG.info('Opening input dataset %s', infile)
    inds = netCDF4.Dataset(infile, 'r')
    invars = resolve_input_variables(inds, config)

    # Convert time vector
    dates = work_wrf_dates(inds.variables['Times'])
    numdates = netCDF4.date2num(dates, 'hours since 0001-01-01 00:00:00.0', calendar='gregorian')

    # Preprocess output files
    LOG.info('Initialize output file %s', outfile)
    if os.path.exists(outfile):
        logging.warning('Will overwrite existing %s', outfile)
    outds = create_output_file(outfile, infile, inds)
    create_output_dimensions(inds, invars, outds)

    LOG.info('Creating output variables')
    outvars = create_output_variables(outds, invars, config)

    # Set chunks
    dt = (dates[1] - dates[0]).seconds
    size_t = len(dates)
    indices = range(0, size_t)
    chunk_size_prefered = int(chunkSize_days * 24. * 3600. / dt)
    chunk_size = min(size_t, chunk_size_prefered)
    LOG.info('Using chunk size: %s', chunk_size)

    # Start the loop through time
    LOG.info('Starting to loop through data')
    for t in xrange(0, size_t, chunk_size):
        chunk = indices[t:t + chunk_size]
        LOG.info('Chunk: %s - %s', dates[chunk[0]], dates[chunk[-1]])
        if len(chunk) != chunk_size:
            LOG.info('Last chunk is shorter than previous chunks')
        for invar, outvar in zip(invars, outvars):
            LOG.info('  Variable %s: %s',
                     invar.name,
                     ', '.join(map(lambda x: '%s[%s]' % x, zip(invar.dimensions, invar.shape)))
                     )
            outvar[:, :, chunk] = invar[:, :, chunk]
        outds.sync()

    # Close our datasets
    inds.close()
    outds.close()

    # Print space saved and time used
    insize = os.path.getsize(infile)
    outsize = os.path.getsize(outfile)
    outpercent = (100.0 * outsize / insize)
    LOG.info('Size: %0.0f MB -> %0.0f MB, reduced to %.2f%%', insize / 1024.0, outsize / 1024.0, outpercent)
    LOG.info('Timing: %.1f (seconds) ', time.time() - start_time)


if __name__ == '__main__':
    main()
