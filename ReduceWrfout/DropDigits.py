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
    LOG.info('Included variables: %s', ' '.join(included_names))
    LOG.debug('Included variables: \n%s', '\n'.join(map(str, invars)))
    LOG.info('Excluded variables: %s', ' '.join(excluded_names))

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


def execute(exe, *args):
    """
    Executes external process and terminates with rc=-1 if it fails

    Args:
        string exe: main executable
        tuple of string args: arguments to command

    Returns:

    """
    cmd = [exe] + list(args)
    LOG.info(' '.join(cmd))
    try:
        __ = subprocess.check_call(cmd)
    except:
        LOG.error('Error running %s', exe)
        sys.exit(-1)


def setup_output_file(netcdf_in, netcdf_out, dates):
    """
    Creeates an empty netcdf file with the same dimensions as an existing one.

    Args:
        string netcdf_in: The netcdf whose dimensions to copy
        string netcdf_out: The name of a netcdf file to create
        dates:

    Returns:

    """

    netcdf_tmp = netcdf_out + '_tmp'

    # Run external utilities to create the initial, mostly empty target file
    try:
        LOG.info('Create temporary copy with HGT,XLAT,XLONG')
        execute('ncks', '-O', '-4',
                '-v', 'HGT,XLAT,XLONG',
                '-d', 'Time,0',
                netcdf_in, netcdf_tmp)

        LOG.info('Average copied variables over time in to 2D')
        execute('ncwa', '-O', '-4',
                '-a', 'Time',
                netcdf_tmp, netcdf_out)

        LOG.info('Copy original Time vector')
        execute('ncks', '-A', '-4',
                '-v', 'Times',
                netcdf_in, netcdf_out)

    finally:
        if os.path.isfile(netcdf_tmp):
            os.remove(netcdf_tmp)

    outds = netCDF4.Dataset(netcdf_out, mode='r+', weakref=True)

    # Add some file meta-data
    LOG.info('Setting/updating global file attributes for output file %s', netcdf_out)
    outds.description2 = 'Copy of %s with reversed order of dimensions' % (netcdf_in)
    outds.history2 = 'Created with python at %s by %s' % (
        datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%S'), os.getlogin())
    outds.institution = 'Belgingur'
    outds.source2 = '%s' % (netcdf_in)

    # Adding more usable time vector
    LOG.info('Creating new dimensions and variables in %s', netcdf_out)
    times = outds.createVariable('times', 'f4', ('Time'))
    times.units = 'hours since 0001-01-01 00:00:00.0'
    times.calendar = 'gregorian'
    times[:] = dates[:]

    # Adding nx/ny-dims
    outds.createDimension('nx', size=outds.variables['XLAT'].shape[0])
    outds.createDimension('ny', size=outds.variables['XLAT'].shape[1])
    latitude = outds.createVariable('latitude', 'f4', ('nx', 'ny'))
    latitude[:] = outds.variables['XLAT'][:]
    longitude = outds.createVariable('longitude', 'f4', ('nx', 'ny'))
    longitude[:] = outds.variables['XLONG'][:]

    # Flush to disk
    outds.sync()
    outds.close()


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
    setup_output_file(infile, outfile, numdates)
    outds = netCDF4.Dataset(outfile, 'r+', format='NETCDF4')
    outds.set_fill_off()

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
        LOG.info('Working chunk: %s - %s', dates[chunk[0]], dates[chunk[-1]])
        if len(chunk) != chunk_size:
            LOG.info('Last chunk is shorter than previous chunks')
        for invar, outvar in zip(invars, outvars):
            LOG.info('  Working variable %s', invar.name)
            outvar[:, :, chunk] = outvar[:, :, chunk]
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
