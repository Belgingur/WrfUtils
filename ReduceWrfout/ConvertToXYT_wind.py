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


def setup_output_file(netcdf_in, netcdf_out, dates):
    # First preprocess dataset
    logging.info('Running ncks on output file %s', netcdf_out)
    cmd = ['ncks', '-O', '-4', '-v', 'HGT,XLAT,XLONG',
           '-d', 'Time,0',
           '--cnk_dmn', 'south_north,16', '--cnk_dmn', 'west_east,16',
           netcdf_in, netcdf_out + '_tmp']
    try:
        __ = subprocess.check_call(cmd)
    except:
        logging.error('Error ncks!')
        sys.exit(-1)
    # First make sure surface fields will be two-dim
    logging.info('Running ncwa on output file %s', netcdf_out)
    cmd = ['ncwa', '-O', '-4', '-a', 'Time',
           netcdf_out + '_tmp', netcdf_out + '_tmp2']
    try:
        subprocess.check_call(cmd)
    except:
        logging.error('Error ncwa!')
        sys.exit(-1)
    # Transpose file
    logging.info('Running ncpdq on output file %s', netcdf_out)
    cmd = ['ncpdq', '-O', '-4', '-a',
           'west_east,south_north',
           netcdf_out + '_tmp2', netcdf_out]
    try:
        __ = subprocess.check_call(cmd)
    except:
        logging.error('Error ncpdq!')
        sys.exit(-1)
    # Add original time vector
    logging.info('Running ncks on output file %s', netcdf_out)
    cmd = ['ncks', '-A', '-4', '-v', 'Times',
           netcdf_in, netcdf_out]
    try:
        __ = subprocess.check_call(cmd)
    except:
        logging.error('Error ncks')
        sys.exit(-1)

    # Adding time vector, nx/ny-dims
    logging.info('Creating new dimensions and variables in %s',
                 netcdf_out)
    outds = netCDF4.Dataset(netcdf_out, mode='r+', keepweakref=True)
    times = outds.createVariable('times', 'f4', ('Time'))
    times.units = 'hours since 0001-01-01 00:00:00.0'
    times.calendar = 'gregorian'
    times[:] = dates[:]
    outds.createDimension('nx', size=outds.variables['XLAT'].shape[0])
    outds.createDimension('ny', size=outds.variables['XLAT'].shape[1])
    latitude = outds.createVariable('latitude', 'f4', ('nx', 'ny'),
                                    chunksizes=(16, 16))
    latitude[:] = outds.variables['XLAT'][:]
    longitude = outds.createVariable('longitude', 'f4', ('nx', 'ny'),
                                     chunksizes=(16, 16))
    longitude[:] = outds.variables['XLONG'][:]

    # Some new attributes to be set and adding time vector, nx/ny-dims
    logging.info('Setting/updating global file attributes for output file %s',
                 netcdf_out)
    outds.description2 = 'Copy of %s with reversed order of dimensions' \
                         % (netcdf_in)
    outds.history2 = 'Created with python: ' + \
                     str(time.ctime()) \
                     + ', by Halfdan Agustsson'
    outds.institution = 'KVT'
    outds.project = 'Statnett database'
    outds.source2 = '%s' % (netcdf_in)
    outds.sync()
    outds.close()

    # Remove temporary files
    os.remove(netcdf_out + '_tmp')
    os.remove(netcdf_out + '_tmp2')


def work_wrf_dates(times):
    # Work our date array
    logging.info('Working dates %s to %s',
                 times[0].tostring(), times[-1].tostring())
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


#############################################################
# Parameters/constants for input data

# infile = 'wrfout_d03_2008-08-16_00:00:00_vegagerd.nc'
chunkSize_days = 10
chunks_fixed = (16, 16, 128)
deflatelevel = 6
siglev = 0
least_sig_dig = {'wd': 1,
                 'ws': 2,
                 't2': 2,
                 'rainnc': 1,
                 'swdown': 1,}
scale_factor = {'wd': 0.1,
                'ws': 0.01,
                't2': 0.01,
                'rainnc': 0.1,
                'swdown': 0.1,}
add_offset = {'wd': 0,
              'ws': 0,
              't2': 273,
              'rainnc': 0,
              'swdown': 0,}
datatype = {'wd': 'u2',
            'ws': 'u2',
            't2': 'i2',
            'rainnc': 'u2',
            'swdown': 'u2',}
desc = {'ws': 'Wind speed',
        'wd': 'Wind direction'}

############################################################
# The main routine!

# Timing
start_time = time.time()

# Setup logging
logging.config.fileConfig('./logging.conf')
logging.info('Have set up logging')

# Now we calculate rime icing
logging.info('Let us do some extra-dimensional wonders...')

# Now read input argument
logging.info('Reading input argument')
parser = argparse.ArgumentParser(description='Define input file')
parser.add_argument('infile', metavar='-i',
                    help='input file from wrf to to process')
args = parser.parse_args()
infile = args.infile
outfile_10m = infile + '_ws10m_xyt.nc4'
outfile_lev1 = infile + '_wslev1_xyt.nc4'

# Open input datasets
logging.info('Opening input dataset %s', infile)
inds = netCDF4.Dataset(infile, 'r')
times = inds.variables['Times']
u_st = inds.variables['U']
v_st = inds.variables['V']
u10 = inds.variables['U10']
v10 = inds.variables['V10']
sina = inds.variables['SINALPHA'][0]
cosa = inds.variables['COSALPHA'][0]

# Convert time vector
logging.info('Converting time vector')
dates = work_wrf_dates(times)
numdates = netCDF4.date2num(dates, 'hours since 0001-01-01 00:00:00.0',
                            calendar='gregorian')

# Preprocess output files
logging.info('Initialize output file %s', outfile_10m)
if os.path.exists(outfile_10m):
    logging.warning('Will overwrite already existing %s', outfile_10m)
logging.info('Initializing outfile %s', outfile_10m)
setup_output_file(infile, outfile_10m, numdates)
outds_10m = netCDF4.Dataset(outfile_10m, 'r+', format='NETCDF4')
outds_10m.set_fill_off()
logging.info('Initialize output file %s', outfile_lev1)
if os.path.exists(outfile_lev1):
    logging.warning('Will overwrite already existing %s', outfile_lev1)
logging.info('Initializing outfile %s', outfile_lev1)
setup_output_file(infile, outfile_lev1, numdates)
outds_lev1 = netCDF4.Dataset(outfile_lev1, 'r+', format='NETCDF4')
outds_lev1.set_fill_off()

# Set chunks
dt = (dates[1] - dates[0]).seconds
size_t = len(times)
indices = range(0, size_t)
chunkSize_prefered = int(chunkSize_days * 24. * 3600. / dt)
if size_t < chunkSize_prefered:
    chunkSize = size_t
else:
    chunkSize = chunkSize_prefered
logging.info('Using chunk size: ' + str(chunkSize))

# Create 10 m wind variables in output file
logging.info('Create variable ws10m')
ws_10m = outds_10m.createVariable('ws', datatype['ws'],
                                  dimensions=('nx', 'ny', 'Time'),
                                  zlib=True, complevel=deflatelevel,
                                  shuffle=True,
                                  least_significant_digit=least_sig_dig['ws'],
                                  chunksizes=chunks_fixed)
ws_10m.description = desc['ws']
ws_10m.level = '10 m a.g.l.'
ws_10m.scale_factor = scale_factor['ws']
ws_10m.add_offset = add_offset['ws']
logging.info('Create variable wd10m')
wd_10m = outds_10m.createVariable('wd', datatype['wd'],
                                  dimensions=('nx', 'ny', 'Time'),
                                  zlib=True, complevel=deflatelevel,
                                  shuffle=True,
                                  least_significant_digit=least_sig_dig['wd'],
                                  chunksizes=chunks_fixed)
wd_10m.description = desc['wd']
wd_10m.level = '10 m a.g.l.'
wd_10m.scale_factor = scale_factor['wd']
wd_10m.add_offset = add_offset['wd']
outds_10m.sync()

# Create sigma level 1 wind variables in output file
logging.info('Create variable wslev1')
ws_lev1 = outds_lev1.createVariable('ws', datatype['ws'],
                                    dimensions=('nx', 'ny', 'Time'),
                                    zlib=True, complevel=deflatelevel,
                                    shuffle=True,
                                    least_significant_digit=least_sig_dig['ws'],
                                    chunksizes=chunks_fixed)
ws_lev1.description = desc['ws']
ws_lev1.level = '20 m a.g.l.'
ws_lev1.scale_factor = scale_factor['ws']
ws_lev1.add_offset = add_offset['ws']
logging.info('Create variable wdlev1')
wd_lev1 = outds_lev1.createVariable('wd', datatype['wd'],
                                    dimensions=('nx', 'ny', 'Time'),
                                    zlib=True, complevel=deflatelevel,
                                    shuffle=True,
                                    least_significant_digit=least_sig_dig['wd'],
                                    chunksizes=chunks_fixed)
wd_lev1.description = desc['wd']
wd_lev1.level = '20 m a.g.l.'
wd_lev1.scale_factor = scale_factor['wd']
wd_lev1.add_offset = add_offset['wd']
outds_lev1.sync()

# Start the loop through time
logging.info('Starting to loop through data')
for t in xrange(0, size_t, chunkSize):
    chunk = indices[t:t + chunkSize]
    logging.info('Working chunk: %s - %s',
                 dates[chunk[0]], dates[chunk[-1]])
    if len(chunk) != chunkSize:
        logging.info('Last chunk is shorter than previous chunks')
    logging.info('Working 10 m wind for chunk')
    ws_10m[:, :, chunk] = numpy.sqrt(u10[chunk] ** 2 + v10[chunk] ** 2).T
    wd_10m[:, :, chunk] = calc_wind_dir(u10[chunk], v10[chunk],
                                        sina, cosa).T
    outds_10m.sync()
    logging.info('Working level %s wind for chunk', siglev)
    u = .5 * (u_st[chunk, siglev, :, :-1] + u_st[chunk, siglev, :, 1:])
    v = .5 * (v_st[chunk, siglev, :-1, :] + v_st[chunk, siglev, 1:, :])
    ws_lev1[:, :, chunk] = numpy.sqrt(u ** 2 + v ** 2).T
    wd_lev1[:, :, chunk] = calc_wind_dir(u, v, sina, cosa).T
    outds_lev1.sync()

# Close our datasets
inds.close()
outds_10m.close()
outds_lev1.close()

# Time elapsed
logging.info('Finished')
logging.info('Timing: %.1f (seconds) ', time.time() - start_time)

################################################
