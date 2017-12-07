#!/usr/bin/env python
#-*- coding:utf-8 -*-

from netCDF4 import Dataset
import pygrib
import os
import sys
import fnmatch
import numpy as np
import datetime

def _get_GRIB_VARS(var):
	DIC = {
'latitude'								: ['XLAT',		'u3', 0.00001   , 'sfc' ],
'longitude'								: ['XLONG',		'u3', 0.00001   , 'sfc' ],
'spfh2m'								: ['Q2',		'i4', 0.00000001, 'sfc' ],
'q'										: ['Q',			'i2', 0.00000001, 'ver' ],
't2m'									: ['T2',		'u2', 0.01      , 'sfc' ],
't'										: ['T',			'u2', 0.01      , 'ver' ],
'sp'									: ['PSFC',		'u4', 0.01      , 'sfc' ],
'pres'									: ['P',			'i2', 0.01      , 'sfc' ],
'u10'									: ['U10',		'u2', 0.01      , 'sfc' ],
'v10'									: ['V10',		'u2', 0.01      , 'sfc' ],
'u'										: ['U',			'i2', 0.01      , 'sfc' ],
'v'										: ['V',			'i2', 0.01      , 'sfc' ],
'w'										: ['W',			'i2', 0.01      , 'sfc' ],
'r2'									: ['RH2',		'u2', 0.01      , 'sfc' ],
'msletmsl'								: ['MSLP',		'u4', 0.01      , 'sfc' ],
'cwatclm'								: ['QCLOUD',	'i4', 0.000001  , 'sfc' ],
'st'									: ['TSLB',		'u2', 0.01      , 'sfc' ],
'sm'									: ['SMOIS'		'u2', 0.0001    , 'sfc' ],
'gfluxsfc'								: ['GRDFLX',	'i2', 0.1       , 'sfc' ],
'sdwe'									: ['SNOW',		'u4', 0.1       , 'sfc' ],
'sde'									: ['SNOWH',		'u2', 0.001     , 'sfc' ],
'gh'									: ['GHT',      	'i2', 10        , 'ver' ],
'snod'									: ['SNOWNC',	'u4', 0.01      , 'sfc' ],
'acpcp'									: ['RAINC',		'u2', 0.01      , 'sfc' ],
'apcp'									: ['RAINNC',	'u2', 0.01      , 'sfc' ],
'Downward short-wave radiation flux'	: ['SWDOWN',	'u2', 0.01      , 'sfc' ],
'Downward long-wave radiation flux'		: ['GLW',		'u2', 0.01      , 'sfc' ],
'hpbl'									: ['PBLH',		'u4', 0.1       , 'sfc' ],
'csnow'									: ['SNOWC',		'u1', 0.01      , 'sfc' ],
'cpofp'									: ['SR',		'u1', 0.01      , 'sfc' ],
'gust'									: ['GUST',		'u2', 0.01      , 'sfc' ]
}
	return DIC.get(var, ['Null', 'Null', 'Null', 'Null'])

def _get_GRIB_DATE(grib):
	"""
		Obtain the initial analise date
	"""
	date_anl = grib.analDate

	return date_anl

def _get_GRIB_TIMEDELTA(grib):
	"""
		Calculate timedelta in ours from analises date 
	"""
	date_anl = grib.analDate
	date_grb = grib.validDate
	delta = date_grb - date_anl
	timedelta = delta.days*24 + delta.seconds/60/60

	return timedelta

def _get_timestep(timedelta, step=3):
	"""
		Returns the timestep based on the timedelta form analises and configurtation of upstrean
	"""
	return timedelta//step

def _push_GRIB_NC(gribfile, grib_fn, nc_file,t=0):
	"""
		
	"""
	d0 = _get_GRIB_DATE(gribfile[1])
	timestep, nc_file = NC_is_new(d0, grib_fn, nc_file)
	for i, grib in enumerate(gribfile,0):
		var = grib.cfVarName
		if var == 'unknown':
			var = grib.name
		_var = _get_GRIB_VARS(var)
		if _var[0] != 'Null':
			if t == 0 and i == 0:
				suc = CREATE_NC(nc_file, grib)
				if suc == True:
					suc = APPEND_NC(nc_file, _var, grib)
				else:
					try:
						suc = APPEND_NC(nc_file, _var, grib)
					except:				
						string = grib_fn.split(".")[0] + ":" + d0.strftime('%Y-%m-%d_%H:%M') + ':' \
																	+ grib_fn.split(".")[2] + '.nc'
						print "Error creating dimensions on file: %s" %(string)	
						exit(1)
			else:
				suc = APPEND_NC(nc_file, _var, grib)
				if suc == False:
					string = grib_fn.split(":")[0] + ":" + d0.strftime('%Y-%m-%d_%H:%M') + ':' \
																	+ grib_fn.split(":")[2] + '.nc'
					print "Error appending %s on file: %s" %(_var[0],string)	
					exit(1)

	if suc == True:
		return(True, nc_file)
	else:
		return(False, 'Null')

def NC_is_new(date_anl, grib_fn, nc_file):
	"""
		Verify if the nice file already exists or is a new one
	"""
	try:
		parcial_fn = grib_fn.split(":")
		nc_fn = parcial_fn[0] + ":" + date_anl.strftime('%Y-%m-%d_%H:%M') + ':' + parcial_fn[2][:10] + '.nc'
	except:
		parcial_fn = grib_fn.split(".")
		nc_fn = parcial_fn[0] + ":" + date_anl.strftime('%Y-%m-%d_%H:%M') + ':' + parcial_fn[2][:10] + '.nc'
	else:
		print "Unknown filename format, %s" %(grib_fn)
		exit(1)
	if os.path.isfile(nc_fn):
		try:
			nc_file = nc_file
			timestep = len(nc_file.variable['times'])
		except:
			try:
				nc_file = Dataset(nc_fn, 'w')
				timestep = len(nc_file.variable['times'])
			except:
				timestep = 0
	else:
		nc_file = Dataset(nc_fn, 'w')
		timestep = 0

	return(timestep, nc_file)

def CREATE_NC_DIMENSION(nc_file, shape, size=0):
	"""
		Time = UNLIMITED ; bottom_top = 40 ; south_north = 192 ; west_east = 192 ;
	"""
	try:
		nc_file.createDimension("Time", size)
		nc_file.createDimension("bottom_top", None)
		nc_file.createDimension("south_north", shape[0])
		nc_file.createDimension("west_east", shape[1])
		return True
	except:
		return False

def CREATE_NC(nc_file, grib, comp_lvl=6):
	"""
		Create the initial variables and dimensions
	"""
	latlons = grib.latlons()
	latlon = np.array(latlons)
	lat = latlon[0,:,:]
	lon = latlon[1,:,:]

	nc_dm = CREATE_NC_DIMENSION(nc_file, lat.shape)

	if nc_dm == True:
		xlat=nc_file.createVariable('XLAT', 'u4', ('south_north', 'west_east'), zlib=True, \
																	least_significant_digit=3, complevel=int(comp_lvl))
		xlon=nc_file.createVariable('XLONG','u4', ('south_north', 'west_east'), zlib=True, \
																	least_significant_digit=3, complevel=int(comp_lvl))
		xlat = lat
		xlon = lon
		times=nc_file.createVariable('times', 'S2', ('Time'), zlib=True, complevel=6)
		nc_file.sync()
		return True

	else:
		return False

def APPEND_NC(nc_file, _var, grib, comp_lvl=6):
	"""
		This function append a new timstep to the nc file if it already exists, or create the variable
	"""
	timedelta = _get_GRIB_TIMEDELTA(grib)
	timestep = _get_timestep(timedelta)
	d0 = _get_GRIB_DATE(grib)
	try:
		if _var[3] == 'sfc': 
			nc_var = nc_file.variables[_var[0]]
			nc_var[timestep, :, :] = grib.values
			time = nc_file.variables['times']
			time[timestep] = (d0 + datetime.timedelta(hours=timedelta)).strftime("%Y-%m-%d_%H")
			return True
		else:
			nc_var = nc_file.variables[_var[0]]
			nc_var[timestep, :, :, grib.level] = grib.values
			time = nc_file.variables['times']
			time[timestep] = (d0 + datetime.timedelta(hours=timedelta)).strftime("%Y-%m-%d_%H")
			return True

	except:
		if _var[3] == 'sfc': 
			nc_var = nc_file.createVariable(_var[0], _var[1], ('Time', 'south_north', 'west_east'), \
							zlib=True, least_significant_digit=int(np.abs(np.log10(_var[2]))), complevel=int(comp_lvl))
			nc_var[timestep, :, :] = grib.values
			time = nc_file.variables['times']
			time[timestep] = (d0 + datetime.timedelta(hours=timedelta)).strftime("%Y-%m-%d_%H")
			return True
		else:
			nc_var = nc_file.createVariable(_var[0], _var[1], ('Time', 'south_north', 'west_east', 'bottom_top'),\
							 zlib=True,	least_significant_digit=int(np.abs(np.log10(_var[2]))), complevel=int(comp_lvl))
			nc_var[timestep, :, :, grib.level] = grib.values
			time = nc_file.variables['times']
			time[timestep] = (d0 + datetime.timedelta(hours=timedelta)).strftime("%Y-%m-%d_%H")
			return True

	else:

		return False

#######################################
path_grib	= (sys.argv[1]) #upstream grib2 folder
path_out	= (sys.argv[2]) #output file full path

try:
	files = [f for f in sorted(os.listdir(path_grib)) if fnmatch.fnmatch(f, '*.grb2') \
																		or fnmatch.fnmatch(f, '*.grib2')]
except:
	files = path_grib.split("/")[-1]
nc_file = 'Null'
for t, file in enumerate(files,0):
	gribfile = pygrib.open(path_grib+file)
	suc, nc_file = _push_GRIB_NC(gribfile, file, nc_file, t)
	if suc != True:
		print 'Error open %s' %file
		exit(1)
	else:
		print 'File %i (%s) of %i done' % (int(t+1), file, int(len(files)))

nc_file.close
