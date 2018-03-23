#!/usr/bin/env python3

from netCDF4 import Dataset
import pygrib
import os
import sys
import fnmatch
import numpy as np
import datetime
import argparse
import yaml


def _get_GRIB_date(grib):
    """
        Obtain the initial analise date
    """
    date_anl = grib.analDate

    return date_anl


def _get_GRIB_offset(grib):
    """
        Calculate offset in ours from analises date 
    """
    date_anl = grib.analDate
    date_grb = grib.validDate
    delta = date_grb - date_anl
    offset = delta.days * 24 + delta.seconds / 60 / 60

    return offset


def _get_timestep(offset, step=3):
    """
        Returns the timestep based on the offset from analises and also step size 
    """
    return int(offset // step)


def _push_GRIB_NC(gribfile, grib_fn, nc_file, yml, t=0):
    """
        Insert grib data into nc file
    """
    d0 = _get_GRIB_date(gribfile[1])
    timestep, nc_file, t = NC_is_new(d0, grib_fn, nc_file, yml['file_name'])
    for i, grib in enumerate(gribfile, 0):
        var = grib.cfVarName
        if var == 'unknown':
            var = grib.name
        _var = yml.get(var, ['Null', 'Null', 'Null', 'Null'])
        if _var[0] != 'Null':
            if t == 0 and i == 0:
                suc = create_NC(nc_file, grib)
                if suc == True:
                    suc = append_NC(nc_file, _var, grib, yml[
                                    'time_step'], yml['levels'])
                else:
                    try:
                        suc = append_NC(nc_file, _var, grib, yml[
                                        'time_step'], yml['levels'])
                    except:
                        string = grib_fn.split(".")[0] + ":" + d0.strftime('%Y-%m-%d_%H:%M') + ':'\
                            + grib_fn.split(".")[2] + '.nc'
                        print("Error creating dimensions on file: %s" % (string))
                        exit(1)
            else:
                suc = append_NC(nc_file, _var, grib, yml[
                                'time_step'], yml['levels'])
                if suc == False:
                    string = grib_fn.split(":")[0] + ":" + d0.strftime('%Y-%m-%d_%H:%M') + ':'\
                        + grib_fn.split(":")[2] + '.nc'
                    print("Error appending %s on file: %s" % (_var[0], string))
                    exit(1)

    if suc == True:
        return(True, nc_file)
    else:
        return(False, 'Null')


def NC_is_new(date_anl, grib_fn, nc_file, fn_config):
    """
        Verify if the nc file already exists or is a new one
    """
    fn_config = fn_config.split(":")
    try:
        partial_fn = grib_fn.split(".")
        nc_fn = partial_fn[
            fn_config.index('{name}')] + ":"\
            + partial_fn[fn_config.index('{analdate}')] + ':' +\
            partial_fn[fn_config.index('{member}')] + '.nc'

    except:
        partial_fn = grib_fn.split(":")
        nc_fn = partial_fn[
            fn_config.index('{name}')] + ":"\
            + partial_fn[fn_config.index('{analdate}')] + ':' +\
            partial_fn[fn_config.index('{member}')] + '.nc'

    else:
        print("Unknown filename format, %s" % (grib_fn))
        print("#####", nc_fn, "Will be used")
        # exit(1)

    if os.path.isfile(nc_fn):
        t = 1
        try:
            nc_file = nc_file
            timestep = len(nc_file.variable['Times'])
        except:
            try:
                nc_file = Dataset(nc_fn, 'r+')
                timestep = len(nc_file.variable['Times'])
            except:
                timestep = 0
    else:
        t = 0
        nc_file = Dataset(nc_fn, 'w')
        timestep = 0

    return(timestep, nc_file, t)


def create_NC_dimension(nc_file, shape, size=0):
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


def create_NC(nc_file, grib, comp_lvl=6):
    """
        Create the initial variables and dimensions
    """
    latlons = grib.latlons()
    latlon = np.array(latlons)
    lat = latlon[0]
    lon = latlon[1]
    nc_dm = create_NC_dimension(nc_file, lat.shape)
    if nc_dm == True:
        nc_file.createVariable('XLAT', 'f8', ('south_north', 'west_east'),
                               zlib=True, least_significant_digit=5, complevel=int(comp_lvl))
        nc_file.createVariable('XLONG', 'f8', ('south_north', 'west_east'),
                               zlib=True, least_significant_digit=5, complevel=int(comp_lvl))
        nc_file.createVariable('Times', 'S1', ('Time'), zlib=True, complevel=6)
        nc_file.createVariable('COSALPHA', 'i2', ('south_north', 'west_east'),
                               zlib=True, least_significant_digit=0, complevel=int(comp_lvl))
        nc_file.createVariable('SINALPHA', 'i2', ('south_north', 'west_east'),
                               zlib=True, least_significant_digit=0, complevel=int(comp_lvl))

        nc_file.variables['XLAT'][:] = lat * -1
        nc_file.variables['XLONG'][:] = lon
        shape_ = np.array(lat).shape
        sin = np.zeros((shape_[0], shape_[1]))
        cos = sin
        cos[np.where(cos == 0)] = 1
        nc_file.variables['COSALPHA'][:] = sin
        nc_file.variables['SINALPHA'][:] = cos

        nc_file.START_DATE = grib.analDate.strftime('%Y-%m-%d_%H:%M')
        nc_file.HISTORY = "Created by Belgingur grib2nc utility at " + \
            datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')

        nc_file.sync()
        return True
    else:
        return False


def append_NC(nc_file, _var, grib, step, levels, comp_lvl=6):
    """
        This function append a new timstep to the nc file if it already exists, or create the variable
    """
    offset = _get_GRIB_offset(grib)
    timestep = _get_timestep(offset, step)
    d0 = _get_GRIB_date(grib)
    try:
        nc_file['Times'][timestep] = (
            d0 + datetime.timedelta(hours=offset)).strftime("%Y-%m-%d_%H:00:00")
    except:
        suc = create_NC(nc_file, grib)
        if suc == True:
            nc_file['Times'][timestep] = (
                d0 + datetime.timedelta(hours=offset)).strftime("%Y-%m-%d_%H:00:00")

    try:
        if _var[3] == 'sfc':
            nc_file.variables[_var[0]][timestep, :,
                                       :] = np.flip(grib.values, axis=0)
            return True
        else:
            level = levels.get(grib.level, 0)
            nc_file.variables[_var[0]][timestep, level,
                                       :, :] = np.flip(grib.values, axis=0)
            return True

    except:
        if _var[3] == 'sfc':
            nc_file.createVariable(_var[0], _var[1], ('Time', 'south_north', 'west_east'),
                                   zlib=True, least_significant_digit=int(np.abs(np.log10(_var[2]))), complevel=int(comp_lvl))
            nc_file.variables[_var[0]][timestep, :,
                                       :] = np.flip(grib.values, axis=0)
            return True
        else:
            level = levels.get(grib.level, 0)
            nc_file.createVariable(_var[0], _var[1], ('Time', 'bottom_top', 'south_north', 'west_east'),
                                   zlib=True, least_significant_digit=int(np.abs(np.log10(_var[2]))), complevel=int(comp_lvl))
            nc_file.variables[_var[0]][timestep, level,
                                       :, :] = np.flip(grib.values, axis=0)
            return True

    else:

        return False

#######################################
parser = argparse.ArgumentParser(
    description="This script transform GRIB2 files in to NetCDF")
parser.add_argument('-g', help="Path to a single grib file or folder",
                    action='store', required=True, dest='GRIB_PATH')
# parser.add_argument('-o', help="Output path for nc file.", action='store', required=False, dest='OUTPUT_PATH')
parser.add_argument('-c', help="Configuration file(YAML) for especific model",
                    action='store', required=True, dest='yml')
args = parser.parse_args()

path_grib = args.GRIB_PATH
# path_out = args.OUTPUT_PATH
yml_file = args.yml

with open(yml_file, 'r') as yf:
    config_yml = yaml.safe_load(yf)

try:
    files = [f for f in sorted(os.listdir(path_grib)) if fnmatch.fnmatch(f, '*.grb2')
             or fnmatch.fnmatch(f, '*.grib2')]
except:
    files = [path_grib]

nc_file = 'Null'
# nc_file = 'Null' or Dataset(path_out, 'w')
for t, file in enumerate(files, 0):
    try:
        gribfile = pygrib.open(path_grib + file)
    except:
        gribfile = pygrib.open(file)
    suc, nc_file = _push_GRIB_NC(gribfile, file, nc_file, config_yml, t)
    if suc != True:
        print('Error open %s' % file)
        exit(1)
    else:
        print('File %i (%s) of %i done' % (int(t + 1), file, int(len(files))))

    nc_file.close()
