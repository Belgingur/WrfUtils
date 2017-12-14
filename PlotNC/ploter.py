#!/usr/bin/env python3

from netCDF4 import Dataset
import argparse
# import numpy as np
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def gen_map_plot(right_c, left_c, file_name, data, scat_lat=0, scat_lon=0):
    """
    Efectivile creates the plot,
    lat [-90, 90]
    lon [-180, 180] or [0, 360]
    """
    try:
        fig1 = plt.figure(file_name,figsize=(16, 9))
        map1 = Basemap(projection='cyl',llcrnrlat=left_c[0], urcrnrlat=right_c[0], llcrnrlon=left_c[1], urcrnrlon=right_c[1], resolution='h')
        map1.drawcoastlines()
        map1.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
        map1.fillcontinents(color='lightgray', zorder=0)

        ny = data.shape[1]
        nx = data.shape[2]
        lons, lats = map1.makegrid(nx, ny)
        x, y = map1(lons, lats)

        map1.contourf(x, y, data[0,:,:])
        map1.colorbar(location='right', size='5%', pad='2%')
        map1.scatter(scat_lon, scat_lat, s=30, marker='o', latlon=True, linewidth=0, )

        plt.savefig(file_name, dpi=300, pad_inches=0)

        return True
    except:
        return False


parser = argparse.ArgumentParser(description="This script genarete plots from NetCDF4")
parser.add_argument('-i', help="Imput NetCDF4 file", action='store',required=True, dest='nc_path')
parser.add_argument('-o', help="Output destination", action='store',required=False, dest='out_path')
parser.add_argument('-v', help="Variables list", action='store',required=False, dest='nc_vars')
parser.add_argument('-l', help="Lat0, Lon0, Lat1, Lon2", action='store',required=False, dest='lat_lon')
parser.add_argument('-s', help="Scatter points yml file", action='store',required=False, dest='scat_vars')

args=parser.parse_args()

nc_path = args.nc_path
nc_vars = args.nc_vars or ['T2','RAINNC', 'Q2', 'PSFC'] 
out_path = args.out_path or './'
left_c, right_c = [args.lat_lon[0], args.lat_lon[1]], [args.lat_lon[2], args.lat_lon[3]] or [-89.9, -179.9], [89.9, 179.9]
scat_file = args.scat_vars or 'Null'

if scat_file != 'Null':
    with open(scat_file,'r') as yf:
        scat_yml = yaml.safe_load(yf)

    scat_lat = scat_yml['lat']
    scat_lon = scat_yml['lon']

nc_file =  Dataset(nc_path, 'r')

for var in nc_vars:
    data = nc_file.variables[var]
    for i in range(0, len(data)):
        file_name = out_path + var + str(i) + '.png'
        suc = gen_map_plot(right_c, left_c, file_name, data[i,:,:], scat_lat, scat_lon)
        if suc == True:
            s = 'plot %s, timestep %s done!' %(var, i)
            print(s)
        else:
            s = 'Fail to plot %s, timestep %s' %(var, i)
            print(s)

nc_file.close()
