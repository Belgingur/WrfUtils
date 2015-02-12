#!/usr/bin/env python
# encoding: utf-8

"""
Generates point-weight masks from a WRF-style geography file and shape files.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Python library imports
import argparse
import netCDF4
import subprocess

import numpy
import pylab
from mpl_toolkits.basemap import Basemap
import shapely.geometry
import ogr
import osr
import sys
import yaml


def read_config():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        epilog=None
    )
    parser.add_argument('--config', default='make_masks.yml',
                        help='Read configuration from this file (def: make_masks.yml)')
    parser.add_argument('--plot', type=str, default='map.png',
                        help='Plot the generated masks to this file (def: map.png)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
        return config, args.plot


def pre_plot(m, x, y, xhgt):
    pylab.figure(figsize=(10, 8))
    m.drawcoastlines(linewidth=1.5, color='black')
    m.readshapefile('joklar/joklar', 'joklar', linewidth=1.5, color='black')
    m.drawparallels(numpy.arange(63., 67., 1.), labels=[1, 0, 0, 0])
    m.drawmeridians(numpy.arange(-26., -12., 2.), labels=[0, 0, 0, 1])
    m.drawmapscale(-15., 63.5, -19., 65., 100., barstyle='fancy', fontsize=12)
    iso = numpy.arange(200., 2300., 200.)
    m.contour(x, y, xhgt, iso, colors='black', linewidths=0.4)


def post_plot(plot_name):
    print('Write plot', plot_name)
    pylab.savefig(plot_name, dpi=300)
    pylab.clf()
    pylab.close()
    id1 = subprocess.Popen(['gm', 'convert', '-trim', '+repage', plot_name, plot_name])
    id1.wait()


def points_area(poly, xlon, xlat):
    print('Using shapely to find area to scale')
    # Some would first transform data to e.g. km using e.g. lambert-projection

    # Step through elements in grid to evaluate if in the study area
    grid_lons = xlon.flatten()
    grid_lats = xlat.flatten()
    mask = numpy.zeros_like(grid_lons, dtype='bool')
    for i in range(len(grid_lons)):
        grid_point = shapely.geometry.Point(grid_lons[i], grid_lats[i])
        if grid_point.within(poly):
            mask[i] = True

    # Stand and deliver
    return mask.reshape(xlon.shape)


def read_shapefile(shpfile, tx):
    # Now open the shapefile and start by reading the only layer and
    # the first feature, as well as its geometry.
    source = ogr.Open(shpfile)
    layer = source.GetLayer()
    counts = layer.GetFeatureCount()
    points = []
    for c in range(counts):
        feature = layer.GetFeature(c)
        geometry = feature.GetGeometryRef()

        # Do the coordinate transformation
        geometry.Transform(tx)

        # Read the polygon (there is just one) and the defining points.
        polygon = geometry.GetGeometryRef(0)
        points.append(polygon.GetPoints())

    # Stand and deliver
    return points


def coord_transform(from_sr_id, to_sr_id):
    # Remember to set: export GDAL_DATA="/usr/share/gdal/1.10/", or
    # equivalent for version of gdal
    from_sr = osr.SpatialReference()
    from_sr.ImportFromEPSG(from_sr_id)
    to_sr = osr.SpatialReference()
    to_sr.ImportFromEPSG(to_sr_id)
    tx = osr.CoordinateTransformation(from_sr, to_sr)
    return tx


def main():
    config, plot_file = read_config()
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Define the coordinate transformation needed for shapefiles
    tx = coord_transform(config['shape_spatial_reference'],
                         config['plot_spatial_reference'])

    # Read input data
    print('Read', config['geofile'])
    with netCDF4.Dataset(config['geofile']) as dataset:
        xhgt = dataset.variables['HGT_M'][0]
        xlon = dataset.variables['XLONG_M'][0]
        xlat = dataset.variables['XLAT_M'][0]

    # Set up map
    print('Create map projection')
    m = Basemap(projection='stere', lat_ts=65., lat_0=65., lon_0=-19.,
                llcrnrlon=-24.2, llcrnrlat=63.25,
                urcrnrlon=-13.1, urcrnrlat=66.5,
                resolution='i' if plot_file else None)
    x, y = m(xlon, xlat)
    
    # Plot the map!
    print('Plot map')
    pre_plot(m, x, y, xhgt)
    for shpfile, color in zip(config['shape_files'], colors):
        print('Read shape file', shpfile)
        points = read_shapefile(shpfile, tx)
        print('Create polygon')
        for p in points:
            poly = shapely.geometry.Polygon(p)
            print('Find grid points within polygon')
            mask = points_area(poly, xlon, xlat)
            m.plot(x[mask], y[mask], color + 'o', ms=2, mec=color)
            x_p, y_p = m(poly.exterior.xy[0], poly.exterior.xy[1])
            m.plot(x_p, y_p, color + '-', lw=2, label=shpfile[11:], alpha=0.5)
    pylab.legend(ncol=2, fontsize='x-small')
    post_plot(plot_file)


main()
