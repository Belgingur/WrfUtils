#!/usr/bin/env python
# encoding: utf-8

"""
Generates point-weight masks from a WRF-style geography file and shape files.
"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals

# Python library imports
import argparse
from itertools import cycle, count
import netCDF4
import os
import subprocess

import numpy
import pylab
from mpl_toolkits.basemap import Basemap
import shapely.geometry
import ogr
import osr
import sys
import yaml

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


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
    id1 = subprocess.Popen([
        'gm', 'convert', '-trim', '+repage',
        plot_name, plot_name
    ])
    id1.wait()


def sub_sample_grid(xlon, xlat, subres):
    """
    Create an array in the shape of the flattened xlon or xlat, containing at
    each position an array of subres*subres sub-sampling points for that cell.
    """
    print('Create %s*%s sub-sampling points in each of %s*%s cells' %
          (subres, subres, xlon.shape[0], xlon.shape[1]))
    i_lim, j_lim = xlon.shape
    sub_lim = subres // 2
    sub_points = []
    for i in range(0, i_lim):
        for j in range(0, j_lim):
            # Find lat0,lon0 centre of grid cell i,j
            # and d_lon, d_lat the sub-sampling steps
            lon0, lat0 = xlon[i, j], xlat[i, j]
            i_b, i_a = max(i - 1, 0), min(i + 1, i_lim - 1)
            j_b, j_a = max(j - 1, 0), min(j + 1, j_lim - 1)
            d_lon = (xlon[i, j_a] - xlon[i, j_b]) / subres / (j_a - j_b)
            d_lat = (xlat[i_a, j] - xlat[i_b, j]) / subres / (i_a - i_b)
            points = []
            sub_points.append(points)
            for sub_i in range(-sub_lim, sub_lim + 1):
                lon = lon0 + sub_i * d_lon
                for sub_j in range(-sub_lim, sub_lim + 1):
                    lat = lat0 + sub_j * d_lat
                    point = shapely.geometry.Point(lon, lat)
                    points.append(point)
    return sub_points


def points_in_polygon(poly, sub_points_per_cell, shape):
    """
    Counts the number of sub-sampling points in each cell which fall within the
    polygon.
    """
    # Some would first transform data to e.g. km using e.g. lambert-projection

    # Step through elements in grid to evaluate if in the study area
    weights = numpy.zeros(shape[0] * shape[1], dtype=int)
    for i, sub_points in enumerate(sub_points_per_cell):
        for point in sub_points:
            if point.within(poly):
                weights[i] += 1
    weights = weights.reshape(shape)
    return weights


def read_shapefile(shpfile, tx):
    """
    Reads a shapefile and returns a list of polygons, each of which is a list of points.

    :param shpfile:
    :param tx:
    :rtype: list of list of int
    """
    # Now open the shapefile and start by reading the only layer and
    # the first feature, as well as its geometry.
    source = ogr.Open(shpfile)
    layer = source.GetLayer()
    counts = layer.GetFeatureCount()
    polygons = []
    for c in range(counts):
        feature = layer.GetFeature(c)
        geometry = feature.GetGeometryRef()

        # Do the coordinate transformation
        geometry.Transform(tx)

        # Read the polygon (there is just one) and the defining points.
        polygon = geometry.GetGeometryRef(0)
        polygons.append(polygon.GetPoints())

    # Stand and deliver
    return polygons


def coord_transform(from_sr_id, to_sr_id):
    # Remember to set: export GDAL_DATA="/usr/share/gdal/1.10/", or
    # equivalent for version of gdal
    from_sr = osr.SpatialReference()
    from_sr.ImportFromEPSG(from_sr_id)
    to_sr = osr.SpatialReference()
    to_sr.ImportFromEPSG(to_sr_id)
    tx = osr.CoordinateTransformation(from_sr, to_sr)
    return tx


def plot_data(m, plot_file, results, x, y, xhgt, levels=9):
    # Plot the map!
    print('Plot map')
    pre_plot(m, x, y, xhgt)
    for (shape_file, polygons, weights), color in zip(results, cycle(COLORS)):
        print('Plot for', shape_file)
        name = os.path.basename(shape_file)
        name = os.path.splitext(name)[0]
        for i, polygon, weight in zip(count(), polygons, weights):
            print('  weight: %0.2f' % (numpy.sum(weight) / levels))
            x_p, y_p = m(polygon.exterior.xy[0], polygon.exterior.xy[1])
            label = name if i == 0 else None
            m.plot(x_p, y_p, color + '-', lw=2, label=label, alpha=0.5)
            for level in range(levels):
                mask1 = weight <= level + 1
                mask2 = weight > level
                mask = mask1 & mask2
                size = 2 * level / levels
                m.plot(x[mask], y[mask], color + 'o', ms=size, mec=color)
    pylab.legend(ncol=2, fontsize='x-small')
    post_plot(plot_file)


def main():
    config, plot_file = read_config()

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

    results = []
    subres = config['sub_sampling']
    sub_points = sub_sample_grid(xlon, xlat, subres)
    for shape_file in config['shape_files']:
        print('Read shape file', shape_file)
        polygon_points = read_shapefile(shape_file, tx)
        polygon_shapes = [shapely.geometry.Polygon(polygon)
                          for polygon in polygon_points]
        weights = [points_in_polygon(shape, sub_points, xlon.shape)
                   for shape in polygon_shapes]
        results.append((shape_file, polygon_shapes, weights))

    if plot_file:
        plot_data(m, plot_file, results, x, y, xhgt, subres ** 2)


main()
