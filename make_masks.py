#!/usr/bin/env python
# encoding: utf-8

"""
Generates point-weight masks from a WRF-style geography file and shape files.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Python library imports
import argparse
from itertools import product
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


# SETUP

def read_config():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        epilog=None
    )
    parser.add_argument('--config', default='make_masks.yml',
                        help='Read configuration from this file (def: make_masks.yml)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Plot the generated masks to this file (def: map.png)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
        return config, args.plot


def sub_sample_grid(xlon, xlat, subres):
    """
    Create an array in the shape of the flattened xlon or xlat, containing at
    each position an array of subres*subres sub-sampling points for that cell.
    """
    print('Create %s*%s sub-sampling points in each of %s*%s cells' %
          (subres, subres, xlon.shape[0], xlon.shape[1]))
    i_lim, j_lim = xlon.shape
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
            # print('% 3.0f % 3.0f\t% 2.4f % 2.4f\t% 2.4f % 2.4f' %
            # (i, j, lat0, lon0, d_lat, d_lon))

            # Make an array of subres*subres evenly spaced points inside cell
            points = []
            sub_points.append(points)
            for sub_i in range(subres):
                sub_i -= (subres - 1) / 2
                lon = lon0 + sub_i * d_lon
                # print(end='\t')
                for sub_j in range(subres):
                    sub_j -= (subres - 1) / 2
                    lat = lat0 + sub_j * d_lat
                    # print('% 2.4f % 2.4f' % (lat, lon), end='\t')
                    point = shapely.geometry.Point(lon, lat)
                    points.append(point)
                    # print()
    return sub_points


def read_shapefile(shapefile, tx):
    """
    Reads a shapefile and returns a list of polygons, each of which is a list of points.
    :rtype: list of Geometry
    """
    # Now open the shapefile and start by reading the only layer and
    # the first feature, as well as its geometry.
    source = ogr.Open(shapefile)
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


# CALCULATE WEIGHTS

class ByFile(object):
    """
    Holds a list of polygons read from a shape file, along with region name
    and weight matrix for each.
    """

    def __init__(self, file, polygons, regions, weights):
        self.file = file
        self.polygons = polygons or []
        self.regions = regions or []
        self.weights = weights or []

    def add(self, polygon, region, weight):
        self.polygons.append(polygon)
        self.regions.append(region)
        self.weights.append(weight)

    def label(self):
        return plain_name(self.file)


class ByRegion(object):
    """
    Holds a list of polygons that define a region and the sum of their weight
    matrices.
    """

    def __init__(self, region, grid_shape):
        self.region = region
        self.polygons = []
        self.weights = numpy.zeros(grid_shape, dtype=int)

    def add(self, polygon, weights):
        self.polygons.append(polygon)
        self.weights += weights

    def label(self):
        return self.region


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


def weigh_shapefiles(shape_files, tx, grid_shape, sub_points, levels, res):
    """
    Reads shapefiles and for each polygon, calculate a weight matrix and store
    along with the name of the region.
    """
    results = []
    for shape_file, regions in shape_files.items():
        print('Read %s -> %s' % (shape_file, ', '.join(regions)))
        polygon_points = read_shapefile(shape_file, tx)
        polygon_shapes = [shapely.geometry.Polygon(polygon)
                          for polygon in polygon_points]
        weights = [points_in_polygon(shape, sub_points, grid_shape)
                   for shape in polygon_shapes]
        results.append(ByFile(shape_file, polygon_shapes, regions, weights))

        if len(weights) != len(regions):
            print('WARNING: %s regions do not correspond to %s read polygons',
                  len(regions), len(weights))
        for weight, region in zip(weights, regions):
            area = numpy.sum(weight) / levels * res[0] * res[1]
            print('  %s: %0.2f km²' % (region, area / 1000000))

    return results


def shuffle_weights(weights_by_file, grid_shape):
    results = dict()
    for by_file in weights_by_file:
        for polygon, region, weights in zip(by_file.polygons, by_file.regions, by_file.weights):
            result = results.get(region, None)
            if result is None:
                result = results[region] = ByRegion(region, grid_shape)
            result.add(polygon, weights)

    return list(results.values())


# OUTPUT DATA

def plain_name(shape_file):
    return os.path.splitext(os.path.basename(shape_file))[0]


def write_weights(out_name, weights_by_region, levels, res):
    data = {
        by_region.region: by_region.weights.tolist()
        for by_region in weights_by_region
    }
    data['levels'] = levels
    data['res'] = res

    data = yaml.dump(data)
    with open(out_name, 'w') as out:
        out.write(data)


# PLOTTING

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


def plot_data(m, plot_file, weights_by_region, x, y, xhgt, levels, res):
    """
    :type m: Basemap
    :type plot_file: string
    :type weights_by_region: dict of string, ByRegion
    """
    print('Plot map')
    pre_plot(m, x, y, xhgt)
    markers = product(('o', '*', 'x'), ('r', 'g', 'b', 'c', 'm', 'y', 'k'))
    sizes = {'o': 2, '*': 3, 'x': 3, }

    for by_region, (style, color) in zip(weights_by_region, markers):
        marker = color + style
        area = numpy.sum(by_region.weights) / levels * res[0] * res[1] / 1000000
        print('Plot %s[%s]\t% 4.2f km²' % (by_region.region, marker, area))
        for i, polygon in enumerate(by_region.polygons):
            x_p, y_p = m(polygon.exterior.xy[0], polygon.exterior.xy[1])
            label = style + ' ' + by_region.region if i == 0 else None
            m.plot(x_p, y_p, color + '-', lw=1, label=label, alpha=0.5)
        for level in range(levels):
            mask1 = by_region.weights <= level + 1
            mask2 = by_region.weights > level
            mask = mask1 & mask2
            size = sizes[style] * level / levels
            m.plot(x[mask], y[mask], marker, ms=size, mec=color)

    pylab.legend(ncol=2, fontsize='x-small')
    post_plot(plot_file)


# MAIN FUNCTION

def main():
    config, plot_file = read_config()

    # Define the coordinate transformation needed for shapefiles
    tx = coord_transform(config['shape_spatial_reference'],
                         config['plot_spatial_reference'])

    print('Read', config['geofile'])
    with netCDF4.Dataset(config['geofile']) as dataset:
        xhgt = dataset.variables['HGT_M'][0]
        xlon = dataset.variables['XLONG_M'][0]
        xlat = dataset.variables['XLAT_M'][0]
        res = [dataset.DX.item(), dataset.DY.item()]
        grid_shape = xlon.shape

    subres = config['sub_sampling']
    levels = subres ** 2
    sub_points = sub_sample_grid(xlon, xlat, subres)

    print('Create map projection')
    m = Basemap(projection='stere', lat_ts=65., lat_0=65., lon_0=-19.,
                llcrnrlon=-24.2, llcrnrlat=63.25,
                urcrnrlon=-13.1, urcrnrlat=66.5,
                resolution='i' if plot_file else None)
    x, y = m(xlon, xlat)

    weights_by_file = weigh_shapefiles(config['shape_files'], tx, grid_shape, sub_points, levels, res)
    weights_by_region = shuffle_weights(weights_by_file, grid_shape)

    if plot_file:
        plot_data(m, plot_file, weights_by_region, x, y, xhgt, levels, res)

    write_weights(config['output'], weights_by_region, levels, res)


if __name__ == '__main__':
    main()
