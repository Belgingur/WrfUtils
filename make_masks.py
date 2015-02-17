#!/usr/bin/env python
# encoding: utf-8

"""
Generates point-weight masks from a WRF-style geography file and shape files.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Python library imports
import argparse
from collections import defaultdict
from itertools import cycle, izip
import netCDF4
import os
from math import isnan, sqrt

import numpy
import pylab
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon
import ogr
import osr
import sys
import yaml


# SETUP

numpy.set_printoptions(precision=3, linewidth=125)

NaN = float('NaN')


def read_config():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        epilog=None
    )
    parser.add_argument('--config', default='make_masks.yml',
                        help='Read configuration from this file (def: make_masks.yml)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
        return config


def linear_interpolate(x0, n, d1, d2):
    # print('linear_interpolate(', x0, n, d1, d2, ')')
    d1 = d2 if isnan(d1) else d1
    d2 = d1 if isnan(d2) else d2
    lim = (n - 1) / 2
    sub = -lim
    x = x0 + sub * d1
    # print(lim, sub, x)

    while sub < -0.4:
        # print('yield<0', sub, x)
        yield x
        sub += 1
        x += d1

    if -0.4 < sub < 0.4:
        # print('yield=0', 0, x)
        yield x0
        sub = 1
        x = x0 + d2

    while sub <= lim:
        # print('yield>0', sub, x)
        yield x
        sub += 1
        x += d2
    return


def sub_sample_grid(xlon, xlat, subres):
    """
    Create an array in the shape of the flattened xlon or xlat, containing at
    each position an array of subres*subres sub-sampling points for that cell.
    """
    # print('Create %s*%s sub-sampling points in each of %s*%s cells' %
    # (subres, subres, xlon.shape[0], xlon.shape[1]))
    i_lim, j_lim = xlon.shape
    sub_points = []
    for i in range(0, i_lim):
        for j in range(0, j_lim):
            # Find lat0,lon0 centre of grid cell i,j
            # and d_lon, d_lat the sub-sampling steps
            i1, i2 = max(i - 1, 0), min(i + 1, i_lim - 1)  # b is before, a is after
            j1, j2 = max(j - 1, 0), min(j + 1, j_lim - 1)

            lon0, lat0 = xlon[i, j], xlat[i, j]
            di_lon_1 = (xlon[i, j] - xlon[i, j1]) / subres / (j - j1) if j > j1 else NaN
            di_lon_2 = (xlon[i, j2] - xlon[i, j]) / subres / (j2 - j) if j2 > j else NaN
            dj_lat_1 = (xlat[i, j] - xlat[i1, j]) / subres / (i - i1) if i > i1 else NaN
            dj_lat_2 = (xlat[i2, j] - xlat[i, j]) / subres / (i2 - i) if i2 > i else NaN
            # print('% 3.0f % 3.0f\t% 2.4f % 2.4f\t% 2.4f % 2.4f\t% 2.4f % 2.4f' %
            # (i, j, lat0, lon0, dj_lat_1, di_lon_1, dj_lat_2, di_lon_2))

            # Make an array of subres*subres evenly spaced points inside cell
            points = []
            sub_points.append(points)
            for lon in linear_interpolate(lon0, subres, di_lon_1, di_lon_2):
                # print(end='\t')
                for lat in linear_interpolate(lat0, subres, dj_lat_1, dj_lat_2):
                    # print('% 2.4f % 2.4f' % (lat, lon), end='\t')
                    point = Point(lon, lat)
                    points.append(point)
                    # print()
    return sub_points


def interpolate_height(height, subres):
    print('\nCreate %s*%s interpolated heights in each of %s*%s cells' %
          (subres, subres, height.shape[0], height.shape[1]))

    i_lim, j_lim = height.shape
    sub_heights = []
    for i in range(0, i_lim):
        for j in range(0, j_lim):
            # Find lat0,lon0 centre of grid cell i,j
            # and d_lon, d_lat the sub-sampling steps
            i1, i2 = max(i - 1, 0), min(i + 1, i_lim - 1)  # b is before, a is after
            j1, j2 = max(j - 1, 0), min(j + 1, j_lim - 1)

            hgt0 = height[i, j]
            di_hgt_1 = (height[i, j] - height[i, j1]) / subres / (j - j1) if j > j1 else NaN
            di_hgt_2 = (height[i, j2] - height[i, j]) / subres / (j2 - j) if j2 > j else NaN
            dj_hgt_1 = (height[i, j] - height[i1, j]) / subres / (i - i1) if i > i1 else NaN
            dj_hgt_2 = (height[i2, j] - height[i, j]) / subres / (i2 - i) if i2 > i else NaN
            # print('% 3.0f % 3.0f\t% 4.4f\t% 2.4f % 2.4f\t% 2.4f % 2.4f' %
            # (i, j, hgt0, di_hgt_1, di_hgt_2, dj_hgt_1, dj_hgt_2))

            # Make an array of subres*subres evenly spaced points inside cell
            heights = []
            for hgt_i in linear_interpolate(hgt0, subres, di_hgt_1, di_hgt_2):
                # print(end='\t')
                for hgt_j in linear_interpolate(hgt0, subres, dj_hgt_1, dj_hgt_2):
                    hgt_ij = max((hgt_i + hgt_j) / 2, 0)
                    # print('% 4.4f' % hgt_ij, end='\t')
                    heights.append(hgt_ij)
                    # print()
            sub_heights.append(heights)
    return sub_heights


def split_points_by_height(sub_points, sub_heights, sub_cell_area, height_res):
    """
    Splits the sub_points array into an array of points for each 100m height
    range, so sub_points_by_height[2] would be the points whose ground-height
    is in [200m, 300m[

    Continues until all points have been put in a range and writes None for
    pointless ranges.

    :param sub_points: Array of points to split
    :param sub_heights: Array of same dimension as sub_points with the height of each point in each location.
    :return: array of numpy arrays of points, one for each height range
    """
    total_point_count = len(sub_points) * len(sub_points[0])
    sorted_point_count = 0
    ranges = []
    range_start = 0
    range_end = range_start + height_res
    while sorted_point_count < total_point_count:
        range_points = []
        range_point_count = 0
        for points, heights in zip(sub_points, sub_heights):
            rp = []
            for point, height in zip(points, heights):
                if range_start <= height < range_end:
                    rp.append(point)
            range_points.append(rp)
            range_point_count += len(rp)
        ranges.append(range_points if range_point_count else None)
        sorted_point_count += range_point_count
        print('%7d points in %4dm .. %4d:% 10.2fkm²' %
              (range_point_count, range_start, range_end, range_point_count * sub_cell_area))
        range_start, range_end = range_end, range_end + height_res
    return ranges


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
        try:
            feature = layer.GetFeature(c)
            geometry = feature.GetGeometryRef()

            # Do the coordinate transformation
            geometry.Transform(tx)

            # Read the polygon (there is just one) and the defining points.
            points = geometry.GetGeometryRef(0).GetPoints()
            polygons.append(Polygon(points))
        except TypeError as te:
            if 'Geometry_Transform' in str(te):
                print('ERROR:', str(te))
                print('You may need to set environment variable GDAL_DATA="/usr/share/gdal/1.10/" or similar')
                sys.exit(1)
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

class LabelledWeights(object):
    def __init__(self, region, min_height, max_height, levels, sub_cell_area, polygons, weights, atomic=True):
        self.region = region
        """ a key for the region """

        self.min_height = min_height
        """ the bottom of the height range we're looking at (inclusive) """

        self.max_height = max_height
        """ the top of the height range we're looking at (exclusive) """

        self.levels = levels
        """ The number of sub-sampling points in each original grid cell """

        self.sub_cell_area = sub_cell_area
        """ The (approximate) area of the cell around each sub-sampling point. """

        self.polygons = {polygons} if isinstance(polygons, Polygon) else set(polygons)
        """ set of 1 or more polygons that the data comes from """

        self.weights = weights
        """
        an ndarray indicating how many of the sub-sampling points from
        the corresponding index in the total grid fall within the height range and
        any of the polygons.
        """

        self.atomic = atomic
        """ Whether this is an original region/height combination as opposed to a summation. """

    def copy(self, content=True):
        if content:
            return LabelledWeights(self.region, self.min_height, self.max_height,
                                   self.levels, self.sub_cell_area,
                                   self.polygons, self.weights, self.atomic)
        else:
            return LabelledWeights(self.region, None, None,
                                   self.levels, self.sub_cell_area,
                                   set(), numpy.zeros_like(self.weights), True)

    def area(self):
        return float(numpy.sum(self.weights)) * self.sub_cell_area

    def __repr__(self):
        return 'LW[%s %s%s-%s #%s %0.2fkm^2]' % \
               (self.region, 'sum:' if not self.atomic else '', self.min_height, self.max_height,
                len(self.polygons), self.area())

    def key(self):
        return self.region, self.min_height, self.max_height

    def __add__(self, other):
        assert self.region == other.region
        assert self.levels == other.levels
        assert self.sub_cell_area == other.sub_cell_area
        return LabelledWeights(
            self.region,
            min(self.min_height, other.min_height) if self.min_height is not None else other.min_height,
            max(self.max_height, other.max_height) if self.max_height is not None else other.max_height,
            self.levels,
            self.sub_cell_area,
            self.polygons.union(other.polygons),
            self.weights + other.weights,
            False
        )


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


def weigh_shapefiles(shape_files, tx, grid_shape, sub_points_by_height, levels, sub_cell_area, height_res):
    """
    Reads shapefiles and for each polygon,height pair, calculate a weight grid
    for how many sub-points at that height fall within the polygon.

    Returns a list of LabelledWeights(region, height, polygons, weights) instances,
    some of which may refer to the same region,height pairs.
    """
    results = []
    for shape_file, regions in shape_files.items():

        print('\nRead', shape_file)

        polygons = read_shapefile(shape_file, tx)
        for poly, region in zip(polygons, regions):
            print('% 15s' % region, end='')
            for height_index, sub_points in enumerate(sub_points_by_height):
                weights = points_in_polygon(poly, sub_points, grid_shape)
                area = numpy.sum(weights) * sub_cell_area
                height = height_index * height_res
                if area:
                    print('\t', height, end='')
                    r = LabelledWeights(region, height, height + height_res, levels, sub_cell_area, poly, weights)
                    results.append(r)
            print()

    return results


def collate_weights(raw_weights):
    print('\nCollate weights by region and height')
    scratch = defaultdict()  # LabelledWeights objects by (region, height)

    # Combine LabelledWeights instances with identical key
    for raw in raw_weights:
        key = raw.key()
        cooked = scratch.get(key)
        if cooked:
            cooked += raw
        else:
            scratch[key] = raw.copy(True)

    # Group instances by region and sort by level
    collated = defaultdict(lambda: [])
    for lw in scratch.values():
        collated[lw.region].append(lw)
    for region, lwl in collated.items():
        collated[region] = sorted(lwl, key=lambda lw: (lw.min_height, lw.max_height))

    for region, lw_list in collated.items():
        print('\n ', region)
        total = lw_list[0].copy(False)
        for lw in lw_list:
            print('    % 5.0f % 8.2fkm²' % (lw.min_height, lw.area()))
            total += lw
        print('    TOTAL % 8.2fkm²' % total.area())
        lw_list.append(total)

    return collated


# OUTPUT DATA

def plain_name(shape_file):
    return os.path.splitext(os.path.basename(shape_file))[0]


def write_weights(collated_weights, weight_file_pattern, region_height_key_pattern, region_total_key_pattern):
    for region, lw_list in collated_weights.items():
        for lw in lw_list:
            pattern = region_height_key_pattern if lw.atomic else region_total_key_pattern
            lwd = lw.__dict__
            key = pattern.format(**lwd)
            file_name = weight_file_pattern.format(key=key, **lwd)
            print('write', file_name)
            lw.weights.tofile(file_name)


# PLOTTING

def setup_basemap(xlat, xlon, levels_and_weights):
    lat_min = +90
    lat_max = -90
    lon_min = +180
    lon_max = -180
    for law in levels_and_weights:
        j_nz = numpy.nonzero(numpy.sum(law.weights, axis=0))
        j_min, j_max = numpy.min(j_nz) - 1, numpy.max(j_nz) + 1
        j_mid = (j_min + j_max) // 2
        i_nz = numpy.nonzero(numpy.sum(law.weights, axis=1))
        i_min, i_max = numpy.min(i_nz) - 1, numpy.max(i_nz) + 1
        i_mid = (i_min + i_max) // 2
        # print('ranges', i_min, i_max, j_min, j_max)

        lat_min = min(lat_min, xlat[i_min, j_mid])
        lat_max = max(lat_max, xlat[i_max, j_mid])

        lon_min = min(lon_min, xlon[i_mid, j_min])
        lon_max = max(lon_max, xlon[i_mid, j_max])
    # print('lat: % 10.2f - % 10.2f' % (lat_min, lat_max))
    # print('lon: % 10.2f - % 10.2f' % (lon_min, lon_max))

    shrink = max(
        (lat_max - lat_min) / (66.5 - 63.25),
        (lon_max - lon_min) / (24.2 - 13.1)
    )

    m = Basemap(projection='stere', lat_ts=65., lat_0=65., lon_0=-19.,
                llcrnrlon=lon_min, llcrnrlat=lat_min,
                urcrnrlon=lon_max, urcrnrlat=lat_max,
                resolution='i')
    x, y = m(xlon, xlat)
    return m, x, y, shrink


def pre_plot(m, x, y, xhgt, height_res):
    pylab.figure(figsize=(11.02, 8.27), dpi=72, frameon=True)
    m.drawcoastlines(linewidth=1.5, color='black')
    m.readshapefile('joklar/joklar', 'joklar', linewidth=1.5, color='black')
    m.drawparallels(numpy.arange(63., 67., .1), labels=[1, 0, 0, 0])
    m.drawmeridians(numpy.arange(-26., -12., .2), labels=[0, 0, 0, 1])
    m.drawmapscale(-15., 63.5, -19., 65., 100., barstyle='fancy', fontsize=12)
    iso = numpy.arange(height_res, 2300, height_res)
    cs = m.contour(x, y, xhgt, iso, colors='#808080', linewidths=0.5)
    pylab.clabel(cs, fmt='%1.0f', colors='#808080', inline=1, fontsize=10)
    return m


def post_plot(plot_file_pattern, region_total_key_pattern, law):
    key = region_total_key_pattern.format(**law.__dict__)
    plot_file = plot_file_pattern.format(key=key)
    print(' ', plot_file)
    pylab.title(law.region)
    pylab.savefig(plot_file)
    pylab.clf()
    pylab.close()


def plot_data(collated_weights, xlat, xlon, xhgt, height_res, plot_file_pattern, region_total_key_pattern):
    """
    :type plot_file_pattern: string
    :type collated_weights: dict of [str, LabelledWeights]
    """

    print('\nPlot maps')
    # symbols = ('o', '*', '+', 'x')
    symbols = ('o', '*')
    colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
    sizes = {'o': 3, '*': 4, 'x': 3, '+': 3}

    for region, levels_and_weights in collated_weights.items():
        markers = izip(cycle(colors), cycle(symbols))
        m, x, y, shrink = setup_basemap(xlat, xlon, levels_and_weights)
        pre_plot(m, x, y, xhgt, height_res)

        for law, (color, style) in zip(levels_and_weights, markers):

            if not law.atomic:
                # Draw the polygons that make up the total region
                for polygon in law.polygons:
                    x_p, y_p = m(polygon.exterior.xy[0], polygon.exterior.xy[1])
                    m.plot(x_p, y_p, 'b-', lw=2, alpha=0.5)

            else:
                # Plot all weight grids that contribute to the region
                marker = color + style
                for level in range(law.levels):
                    mask1 = law.weights <= level + 1
                    mask2 = law.weights > level
                    mask = mask1 & mask2
                    size = sizes[style] * sqrt((level + 1) / law.levels) / shrink
                    m.plot(x[mask], y[mask], marker, ms=size, mec=color, alpha=3 / 4)

        post_plot(plot_file_pattern, region_total_key_pattern, law)


# MAIN FUNCTION

def main():
    config = read_config()

    # Define the coordinate transformation needed for shapefiles
    tx = coord_transform(config['shape_spatial_reference'],
                         config['plot_spatial_reference'])

    print('Read', config['geofile'])
    with netCDF4.Dataset(config['geofile']) as dataset:
        xhgt = dataset.variables['HGT_M'][0]
        xlon = dataset.variables['XLONG_M'][0]
        xlat = dataset.variables['XLAT_M'][0]
        height = dataset.variables['HGT_M']
        height = height[0]
        resolution = [dataset.DX.item(), dataset.DY.item()]
        grid_shape = xlon.shape

    subres = config['sub_sampling']
    height_res = int(config['height_resolution'])
    levels = subres ** 2
    sub_cell_area = resolution[0] * resolution[1] / levels / 1000000
    sub_points = sub_sample_grid(xlon, xlat, subres)
    sub_heights = interpolate_height(height, subres)
    sub_points_by_height = split_points_by_height(sub_points, sub_heights, sub_cell_area, height_res)

    labelled_weights = weigh_shapefiles(config['shape_files'], tx, grid_shape, sub_points_by_height,
                                        levels, sub_cell_area, height_res)
    collated_weights = collate_weights(labelled_weights)

    write_weights(collated_weights,
                  config['weight_file_pattern'],
                  config['region_height_key_pattern'],
                  config['region_total_key_pattern'])

    if config['plot_file_pattern']:
        plot_data(collated_weights, xlat, xlon, xhgt, height_res,
                  config['plot_file_pattern'],
                  config['region_total_key_pattern'])


if __name__ == '__main__':
    main()
