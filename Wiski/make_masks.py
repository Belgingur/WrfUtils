#!/usr/bin/env python3
# encoding: utf-8

"""
Generates point-weight masks from a WRF-style geography file and shape files.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle, count, repeat
from math import isnan, sqrt
from typing import List, Dict, Tuple, Optional, Sequence, Any, Callable

import netCDF4 as nc
import numpy as np
import ogr
import osr
import pylab
import yaml
from mpl_toolkits.basemap import Basemap
from osgeo.osr import CoordinateTransformation
from shapely.geometry import Point, Polygon

# SETUP

np.set_printoptions(precision=3, edgeitems=20, linewidth=125)

NaN = float('NaN')

CFG = Callable[[str], Optional[Any]]


class ConfigGetter:

    def __init__(self, parser: argparse.ArgumentParser):
        self._parser = parser
        self._args = self._parser.parse_args()
        with open(self._args.config) as f:
            self._config = yaml.load(f)
        self._sim_config = self._config.get('simulations', {}).get(self._args.simulation, {})

    def get(self, key: str, default=...):
        #print('get', key, '→', end=' ')
        value = getattr(self._args, key, None)
        if value is not None:
            #print(value, '[command-line]')
            return value

        value = self._sim_config.get(key)
        if value is not None:
            #print(value, '[simulation]')
            return value

        value = self._config.get(key)
        setattr(self._args, key, value)

        if value is None:
            if default is ...:
                self._parser.error(f'No configuration found for "{key}" in command-line arguments, simulation config or root config')
            #print(default, '[default]')
            return default

        #print(value, '[root]')
        return value

    def __getattribute__(self, name: str) -> Any:
        if name in ('get', 'error') or name.startswith('_'):
            return super().__getattribute__(name)
        else:
            return self.get(name)

    def error(self, message: str):
        self._parser.error(message)


def read_config() -> ConfigGetter:
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        epilog=None
    )
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Write more progress data')
    parser.add_argument('-c', '--config', default='wiski.yml',
                        help='Read configuration from this file (def: wiski.yml)')
    parser.add_argument('-s', '--simulation',
                        help='Configured simulation to work with.')
    parser.add_argument('geo', nargs='?',
                        help='WRF geo file to calculate from. Overrides setting in config.')

    return ConfigGetter(parser)


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
    j_lim, i_lim = xlon.shape
    sub_points = []
    for j in range(0, j_lim):
        for i in range(0, i_lim):
            # Find lat0,lon0 centre of grid cell j,i
            # and d_lon, d_lat the sub-sampling steps
            i1, i2 = max(j - 1, 0), min(j + 1, j_lim - 1)  # b is before, a is after
            j1, j2 = max(i - 1, 0), min(i + 1, i_lim - 1)

            lon0, lat0 = xlon[j, i], xlat[j, i]
            di_lon_1 = (xlon[j, i] - xlon[j, j1]) / subres / (i - j1) if i > j1 else NaN
            di_lon_2 = (xlon[j, j2] - xlon[j, i]) / subres / (j2 - i) if j2 > i else NaN
            dj_lat_1 = (xlat[j, i] - xlat[i1, i]) / subres / (j - i1) if j > i1 else NaN
            dj_lat_2 = (xlat[i2, i] - xlat[j, i]) / subres / (i2 - j) if i2 > j else NaN
            # print('% 3.0f % 3.0f\t% 2.4f % 2.4f\t% 2.4f % 2.4f\t% 2.4f % 2.4f' %
            # (j, i, lat0, lon0, dj_lat_1, di_lon_1, dj_lat_2, di_lon_2))

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
    print('\nCreate %s×%s supersampled cells in each of %s×%s cells' %
          (subres, subres, height.shape[0], height.shape[1]))

    j_lim, i_lim = height.shape
    sub_heights = []
    for j in range(0, j_lim):
        for i in range(0, i_lim):
            # Find lat0,lon0 centre of grid cell j,i
            # and d_lon, d_lat the sub-sampling steps
            i1, i2 = max(j - 1, 0), min(j + 1, j_lim - 1)  # b is before, a is after
            j1, j2 = max(i - 1, 0), min(i + 1, i_lim - 1)

            hgt0 = height[j, i]
            di_hgt_1 = (height[j, i] - height[j, j1]) / subres / (i - j1) if i > j1 else NaN
            di_hgt_2 = (height[j, j2] - height[j, i]) / subres / (j2 - i) if j2 > i else NaN
            dj_hgt_1 = (height[j, i] - height[i1, i]) / subres / (j - i1) if j > i1 else NaN
            dj_hgt_2 = (height[i2, i] - height[j, i]) / subres / (i2 - j) if i2 > j else NaN
            # print('% 3.0f % 3.0f\t% 4.4f\t% 2.4f % 2.4f\t% 2.4f % 2.4f' %
            # (j, i, hgt0, di_hgt_1, di_hgt_2, dj_hgt_1, dj_hgt_2))

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
        print('%7d points in %4dm .. %4d:% 10.2f km²' %
              (range_point_count, range_start, range_end, range_point_count * sub_cell_area))
        range_start, range_end = range_end, range_end + height_res
    return ranges


def print_shape(indent, shape, name='shape'):
    s = str(shape)
    if len(s) > 60:
        s = s[:40] + ' … ' + s[-20:]
    print(indent, name + ':', type(shape), s)


def handle_shape(indent, polygons: List[Polygon], shape: ogr.Geometry, verbose: bool):
    if verbose:
        print_shape(indent, shape)
    try:
        if isinstance(shape, ogr.Geometry):

            # Try to get a polygon from the geometry's points
            gtype = shape.GetGeometryType()
            if gtype == 2:
                try:
                    points: List[Point] = shape.GetPoints()
                    if points is not None and len(points) > 0:
                        polygon = Polygon(points)
                        if verbose:
                            print(indent, '→ polygon with ', len(points), 'points')
                        polygons.append(polygon)
                except Exception as e:
                    print(indent, 'exception making polygon', e)

            elif gtype in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                ...
            else:
                print('Unexpected gtype', gtype)

            sub_count = shape.GetGeometryCount()
            # print(indent, "sub_count:", sub_count)
            for sub_idx in range(sub_count):
                sub_shape: ogr.Geometry = shape.GetGeometryRef(sub_idx)
                handle_shape(indent + '    ', polygons, sub_shape, verbose)
                continue

        else:
            print(indent, 'It is an unexpected', type(shape))

    except Exception as e:
        print(indent, 'failed!', e)
        raise


def read_shapefile(shapefile: str, tx: CoordinateTransformation, verbose: bool) -> List[Polygon]:
    """
    Reads a shapefile and returns a list of polygons, each of which is a list of points.
    """
    # Now open the shapefile and start by reading the only layer and
    # the first feature, as well as its shape.
    source: ogr.DataSource = ogr.Open(shapefile)
    layer: ogr.Layer = source.GetLayer()
    counts: int = layer.GetFeatureCount()
    polygons = []

    for c in range(counts):
        try:
            feature: ogr.Feature = layer.GetFeature(c)
            # if get_field_value(feature, 'catagory') == 'intrnl_rock':
            #    continue

            # Convert feature into a polygon in the target projection
            shape: ogr.Geometry = feature.GetGeometryRef()
            shape.Transform(tx)
            handle_shape('    ', polygons, shape, verbose)

        except TypeError as te:
            if 'Geometry_Transform' in str(te):
                print('ERROR:', str(te))
                print('You may need to set environment variable GDAL_DATA="/usr/share/gdal/1.10/" or similar')
                sys.exit(1)
    if verbose:
        print()
    return polygons


def get_field_value(feature: ogr.Feature, fld_name: str, default=None):
    try:
        return feature.GetField(fld_name)
    except ValueError:
        if default is ...:
            raise
        return default


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

@dataclass
class LabelledWeights(object):
    region: str
    """ a key for the region """

    min_height: Optional[float]
    """ the bottom of the height range we're looking at (inclusive) """

    max_height: Optional[float]
    """ the top of the height range we're looking at (exclusive) """

    levels: int
    """ The number of sub-sampling points in each original grid cell """

    sub_cell_area: float
    """ The (approximate) area of the cell around each sub-sampling point. """

    polygons: List[Polygon]
    """ List set of 1 or more unique polygons that the data comes from """

    weights: np.ndarray
    """
    an ndarray indicating how many of the sub-sampling points from
    the corresponding index in the total grid fall within the height range and
    any of the polygons.
    """

    atomic: bool = True
    """ Whether this is an original region/height combination as opposed to a summation. """

    def copy(self, content=True):
        if content:
            return LabelledWeights(self.region, self.min_height, self.max_height,
                                   self.levels, self.sub_cell_area,
                                   self.polygons, self.weights, self.atomic)
        else:
            return LabelledWeights(self.region, None, None,
                                   self.levels, self.sub_cell_area,
                                   [], np.zeros_like(self.weights), True)

    def area(self):
        return float(np.sum(self.weights)) * self.sub_cell_area

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

        # Union of two lists of the unhashable Polygon instances
        polygons = list(self.polygons)
        for p in other.polygons:
            if p not in self.polygons:
                polygons.append(p)

        return LabelledWeights(
            self.region,
            min(self.min_height, other.min_height) if self.min_height is not None else other.min_height,
            max(self.max_height, other.max_height) if self.max_height is not None else other.max_height,
            self.levels,
            self.sub_cell_area,
            polygons,
            self.weights + other.weights,
            False
        )

    def cropped_weights(self):
        """
        Return the weights array with any margin of zero values removed, along with the offset into the original area
        at which the cropped array starts.

        :return: (j0, i0), cropped_weights
        :rtype: ((int, int), ndarray)
        """
        W = np.argwhere(self.weights)
        i0, j0 = W.min(0)
        i1, j1 = W.max(0) + 1
        cropped_weights = self.weights[i0:i1, j0:j1]

        # print('  crop weights %s -> %s at %s' %
        #      (self.weights.shape, cropped_weights.shape, (j0, i0)))

        return (i0, j0), cropped_weights


def points_in_polygon(poly: Polygon, sub_points_per_cell: Sequence[Sequence[Point]], shape: Tuple[int, int]):
    """
    Counts the number of sub-sampling points in each cell which fall within the
    polygon.
    """
    # Some would first transform data to e.g. km using e.g. lambert-projection

    # Step through elements in grid to evaluate if in the study area
    weights = np.zeros(shape[0] * shape[1], dtype=int)
    for n, sub_points in enumerate(sub_points_per_cell):
        for point in sub_points:
            if point.within(poly):
                weights[n] += 1
    weights = weights.reshape(shape)
    return weights


def weigh_shapefiles(
        shape_files: Dict[str, Sequence[str]],
        tx: CoordinateTransformation,
        grid_shape: Tuple[int, int],
        sub_points_by_height: List[Optional[List[List[Point]]]],
        levels: int,
        sub_cell_area: float,
        height_res: float,
        verbose: bool,
) -> List[LabelledWeights]:
    """
    Reads shape_files and for each (polygon,height) pair, calculate a weight grid
    for how many sub-points at that height fall within the polygon.

    Returns a list of LabelledWeights(region, height, polygons, weights) instances,
    some of which may refer to the same region,height pairs.
    """
    results = []
    for shape_file, regions in shape_files.items():
        if isinstance(regions, str):
            regions = repeat(regions)

        print('\nRead', shape_file)

        polygons = read_shapefile(shape_file, tx, verbose)
        print(f'Found {len(polygons)} polygons')
        for i, poly, region in zip(count(), polygons, regions):
            print('% 2s % 15s' % (i, region), end='')
            for height_index, sub_points in enumerate(sub_points_by_height):
                weights = points_in_polygon(poly, sub_points, grid_shape)
                area = np.sum(weights) * sub_cell_area
                height = height_index * height_res
                if area:
                    print('\t', height, end='')
                    r = LabelledWeights(region, height, height + height_res, levels, sub_cell_area, [poly], weights)
                    results.append(r)
            print()

    return results


def collate_weights(raw_weights: List[LabelledWeights]) -> Dict[str, List[LabelledWeights]]:
    """
    Group together weights files belonging to the same region and merge the
    weights that belong to the same region *and* height range.
    """
    print('\nCollate weights by region and height')
    scratch: Dict[Tuple[str, float, float], LabelledWeights] = defaultdict()

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
        if len(lw_list) > 1:
            print('    TOTAL % 8.2fkm²' % total.area())
        lw_list.append(total)

    return collated


# OUTPUT DATA

def plain_name(shape_file):
    return os.path.splitext(os.path.basename(shape_file))[0]


def write_weights(simulation: str, collated_weights: Dict[str, List[LabelledWeights]], weight_file_pattern: str, region_height_key_pattern: str, region_total_key_pattern: str):
    """
    Write weights to a compressed numpy npz file
    """
    print()
    print('Write weights')
    output_map = {}
    for region, lw_list in collated_weights.items():
        for lw in lw_list:
            pattern = region_height_key_pattern if lw.atomic else region_total_key_pattern
            offset, weights = lw.cropped_weights()
            lwd = dataclasses.asdict(lw)
            key = pattern.format(simulation=simulation, offset=offset, **lwd)
            print('  add', key, '×'.join(map(str, weights.shape)), 'weights')
            output_map[key] = weights
    weight_file = weight_file_pattern.format(simulation=simulation)
    print('Write', weight_file)
    np.savez_compressed(weight_file, **output_map)


# PLOTTING

def setup_basemap(xlat, xlon, levels_and_weights):
    lat_min = +90
    lat_max = -90
    lon_min = +180
    lon_max = -180
    for law in levels_and_weights:
        j_nz = np.nonzero(np.sum(law.weights, axis=0))
        j_min, j_max = np.min(j_nz) - 1, np.max(j_nz) + 1
        j_mid = (j_min + j_max) // 2
        i_nz = np.nonzero(np.sum(law.weights, axis=1))
        i_min, i_max = np.min(i_nz) - 1, np.max(i_nz) + 1
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
    m.drawparallels(np.arange(63., 67., .1), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-26., -12., .2), labels=[0, 0, 0, 1])
    m.drawmapscale(-15., 63.5, -19., 65., 100., barstyle='fancy', fontsize=12)
    iso = np.arange(height_res, 2300, height_res)
    cs = m.contour(x, y, xhgt, iso, colors='#808080', linewidths=0.5)
    pylab.clabel(cs, fmt='%1.0f', colors='#808080', inline=1, fontsize=10)
    return m


def post_plot(plot_file_pattern: str, plot_title_pattern: str, simulation: str, law: LabelledWeights):
    law_dict = dataclasses.asdict(law)
    plot_title = plot_title_pattern.format(simulation=simulation, **law_dict)
    plot_file = plot_file_pattern.format(simulation=simulation, **law_dict)
    print(' ', plot_file)
    pylab.title(plot_title)
    pylab.savefig(plot_file)
    pylab.clf()
    pylab.close()


def plot_data(
        simulation: str,
        collated_weights: Dict[str, List[LabelledWeights]],
        xlat: np.ndarray,
        xlon: np.ndarray,
        xhgt: np.ndarray,
        height_res: float,
        plot_file_pattern: str,
        plot_title_pattern: str
):
    print('\nPlot maps for', simulation)
    # symbols = ('o', '*', '+', 'x')
    symbols = ('o', '*')
    colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k')
    sizes = {'o': 3, '*': 4, 'x': 3, '+': 3}

    for region, levels_and_weights in collated_weights.items():
        markers = zip(cycle(colors), cycle(symbols))
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

        post_plot(plot_file_pattern, plot_title_pattern, simulation, law)


# MAIN FUNCTION

def main():
    cfg = read_config()

    # Define the coordinate transformation needed for shapefiles
    tx = coord_transform(cfg.shape_spatial_reference, cfg.plot_spatial_reference)

    geo_path = cfg.geo
    geo_path = os.path.expandvars(geo_path)
    geo_path = os.path.expanduser(geo_path)
    print('Read', geo_path)
    with nc.Dataset(geo_path) as dataset:
        xhgt = dataset.variables['HGT_M'][0]
        xlon = dataset.variables['XLONG_M'][0]
        xlat = dataset.variables['XLAT_M'][0]
        height = dataset.variables['HGT_M']
        height = height[0]
        resolution = [dataset.DX.item(), dataset.DY.item()]
        grid_shape = xlon.shape

    height_res = int(cfg.get('height_resolution', 10000))  # Default results in a single band
    sub_res = cfg.sub_sampling
    sub_levels = sub_res ** 2
    sub_cell_area = resolution[0] * resolution[1] / sub_levels / 1000000
    sub_points = sub_sample_grid(xlon, xlat, sub_res)
    sub_heights = interpolate_height(height, sub_res)
    sub_points_by_height = split_points_by_height(sub_points, sub_heights, sub_cell_area, height_res)

    labelled_weights = weigh_shapefiles(cfg.shape_files, tx, grid_shape, sub_points_by_height, sub_levels, sub_cell_area, height_res, cfg.verbose)
    collated_weights = collate_weights(labelled_weights)

    write_weights(cfg.simulation,
                  collated_weights,
                  cfg.weight_file_pattern,
                  cfg.region_height_key_pattern,
                  cfg.region_total_key_pattern)

    if cfg.plot_file_pattern:
        plot_data(cfg.simulation, collated_weights,
                  xlat, xlon, xhgt, height_res,
                  cfg.plot_file_pattern,
                  cfg.plot_title_pattern)


if __name__ == '__main__':
    main()
