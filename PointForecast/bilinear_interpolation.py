#!/usr/bin/env python
# encoding: utf-8

"""

Functions to perform bilinear interpolation based only on the station lon/lat and wrfout file.

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import logging

import netCDF4
import numpy as np


LOG = logging.getLogger('belgingur.bilinear')

EARTH_RADIUS_M = 6378168


class TargetOutsideGridError(ValueError):
    """ Raised when the target point is outside the borders of the forecast grid. """

def distance_to_side(point_1, point_2, station):

    """ Analytical geometric distance between a point and a line crossing two other points. """

    s_x, s_y = station['lon'], station['lat']
    p1_x, p1_y = point_1['lon'], point_1['lat']
    p2_x, p2_y = point_2['lon'], point_2['lat']

    if p2_x == p1_x:  # because if the line is vertical, the line equation would need need to divide by zero.
        p1_x, p1_y = p1_y, p1_x
        p2_x, p2_y = p2_y, p2_x
        s_x, s_y = s_y, s_x

    top = ((p2_y - p1_y) / (p2_x - p1_x)) * s_x - s_y + ((p2_x * p1_y - p1_x * p2_y) / (p2_x - p1_x))
    bottom = (1 + ((p2_y - p1_y)/(p2_x - p1_x)) ** 2) ** 0.5
    dist = abs(top/bottom)
    return dist


def generate_weights_bilinear(station, corners):

    """ Calculate weights for bilinear interpolation based on distances from each 'wall' of a grid cell. """

    distances = {
        "_0": distance_to_side(corners[(0, 0)], corners[(1, 0)], station),
        "_1": distance_to_side(corners[(0, 1)], corners[(1, 1)], station),
        "0_": distance_to_side(corners[(0, 0)], corners[(0, 1)], station),
        "1_": distance_to_side(corners[(1, 0)], corners[(1, 1)], station)
    }

    denominator = (distances['_0'] + distances['_1']) * (distances['0_'] + distances['1_'])

    weights = [  # [point_id, weight]
        [corners[(0, 0)]['point_id'], distances["1_"] * distances["_1"] / denominator],
        [corners[(0, 1)]['point_id'], distances["1_"] * distances["_0"] / denominator],
        [corners[(1, 0)]['point_id'], distances["0_"] * distances["_1"] / denominator],
        [corners[(1, 1)]['point_id'], distances["0_"] * distances["_0"] / denominator]
    ]
    return weights


def globe_distance_deg(lat1, lon1, lat2, lon2):

    """ Distance between two points [deg lat/lon]. The distance has the same units as the radius, m by default. """

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    d_lat = (lat2 - lat1) / 2
    d_lon = (lon2 - lon1) / 2
    a = math.sin(d_lat) * math.sin(d_lat) + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon) * math.sin(d_lon)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = EARTH_RADIUS_M * c

    return d


def closest_point(s_lon, s_lat, longs, lats):

    """
    First find nearest neighbour candidate by approximating the cell size
    and collecting all points distant in lon/lat less than cell size in lon and lat, respectively.
    """

    long_span = (longs.max() - longs.min()) / longs.shape[1]
    lat_span = (lats.max() - lats.min()) / lats.shape[0]

    candidates = []
    for (j, i) in np.ndindex(longs.shape):
        if abs(longs[j, i] - s_lon) < long_span and abs(lats[j, i] - s_lat) < lat_span:
            candidates.append((i, j))

    if not candidates:
        # print('No estimated candidates, indexing the whole grid')
        for (j, i) in np.ndindex(longs.shape):
            candidates.append((i, j))

    cand_dict = {(i, j): globe_distance_deg(s_lat, s_lon, lats[j, i], longs[j, i]) for (i, j) in candidates}
    (i, j) = min(cand_dict, key=cand_dict.get)

    return {'point_id': (i, j), 'lon': longs[j, i], 'lat': lats[j, i], 'i': i, 'j': j}


def point_info(i, j, lons, lats):
    if i < 0 or j < 0:
        raise IndexError('Negative value in point indexes')
    return {'point_id': (i, j), 'lon': lons[j, i], 'lat': lats[j, i]}


def extract_coordinates(wrfout, margin):
    with netCDF4.Dataset(wrfout) as dataset:
        if margin:
            y, x = dataset.variables['XLAT'][0].shape
            if y - 2 * margin <= 0 or x - 2 * margin <= 0:
                raise ValueError('Requested margin is larger than the domain dimensions')
            lats = dataset.variables['XLAT'][0, margin:-margin, margin:-margin]
            lons = dataset.variables['XLONG'][0, margin:-margin, margin:-margin]
        else:
            lats, lons = dataset.variables['XLAT'][0], dataset.variables['XLONG'][0]
    return lats, lons


def do_weights(station, wrfout, margin=0, nearest_neighbour=False):

    """
    Given a station and wrfout pair, seeks the 'corner grid points' for the station location
    and calculates bilinear interpolation weights from the distances to each corner.
    """

    lats, lons = extract_coordinates(wrfout, margin)

    s_lon = station['lon']
    s_lat = station['lat']

    nearest = closest_point(s_lon, s_lat, lons, lats)

    x2_off = 1 if s_lon > nearest['lon'] else -1
    y2_off = 1 if s_lat > nearest['lat'] else -1

    try:
        corners = {
            (0, 0): nearest,
            (0, 1): point_info(nearest['i'], nearest['j'] + y2_off, lons, lats),
            (1, 0): point_info(nearest['i'] + x2_off, nearest['j'], lons, lats),
            (1, 1): point_info(nearest['i'] + x2_off, nearest['j'] + y2_off, lons, lats)
        }
    except IndexError:
        raise TargetOutsideGridError('The selected point is outside the borders of the grid.')

    if nearest_neighbour:
        return {nearest['point_id']: 1}

    weights = generate_weights_bilinear(station, corners)
    weights = {k: v for [k, v] in weights}

    return weights

