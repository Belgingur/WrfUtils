from __future__ import absolute_import, division, print_function, unicode_literals

from nose.tools import assert_almost_equal, assert_equal, assert_list_equal

from bilinear_interpolation import distance_to_side, generate_weights_bilinear


def test_distance_to_side():
    s, p1, p2 = {'lon': 2, 'lat': 2}, {'lon': 1, 'lat': 2}, {'lon': 3, 'lat': 2}
    dist = distance_to_side(p1, p2, s)
    assert_equal(0, dist)

    s, p1, p2 = {'lon': 0, 'lat': 1}, {'lon': 7, 'lat': 0}, {'lon': 1, 'lat': 8}
    dist = distance_to_side(p1, p2, s)
    assert_equal(5, dist)
    s, p1, p2 = {'lat': 0, 'lon': 1}, {'lat': 7, 'lon': 0}, {'lat': 1, 'lon': 8}
    dist = distance_to_side(p1, p2, s)
    assert_equal(5, dist)

    s, p1, p2 = {'lon': 3, 'lat': 3}, {'lon': 3, 'lat': 0}, {'lon': 0, 'lat': 3}
    dist = distance_to_side(p1, p2, s)
    assert_almost_equal(1.5 * 2 ** 0.5, dist)
    s, p1, p2 = {'lat': 3, 'lon': 3}, {'lat': 3, 'lon': 0}, {'lat': 0, 'lon': 3}
    dist = distance_to_side(p1, p2, s)
    assert_almost_equal(1.5 * 2 ** 0.5, dist)

    s, p1, p2 = {'lon': 0, 'lat': 2}, {'lon': 0, 'lat': 1}, {'lon': 0, 'lat': 13}
    dist = distance_to_side(p1, p2, s)
    assert_equal(0, dist)

    s, p1, p2 = {'lon': 13, 'lat': 5}, {'lon': 53, 'lat': 23}, {'lon': 12, 'lat': 33}
    dist1 = distance_to_side(p1, p2, s)
    s, p1, p2 = {'lat': 13, 'lon': 5}, {'lat': 53, 'lon': 23}, {'lat': 12, 'lon': 33}
    dist2 = distance_to_side(p1, p2, s)
    assert_almost_equal(dist1, dist2)

    s, p1, p2 = {'lon': 314, 'lat': 315}, {'lon': 84, 'lat': 24}, {'lon': 122, 'lat': 43}
    dist1 = distance_to_side(p1, p2, s)
    s, p1, p2 = {'lat': 314, 'lon': 315}, {'lat': 84, 'lon': 24}, {'lat': 122, 'lon': 43}
    dist2 = distance_to_side(p1, p2, s)
    assert_almost_equal(dist1, dist2)

    s, p1, p2 = {'lon': 0, 'lat': 0}, {'lon': 0, 'lat': 0}, {'lon': 12, 'lat': 33}
    dist1 = distance_to_side(p1, p2, s)
    s, p1, p2 = {'lat': 0, 'lon': 0}, {'lat': 0, 'lon': 0}, {'lat': 12, 'lon': 33}
    dist2 = distance_to_side(p1, p2, s)
    assert_almost_equal(dist1, dist2)


def test_generate_weights_bilinear():
    station = {'lon': 2, 'lat': 2}
    corners = {
        (0, 0): {'lon': 0, 'lat': 2, 'point_id': 10},
        (1, 0): {'lon': 2, 'lat': 0, 'point_id': 11},
        (0, 1): {'lon': 2, 'lat': 4, 'point_id': 12},
        (1, 1): {'lon': 4, 'lat': 2, 'point_id': 13}
    }
    weights = generate_weights_bilinear(station, corners)
    expected = [
        [10, 0.25],
        [11, 0.25],
        [12, 0.25],
        [13, 0.25]
    ]
    assert_list_equal(expected, sorted(weights, key=lambda x: x[0]))

    station = {'lon': 4/3.0, 'lat': 2}
    weights = generate_weights_bilinear(station, corners)
    expected = [
        [10, 4/9.0],
        [11, 2/9.0],
        [12, 2/9.0],
        [13, 1/9.0]
    ]
    for (e, a) in zip(expected, sorted(weights, key=lambda x: x[0])):
        assert_almost_equal(e[1], a[1])

    station = {'lon': 1, 'lat': 1.5}
    weights = generate_weights_bilinear(station, corners)
    expected = [
        [10, 17.5/32],
        [11, 10.5/32],
        [12, 2.5/32],
        [13, 1.5/32]
    ]
    for (e, a) in zip(expected, sorted(weights, key=lambda x: x[0])):
        assert_almost_equal(e[1], a[1])
