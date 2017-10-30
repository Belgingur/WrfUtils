# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from nose.tools import assert_almost_equal

from point_forecast import calculate_pf, weighted_avg, circular_weighted_avg


def test_circular_weighted_avg():
    forecasts = [360, 1, 3, 4]
    weights = [0.5, 0.2, 0.2, 0.1]
    expected = 1.2
    actual = circular_weighted_avg(forecasts, weights)
    assert_almost_equal(expected, actual, 3)

    forecasts = [10, 14, 10, 15]
    weights = [0.3, 0.3, 0.3, 0.1]
    expected = 3 + 4.2 + 3 + 1.5
    actual = circular_weighted_avg(forecasts, weights)
    assert_almost_equal(expected, actual, 3)


def test_weighted_avg():
    forecasts = [10, 14, 10, 15]
    weights = [0.3, 0.3, 0.3, 0.1]
    expected = 3 + 4.2 + 3 + 1.5
    actual = weighted_avg(forecasts, weights)
    assert_almost_equal(expected, actual)


def test_make_pf():
    grid = np.arange(30).reshape((2, 3, 5))

    weights = {(1, 1): 0.2, (2, 2): 0.3, (4, 0): 0.5} # note: this is (i, j) and the grid is (j, i)
    constant = 0.025

    result = calculate_pf(grid, weights, constant)
    expected = [0.2 * 6 + 0.3 * 12 + 0.5 * 4 + 0.025, 0.2 * 21 + 0.3 * 27 + 0.5 * 19 + 0.025]

    for (a, b) in zip(expected, result):
        assert_almost_equal(a, b)
