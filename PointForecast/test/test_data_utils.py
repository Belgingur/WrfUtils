# encoding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import codecs
import os

from nose.tools import assert_list_equal, assert_equal, assert_raises
from mock import patch
from utilities import mk_datetime
from data_utils import save_timeseries, templated_filename, map_chars, select_stations, load_stations
import data_utils

test_path = os.path.join(os.path.dirname(__file__), 'data', 'test_timeseries.txt')


ts = [mk_datetime(2014, 2, 1, 10), mk_datetime(2014, 2, 1, 11)]
data = dict(
    wind_dir=[2.454545, 2.5]
)

ts2 = [mk_datetime(2014, 2, 1, 10), mk_datetime(2014, 2, 1, 11), mk_datetime(2014, 2, 1, 12)]
data2 = dict(
    wind_dir=[2.454545, 2.5, float('nan')],
    temp=[10.6161616, float('nan'), 10.2]
)


def test_save_timeseries__simple():
    save_timeseries(ts, data, test_path)
    with open(test_path) as f:
        lines = f.readlines()
    assert_list_equal([
        'time, wind_dir\n',
        '2014-02-01T10:00, 2.4545\n',
        '2014-02-01T11:00, 2.5000\n'
    ], lines)


def test_save_timeseries__header():
    save_timeseries(ts, data, test_path, {'this_is_header': 'HEADER'})
    with open(test_path) as f:
        lines = f.readlines()
    assert_list_equal([
        '# this_is_header: HEADER\n',
        'time, wind_dir\n',
        '2014-02-01T10:00, 2.4545\n',
        '2014-02-01T11:00, 2.5000\n'
    ], lines)


def test_save_timeseries__formating():
    save_timeseries(ts, data, test_path, separator='::', valueformat='{:.2f}')
    with open(test_path) as f:
        lines = f.readlines()
    assert_list_equal([u'time::wind_dir\n', '2014-02-01T10:00::2.45\n', '2014-02-01T11:00::2.50\n'], lines)


def test_save_timeseries__simple2():
    save_timeseries(ts2, data2, test_path)
    with open(test_path) as f:
        lines = f.readlines()
    assert_list_equal([
        'time, temp, wind_dir\n',
        '2014-02-01T10:00, 10.6162, 2.4545\n',
        '2014-02-01T11:00, -9999, 2.5000\n',
        '2014-02-01T12:00, 10.2000, -9999\n'
    ], lines)


def test_save_timeseries__utf8():
    save_timeseries(ts, data, test_path, {'name': 'Ólafsfjarðarmúli'})
    with codecs.open(test_path, encoding='UTF8') as f:
        lines = f.readlines()
    assert_list_equal([
        '# name: Ólafsfjarðarmúli\n',
        'time, wind_dir\n',
        '2014-02-01T10:00, 2.4545\n',
        '2014-02-01T11:00, 2.5000\n'
    ], lines)


def test_save_timeseries__header__not_comment_column_names():
    save_timeseries(ts, data, test_path, {'name': 'Ólafsfjarðarmúli'}, comment_column_names=False)
    with codecs.open(test_path, encoding='UTF8') as f:
        lines = f.readlines()
    assert_list_equal([
        '# name: Ólafsfjarðarmúli\n',
        'time, wind_dir\n',
        '2014-02-01T10:00, 2.4545\n',
        '2014-02-01T11:00, 2.5000\n'
    ], lines)


def test_save_timeseries__error():
    with assert_raises(TypeError):
        save_timeseries('a;b;c;d;', '.')


def test_templated_filename():
    config = {'store_dir': '/home/me/nothing'}
    analysis = mk_datetime(2015, 4, 1, 10)
    station_ref = 'vi.is.wroclaw'

    expected = '/home/me/nothing/pf-vi.is.wroclaw-2015-04-01_10:00:00.csv'
    assert_equal(expected, templated_filename(config, analysis_date=analysis, ref=station_ref, create_dirs=False))

    config = {'output_template': 'pf/output-{analysis_date:%Y-%m-%d_%H:%M:%S}/pf-{station_ref}.csv'}

    expected = 'pf/output-2015-04-01_10:00:00/pf-vi.is.wroclaw.csv'
    assert_equal(expected, templated_filename(config, analysis_date=analysis, station_ref=station_ref, create_dirs=False))

    config = {'output_template': 'pf/output-{analysis_date:%Y-%m-%d_%H:%M:%S}/Poland/{station_name}/pf-{station_ref}.csv'}
    expected = 'pf/output-2015-04-01_10:00:00/Poland/Wroclaw/pf-vi.is.wroclaw.csv'
    meta = {'station_name': 'Wroclaw'}
    assert_equal(expected, templated_filename(config, analysis_date=analysis, station_ref=station_ref, create_dirs=False, **meta))


def test_map_chars():
    char_map = {'ó': 'o', 'ą': 'a', 'ź': 'z', 'ż': 'z', 'ń': 'n'}

    assert_equal('zyzn', map_chars('żyźń', char_map))
    assert_equal('ćn', map_chars('ćń', char_map))


def test_select_stations():
    source = [
        {'ref': 'vi.is.1'},
        {'ref': 'vi.is.2'},
        {'ref': 'vi.is.3'},
        {'ref': 'vi.is.5'},
        {'ref': 'vi.is.55'},
        {'ref': 'vi.is.53'},
        {'ref': 'vi.is.51'},
        {'ref': 'vi.si.1'},
    ]
    target = 'vi.is.%'

    result = select_stations(target, source)
    assert_list_equal(sorted([r['ref'] for r in result]), sorted(['vi.is.1', 'vi.is.2', 'vi.is.55', 'vi.is.53', 'vi.is.3', 'vi.is.51', 'vi.is.5']))

    target = 'vi.si.%'

    result = select_stations(target, source)
    assert_list_equal([r['ref'] for r in result], ['vi.si.1'])

    target = ['vi.%.1', 'vi.is.5%']

    result = select_stations(target, source)
    assert_list_equal(sorted([r['ref'] for r in result]), sorted(['vi.si.1', 'vi.is.1', 'vi.is.55', 'vi.is.53', 'vi.is.51', 'vi.is.5']))

    result = select_stations([], source)
    assert_list_equal([], result)

    target = 'vi.is.(53|55)'
    result = select_stations(target, source)
    assert_equal(sorted([r['ref'] for r in result]), sorted(['vi.is.53', 'vi.is.55']))

    target = 'vi.is.[1-3]'
    result = select_stations(target, source)
    assert_equal(sorted([r['ref'] for r in result]), sorted(['vi.is.1', 'vi.is.2', 'vi.is.3']))

    target = 'vi.is.[15]'
    result = select_stations(target, source)
    assert_equal(sorted([r['ref'] for r in result]), sorted(['vi.is.1', 'vi.is.5']))


def test_load_point_metadata__single_file():
    cfg_all_stations = dict(
        key='value',
        stations=[
            dict(lat=10, lon=10, ref='10', name='Station 10'),
            dict(lat=11, lon=11, ref='11', name='Station 11')
        ]
    )
    expected = [
            dict(lat=10, lon=10, ref='10', name='Station 10'),
            dict(lat=11, lon=11, ref='11', name='Station 11')
    ]
    actual = load_stations(cfg_all_stations)
    assert_list_equal(expected, actual)


def test_load_point_metadata__meta_file():
    meta_stations = [
        dict(lat=10, lon=10, ref='10', name='Station 10'),
        dict(lat=11, lon=11, ref='11', name='Station 11'),
        dict(lat=55, lon=55, ref='55', name='Station 55')
    ]
    config = dict(
        station_metadata='stations.yml',
        stations='1%'
    )
    expected = [
            dict(lat=10, lon=10, ref='10', name='Station 10'),
            dict(lat=11, lon=11, ref='11', name='Station 11')
    ]

    with patch('data_utils.read_all_stations', return_value=meta_stations):
        actual = data_utils.load_stations(config)
    assert_list_equal(expected, sorted(actual))