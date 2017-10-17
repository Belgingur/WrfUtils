"""

Common methods for operations with timeseries

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import codecs
import os
import logging
import re
from collections import OrderedDict

import yaml
import numpy as np

from utilities import make_regexp


LOG = logging.getLogger('belgingur.data_utils')
COMPONENTS_ORDER = {'temp': 1, 'wind_speed': 2, 'wind_dir': 3, 'prec_rate': 4, 'pressure': 5, 'total_clouds': 6}


# Timeseries

def save_timeseries(data, filepath, metadata=OrderedDict(), separator=', ', valueformat='%.4f', nodata=-9999,
                    comment_column_names=True):

    """ Save the timeseries with additional metadata in front. """

    LOG.info('Saving timeseries to %s', filepath)
    headers = ['# %s: %s' % (k, v) for k, v in metadata.iteritems()]

    def joiner(table):
        for row in zip(*table):
            yield ','.join(row)

    if isinstance(data, dict) and isinstance(data.get('timestamps'), np.ndarray):
        # We have a dict with a bunch of numpy arrays including one for timestamps
        data_keys = sorted(data.keys(), key=lambda x: COMPONENTS_ORDER.get(x, 1000))
        to_save = joiner(data[k] for k in data_keys)

    elif isinstance(data, dict):
        # We have a dict with a bunch of dicts from timestamps to values
        data_keys = sorted(data.keys(), key=lambda x: COMPONENTS_ORDER.get(x, 1000))
        ts_keys_sorted = sorted(list(set([d for dk in data_keys for d in data[dk].keys()])))
        to_save = []
        for ts in ts_keys_sorted:
            values = [ts.strftime('%Y-%m-%dT%H:%M')]
            for key in data_keys:
                values.append(valueformat % data[key].get(ts, nodata))
            to_save.append(separator.join(values))

    elif isinstance(data, tuple):
        # We have a tuple with a bunch of dicts from timestamps to values
        data_keys = sorted(data[1].keys(), key=lambda x: COMPONENTS_ORDER.get(x, 1000))
        ts_keys_sorted = sorted(data[0])
        to_save = []
        for i, ts in enumerate(ts_keys_sorted):
            values = [ts.strftime('%Y-%m-%dT%H:%M')]
            for key in data_keys:
                values.append(valueformat % data[1][key][i])
            to_save.append(separator.join(values))

    else:
        raise TypeError('Unsupported data type for the data: %s' % type(data))

    try:
        with codecs.open(filepath, 'w', encoding='utf-8') as out:
            out.write('\n'.join(headers) + ('\n' if headers else ''))
            out.write(separator.join(['# columns: time' if comment_column_names else 'time'] + data_keys) + '\n')
            for line in to_save:
                out.write(line)
                out.write('\n')

    except IOError as exc:
        LOG.exception('Problem with saving timeseries to file', exc)
        raise


# Files

def templated_filename(config, create_dirs=True, ext='csv', char_mapping=None, **kwargs):

    """ Create filename for the output based on template. """

    out_dir = config.get('store_dir', '')
    template = config.get('output_template', 'pf-{ref}-{analysis_date:%Y-%m-%d_%H:%M:%S}.' + ext)
    try:
        path = os.path.join(out_dir, map_chars(unicode(template).format(**kwargs), char_mapping))
    except KeyError as e:
        LOG.error('The following keys in the template %s are not provided: %s', template, e)
        raise
    directory = os.path.dirname(path)
    if not os.path.exists(directory.encode('utf-8')) and create_dirs and directory != '':
        os.makedirs(directory.encode('utf-8'))

    return path


# Languages

def map_chars(text, char_map):
    if char_map is None:
        return text
    for non_asc, asc in char_map.iteritems():
        text = text.replace(non_asc, asc)

    try:
        text.decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError):
        LOG.warning('Not all non-ascii characters have been removed from the string `%s`. Is your character map complete?', text)
    return text


def load_point_metadata(config):

    """ Load information about stations and select those relevant to stations we are interested in. """

    stations_img = config['station_metadata']
    known_stations = read_all_stations(stations_img)

    stations_pf = select_stations(config.get('stations', []), known_stations)

    return stations_pf


def read_all_stations(filename):

    """ Read all available stations info stored in the yaml file and return them as a list. """

    LOG.info("Reading stations from file %s", filename)
    try:
        with codecs.open(filename, encoding='utf-8') as yml_file:
            data = yaml.safe_load(yml_file)
        return data

    except (IOError, KeyError, ValueError, yaml.scanner.ScannerError) as exc:
        LOG.error('Problem with reading from file. Does the file exist? Is it in the correct format?')
        LOG.exception(exc)
        raise


def select_stations(target, source):

    """ From the available stations list, choose those we are interested in for a particular point forecast.

    Target stations/POI for processing can be given either by single entry or a list.
    Each entry (single or within a list) can have wildcards for regexp-like matching.

    Rules for wildcard/regexp matching:
     - % is interpreted as 'match any 0 or more characters' (like in sql)
     - dots (.) are interpreted as dots
     - other rules (number matching e.g. [1-9][0-9]* etc.) are interpreted as in regexp

    Example:
     'lv.is.%' will match all stations from Landsvirkjun/IS
     'lv.(is|fo).5' will match station 5 from both lv.is and lv.fo
    """

    data = {s['ref']: s for s in source}

    all_keys = [k for k, v in data.iteritems()]

    if not type(target) == list:
        target = [target]

    required_keys = []

    for ref in target:
        ref = make_regexp(ref)
        required_keys.extend([k for k in all_keys if re.match(ref, k)])

    required_keys = list(set(required_keys))
    return [data[k] for k in required_keys]


