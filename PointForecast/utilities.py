import codecs
import logging
import logging.config
from datetime import datetime

import yaml
from pytz import utc


LOG = logging.getLogger('belgingur.utilities')


def load_config(config_file_path):
    with codecs.open(config_file_path, encoding='utf-8') as yml_file:
        config = yaml.safe_load(yml_file)
    LOG.info('Loaded config file %s', config_file_path)
    return config


def configure_logging(config_file_path):
    with open(config_file_path) as config:
        log_config = yaml.load(config)
    logging.config.dictConfig(log_config)
    LOG.info('Configured logging from %s', config_file_path)


def make_regexp(ref):

    """ Convert format of station refs patterns to a proper regexp """

    if not ref.startswith('^'):
        ref = '^' + ref
    if not ref.endswith('$'):
        ref += '$'
    ref = ref.replace('.', '\.').replace('%', '.*')
    return ref


def mk_datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=utc):
    """
    Builds a UTC datetime from the given fields after converting them to int

    :rtype: datetime
    """
    year, month, day, hour, minute, second, microsecond = \
        (int(s) for s in (year, month, day, hour, minute, second, microsecond))
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo)


def parse_iso_date(s):
    """ Parse a datetime from string in ISO date format. Assumes UTC if no TZ is given """

    s = s.strip()
    FORMATS = ('%Y-%m-%dT%H:%MZ',
               '%Y-%m-%dT%H:%M',
               '%Y-%m-%dT%H:%M:%SZ',
               '%Y-%m-%dT%H:%M:%S',)
    for fmt in FORMATS:
        try:
            ts = datetime.strptime(s, fmt)
            if not ts.tzinfo:
                ts = ts.replace(tzinfo=utc)
            return ts
        except ValueError as e:
            pass
    raise ValueError('%s is not in a supported date format: %s' % (s, ', '.join(FORMATS)))
