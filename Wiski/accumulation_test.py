from argparse import Namespace
from datetime import datetime

import numpy as np
import pytz

from accumulation import read_from_rate
from make_masks import ConfigGetter


class TestConfigGetter(ConfigGetter):
    def __init__(self, data):
        self._config = data
        self._args = Namespace()
        self._sim_config = {}


def test_read_from_rate():
    """
    Assume that at least `202310-carra-sfc_wod.nc` and `202311-carra-sfc_wod.nc` are present in the `accumulation_test` subfolder
    """
    cfg = TestConfigGetter(dict(
        wrfouts='accumulation_test/*-carra-sfc_wod.nc',
        date_pattern='%Y%m-carra-sfc_wod.nc'
    ))
    date_from = datetime(2023, 10, 15, tzinfo=pytz.UTC)
    date_to = datetime(2023, 11, 15, tzinfo=pytz.UTC)
    result = read_from_rate(cfg, date_to, date_from, True)
    sum_result = np.sum(result)
    expected = 4668162.75
    assert abs(sum_result - expected) < 0.001
