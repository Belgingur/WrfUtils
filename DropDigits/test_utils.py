import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from netCDF4 import Dataset

from utils import read_wrf_dates
from utils_testing import mock_dataset_meta

LOG = logging.getLogger(__name__)

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = TEST_DIR / 'wrfout-africa-50.nc'
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

logging.basicConfig(level=logging.INFO)


def test_work_wrf_dates():
    ds = Dataset(WRFOUT_AFRICA)
    dates = read_wrf_dates(ds)
    assert isinstance(dates, np.ndarray)
    assert len(dates) == len(ds.variables['Times'])
    for d in dates:
        assert isinstance(d, datetime)
    for d1, d2 in zip(dates[:-1], dates[1:]):
        assert d1 < d2


def test_mock_dataset_africa():
    dsa = mock_dataset_meta(WRFOUT_AFRICA_DUMP)

    # Uncomment to verify that tets pass against a real Dataet
    # dsa = Dataset('wrfout-africa-50.nc')

    assert dsa is not None
    assert dsa.START_DATE == '2016-10-24_06:00:00'
    assert type(dsa.START_DATE) == str
    assert dsa.ISOILWATER == 14
    assert type(dsa.ISOILWATER) == np.int32
    assert dsa.POLE_LAT == 90.0
    assert type(dsa.POLE_LAT) == np.float32

    assert isinstance(dsa.variables, dict)
    assert dsa.dimensions['bottom_top'].size == 40
    assert dsa.dimensions['west_east_stag'].size == 185
    assert dsa.dimensions['soil_layers_stag'].size == 4

    assert isinstance(dsa.dimensions, dict)

    T = dsa.variables['T']
    assert T.dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')
    assert T.datatype == 'float32'
    assert T.units == 'K'

    Times = dsa.variables['Times']
    assert Times.dimensions == ('Time', 'DateStrLen')
    assert Times.datatype == '|S1'
    assert not hasattr(Times, 'units') or isinstance(getattr(Times, 'units'), MagicMock)

    U = dsa.variables['U']
    assert U.dimensions == ('Time', 'bottom_top', 'south_north', 'west_east_stag')
    assert U.datatype == 'float32'
    assert U.units == 'm s-1'

    SST = dsa.variables['SST']
    assert SST.dimensions == ('Time', 'south_north', 'west_east')
    assert U.datatype == 'float32'
    assert SST.units == 'K'
    assert SST.coordinates == 'XLONG XLAT'
