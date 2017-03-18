import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np

from utils import read_wrf_dates, destagger_array
from utils_testing import mock_dataset_meta, slice_call

LOG = logging.getLogger(__name__)

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = TEST_DIR / 'wrfout-africa-50.nc'
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

logging.basicConfig(level=logging.INFO)


def test_work_wrf_dates():
    times = [
        b'2016-10-24_06:00:00', b'2016-10-24_07:00:00', b'2016-10-24_08:00:00', b'2016-10-24_09:00:00',
        b'2016-10-24_10:00:00', b'2016-10-24_11:00:00', b'2016-10-24_12:00:00', b'2016-10-24_13:00:00',
        b'2016-10-24_14:00:00', b'2016-10-24_15:00:00', b'2016-10-24_16:00:00', b'2016-10-24_17:00:00',
        b'2016-10-24_18:00:00', b'2016-10-24_19:00:00', b'2016-10-24_20:00:00', b'2016-10-24_21:00:00',
        b'2016-10-24_22:00:00', b'2016-10-24_23:00:00', b'2016-10-25_00:00:00', b'2016-10-25_01:00:00',
        b'2016-10-25_02:00:00', b'2016-10-25_03:00:00', b'2016-10-25_04:00:00', b'2016-10-25_05:00:00',
        b'2016-10-25_06:00:00', b'2016-10-25_07:00:00', b'2016-10-25_08:00:00', b'2016-10-25_09:00:00'
    ]
    times = [[b[i:i + 1] for i in range(len(b))] for b in times]
    times = np.array(times)

    ds = mock_dataset_meta(WRFOUT_AFRICA_DUMP)
    ds.variables['Times'].__getitem__.return_value = times

    dates = read_wrf_dates(ds)
    assert isinstance(dates, np.ndarray)
    assert len(dates) == 28
    for d in dates:
        assert isinstance(d, datetime)
    for d1, d2 in zip(dates[:-1], dates[1:]):
        assert d1 < d2

def test_mock_dataset_africa():
    dsa = mock_dataset_meta(WRFOUT_AFRICA_DUMP)

    # Uncomment to verify that tets pass against a real Dataet
    # dsa = Dataset(str(WRFOUT_AFRICA))

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
    assert T.datatype == np.dtype('float32')  # Yes, all of these are true
    assert T.datatype == np.float32
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


def test_destagger_array_0():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    b = destagger_array(a, axis=0)
    assert type(b) == np.ndarray
    assert b.tolist() == [
        [2.5, 3.5, 4.5],
        [5.5, 6.5, 7.5],
    ]


def test_destagger_array_1():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    b = destagger_array(a, axis=1)
    assert type(b) == np.ndarray
    assert b.tolist() == [
        [1.5, 2.5],
        [4.5, 5.5],
        [7.5, 8.5],
    ]


def test_slices_call():
    assert slice_call[0] == call(0)
    assert slice_call[0, 2, 4] == call((0, 2, 4))
    assert slice_call[10:-10:2] == call(slice(10, -10, 2))
    assert slice_call[10:-10, ..., 5, 2::4] == call((
        slice(10, -10),
        Ellipsis,
        5,
        slice(2, None, 4)
    ))
