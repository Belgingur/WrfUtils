import logging
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from Elevator import DIM_NAMES_GEO
from calculators import CALCULATORS, ChunkCalculator, derived, wind_dir, wind_dir_10
from utils import DIM_BOTTOM_TOP
from utils_testing import mock_dataset_meta, slice_call

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = str(TEST_DIR / 'wrfout-africa-50.nc')
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

WRFOUT_RÁV = TEST_DIR / 'wrfout-ráv2.ncdump'
GEO_RÁV = TEST_DIR / 'geo-ráv2.ncdump'

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)


def mock_africa_ds():
    return mock_dataset_meta(WRFOUT_AFRICA_DUMP)


def test_init():
    LOG.info('CALCULATORS: %s', CALCULATORS)
    for name, calc in CALCULATORS.items():
        LOG.info('%s(%s): %s', name, ', '.join(calc.inputs), ', '.join(calc.dimensions))

    assert isinstance(CALCULATORS['U_true'], FunctionType)

    assert CALCULATORS['U_true'].inputs == ('U', 'V', 'COSALPHA', 'SINALPHA')
    assert CALCULATORS['U_true'].dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')

    assert CALCULATORS['V_true'].inputs == ('U', 'V', 'COSALPHA', 'SINALPHA')
    assert CALCULATORS['V_true'].dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')

    assert CALCULATORS['wind_speed'].inputs == ('U', 'V')
    assert CALCULATORS['wind_speed'].dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')

    assert CALCULATORS['wind_dir'].inputs == ('U_true', 'V_true')
    assert CALCULATORS['wind_dir'].dimensions == ('Time', 'bottom_top', 'south_north', 'west_east')


def test_calculator_native_aligned():
    in_ds = mock_africa_ds()

    c = ChunkCalculator(10, 20, [10, 50, 100], True)
    c._ipor_alig = MagicMock(name='ipor_alig')  # Interpolates T
    c._ipor_alig.dimension = DIM_BOTTOM_TOP
    c._ipor_alig.max_k = 7

    T_ORIGINAL = in_ds.variables['T'].__getitem__.return_value
    T_ORIGINAL.shape = (10, 10, 100, 100)
    T_INTERPOLATED = c._ipor_alig.return_value

    # Make the call
    c.add_dataset(in_ds)
    T = c('T')

    # Retrieved data
    assert in_ds.variables['T'].__getitem__.call_args_list == [
        call((
            slice(10, 20),  # Time slice
            slice(0, 8),  # Vertical slice
        ))
    ]
    # Interpolated on bottom_top
    assert c._ipor_alig.call_args_list == [
        call(T_ORIGINAL),
    ]
    # Returned the interpolated value
    assert T is T_INTERPOLATED


def test_calculator_native_staggered():
    in_ds = mock_africa_ds()
    c = ChunkCalculator(5, 12, [33, 66, 99], True)
    c.add_dataset(in_ds)
    c._ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    c._ipor_alig.dimension = DIM_BOTTOM_TOP
    c._ipor_alig.max_k = 5

    with patch('utils.destagger_array') as destagger_array:
        U_ORIGINAL = in_ds.variables['U'].__getitem__.return_value
        U_ORIGINAL.shape = (7, 10, 80, 70)
        U_DESTAGGERED = destagger_array.return_value
        U_INTERPOLATED = c._ipor_alig.return_value

        # Make the call
        U = c('U')

        # Retrieved data
        assert in_ds.variables['U'].__getitem__.call_args_list == [
            call((
                slice(5, 12),  # Time slice
                slice(0, 6)  # Vertical slice
            ))
        ]
        # Destagger U along the west_east axis
        assert destagger_array.call_args_list == [
            call(U_ORIGINAL, 3),
        ]
        # Interpolated on bottom_top
        assert c._ipor_alig.call_args_list == [
            call(U_DESTAGGERED),
        ]
        # Returned the interpolated value
        assert U is U_INTERPOLATED


def test_calculator_derived():
    in_ds = mock_africa_ds()
    c = ChunkCalculator(32, 64, [50, 150, 300], True)
    c.add_dataset(in_ds)

    c._ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    c._ipor_alig.dimension = DIM_BOTTOM_TOP
    c._ipor_alig.max_k = 6
    c._ipor_alig.side_effect = lambda var: var.ipor_alig

    with patch('utils.destagger_array') as destagger_array:
        VAR_U = in_ds.variables['U'].__getitem__.return_value
        VAR_U.shape = (7, 10, 80, 71)
        VAR_U.destagger_3.shape = (7, 10, 80, 70)
        VAR_U.destagger_3.ipor_alig.shape = (7, 3, 80, 70)

        VAR_V = in_ds.variables['V'].__getitem__.return_value
        VAR_V.shape = (7, 10, 81, 70)
        VAR_V.destagger_2.shape = (7, 10, 80, 70)
        VAR_V.destagger_2.ipor_alig.shape = (7, 3, 80, 70)

        destagger_array.side_effect = lambda var, axis, **kw: getattr(var, 'destagger_' + str(axis))

        MOCK_RESULT = MagicMock(name='MOCK_RESULT')
        MOCK_MOCK_RESULT = MagicMock(name='MOCK_MOCK_RESULT')

        @derived('Time', 'bottom_top', 'west_east', 'south_north')
        def mock(U, V, mock_mock):
            LOG.info('U: %s', U)
            LOG.info('V: %s', V)
            LOG.info('mock_mock: %s', mock_mock)
            assert U == VAR_U.destagger_3.ipor_alig
            assert V == VAR_V.destagger_2.ipor_alig
            assert mock_mock is MOCK_MOCK_RESULT
            return MOCK_RESULT

        @derived('Time', 'bottom_top', 'west_east', 'south_north')
        def mock_mock(U):
            LOG.info('U: %s', U)
            assert U == VAR_U.destagger_3.ipor_alig
            return MOCK_MOCK_RESULT

        # Make the call
        MR = c('mock')

        # Retrieved data
        assert in_ds.variables['U'].__getitem__.call_args_list == [
            call((
                slice(32, 64),  # Time slice
                slice(0, 7),  # Vertical slice
            )),
        ]
        # Destagger U along the west_east axis
        assert destagger_array.call_args_list == [
            call(VAR_U, 3),
            call(VAR_V, 2),
        ]
        # Interpolated on bottom_top
        assert c._ipor_alig.call_args_list == [
            call(VAR_U.destagger_3),
            call(VAR_V.destagger_2),
        ]
        # Returned the interpolated value
        assert MR is MOCK_RESULT


def test_calculator_height():
    in_ds = MagicMock()
    in_ds.variables = dict(
        T=MagicMock(),
        U=MagicMock(),
        V=MagicMock(),
    )
    c = ChunkCalculator(5, 12, [10, 50, 80, 100, 150], True)
    c.add_dataset(in_ds)
    c._ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    c._ipor_alig.heights = c.heights
    c._ipor_alig.dimension = DIM_BOTTOM_TOP
    c._ipor_alig.max_k = 6

    # Make the call
    height = c('height')

    assert type(height) == np.ndarray
    assert height.tolist() == [10, 50, 80, 100, 150]


def test_fallback():
    in_ds = MagicMock()
    in_ds.variables = dict(
        T=MagicMock(),
        U=MagicMock(),
        V=MagicMock(),
    )

    HGT = MagicMock(name='HGT')
    HGT.dimensions = ('south_north', 'west_east')
    geo_ds = MagicMock()
    geo_ds.variables = dict(
        XLAT=MagicMock(),
        XLONG=MagicMock(),
        HGT=HGT,
    )

    c = ChunkCalculator(10, 20, [100, 200, 500], False)
    c.add_dataset(in_ds)  # Does not have the geo variables
    c.add_dataset(geo_ds, 10, DIM_NAMES_GEO)
    r = c('HGT')

    assert r is HGT.__getitem__.return_value
    assert HGT.__getitem__.call_args_list == [slice_call[10:20, ..., 10:-10, 10:-10]]


def test_fallback_geo_z():
    in_ds = mock_dataset_meta(WRFOUT_RÁV)
    geo_ds = mock_dataset_meta(GEO_RÁV)

    c = ChunkCalculator(32, 64, [100, 200, 500], True)
    c.add_dataset(in_ds)  # Does not have the geo variables
    c.add_dataset(geo_ds, 10, DIM_NAMES_GEO)
    c.z_stag()

    assert in_ds.variables['PH'].__getitem__.call_args_list == [slice_call[32:64, ..., :, :]]
    assert in_ds.variables['PHB'].__getitem__.call_args_list == [slice_call[32:64, ..., :, :]]
    assert geo_ds.variables['HGT_M'].__getitem__.call_args_list == [slice_call[0, 10:-10, 10:-10]]


def test_static_cosalpha_sinalpha():
    in_ds = mock_dataset_meta(WRFOUT_RÁV)
    geo_ds = mock_dataset_meta(GEO_RÁV)

    c = ChunkCalculator(32, 64, [100, 200, 500], True)
    c.add_dataset(in_ds)  # Does not have the geo variables
    c.add_dataset(geo_ds, 10, DIM_NAMES_GEO)
    c.make_vars_static('COSALPHA', 'SINALPHA')
    c('COSALPHA')
    assert geo_ds.variables['COSALPHA'].__getitem__.call_args_list == [slice_call[0, ..., 10:-10, 10:-10]]
    c('SINALPHA')
    assert geo_ds.variables['SINALPHA'].__getitem__.call_args_list == [slice_call[0, ..., 10:-10, 10:-10]]


@pytest.mark.parametrize('v, u, ex_dir', [
    (-1, +0, 000.),
    (+0, -1, 090.),
    (+1, +0, 180.),
    (+0, +1, 270.),
    (+1, +1, 225.),
    (+1, -1, 135.),
    (-1, +1, 315.),
    (-1, -1, 045.),
])
def test_wind_dir(u: float, v: float, ex_dir: float):
    ac_dir = wind_dir(u, v)
    assert ac_dir == ex_dir


@pytest.mark.parametrize('v, u, ex_dir', [
    (-1, +0, 000.),
    (+0, -1, 090.),
    (+1, +0, 180.),
    (+0, +1, 270.),
    (+1, +1, 225.),
    (+1, -1, 135.),
    (-1, +1, 315.),
    (-1, -1, 045.),
])
def test_wind_dir_10(u: float, v: float, ex_dir: float):
    ac_dir = wind_dir_10(u, v)
    assert ac_dir == ex_dir
