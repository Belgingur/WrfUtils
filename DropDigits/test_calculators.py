import logging
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, call, patch

import numpy as np

from calculators import CALCULATORS, ChunkCalculator, derived
from utils import DIM_BOTTOM_TOP, DIM_BOTTOM_TOP_STAG
from utils_testing import mock_dataset_meta

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = str(TEST_DIR / 'wrfout-africa-50.nc')
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

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
    ipor_stag = MagicMock(name='ipor_stag')  # Unused

    ipor_alig = MagicMock(name='ipor_alig')  # Interpolates T
    ipor_alig.dimension = DIM_BOTTOM_TOP
    ipor_alig.max_k = 7

    T_ORIGINAL = in_ds.variables['T'].__getitem__.return_value
    T_ORIGINAL.shape = (10, 10, 100, 100)
    T_INTERPOLATED = ipor_alig.return_value

    # Make the call
    c = ChunkCalculator(in_ds, 10, 20, ipor_alig, ipor_stag)
    T = c('T')

    # Retrieved data
    assert in_ds.variables['T'].__getitem__.call_args_list == [
        call((
            slice(10, 20),  # Time slice
            slice(0, 8),  # Vertical slice
        ))
    ]
    # Interpolated on bottom_top
    assert ipor_alig.call_args_list == [
        call(T_ORIGINAL),
    ]
    # Returned the interpolated value
    assert T is T_INTERPOLATED


def test_calculator_native_staggered():
    in_ds = mock_africa_ds()
    ipor_stag = MagicMock(name='ipor_stag')  # Unused

    ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    ipor_alig.dimension = DIM_BOTTOM_TOP
    ipor_alig.max_k = 5

    with patch('utils.destagger_array') as destagger_array:
        U_ORIGINAL = in_ds.variables['U'].__getitem__.return_value
        U_ORIGINAL.shape = (7, 10, 80,70)
        U_DESTAGGERED = destagger_array.return_value
        U_INTERPOLATED = ipor_alig.return_value

        # Make the call
        c = ChunkCalculator(in_ds, 5, 12, ipor_alig, ipor_stag)
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
        assert ipor_alig.call_args_list == [
            call(U_DESTAGGERED),
        ]
        # Returned the interpolated value
        assert U is U_INTERPOLATED


def test_calculator_derived():
    in_ds = mock_africa_ds()
    ipor_stag = MagicMock(name='ipor_stag')  # Unused
    ipor_stag.dimension = DIM_BOTTOM_TOP_STAG
    ipor_stag.max_k = 5

    ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    ipor_alig.dimension = DIM_BOTTOM_TOP
    ipor_alig.max_k = 6

    with patch('utils.destagger_array') as destagger_array:
        VAR_U = in_ds.variables['U'].__getitem__.return_value
        VAR_U.shape = (7, 10, 80, 71)
        VAR_U.destagger_3.shape = (7, 10, 80, 70)
        VAR_U.destagger_3.ipor_alig.shape = (7, 3, 80, 70)

        VAR_V = in_ds.variables['V'].__getitem__.return_value
        VAR_V.shape = (7, 10, 81, 70)
        VAR_V.destagger_2.shape = (7, 10, 80, 70)
        VAR_V.destagger_2.ipor_alig.shape = (7, 3, 80, 70)

        destagger_array.side_effect = lambda var, axis, **kw: getattr(var, 'destagger_'+str(axis))
        ipor_alig.side_effect = lambda var: var.ipor_alig
        ipor_stag.side_effect = lambda var: var.ipor_stag

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
        c = ChunkCalculator(in_ds, 32, 64, ipor_alig, ipor_stag)
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
        assert ipor_alig.call_args_list == [
            call(VAR_U.destagger_3),
            call(VAR_V.destagger_2),
        ]
        # Returned the interpolated value
        assert MR is MOCK_RESULT


def test_calculator_height():
    in_ds = MagicMock()
    ipor_alig = MagicMock(name='ipor_alig')  # Interpolates U
    ipor_alig.heights = [10, 50, 80, 100, 150]
    ipor_alig.dimension = DIM_BOTTOM_TOP
    ipor_alig.max_k = 6

    # Make the call
    c = ChunkCalculator(in_ds, 5, 12, ipor_alig, None)
    height = c('height')

    assert type(height) == np.ndarray
    assert height.tolist() == [10, 50, 80, 100, 150]
