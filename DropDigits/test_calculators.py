import logging
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, call, patch

from calculators import CALCULATORS, ChunkCalculator
from utils import DIM_BOTTOM_TOP
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
