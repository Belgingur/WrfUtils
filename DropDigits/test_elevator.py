import logging
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np

from Elevator import resolve_input_variables, resolve_input_dimensions, resolve_output_dimensions, \
    create_output_dimensions, create_output_variables, destagger_dim_name
from utils_testing import mock_dataset_meta

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR / 'test_data'
WRFOUT_AFRICA = str(TEST_DIR / 'wrfout-africa-50.nc')
WRFOUT_AFRICA_DUMP = TEST_DIR / 'wrfout-africa-50.ncdump'

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger(__name__)


def mock_africa_ds():
    return mock_dataset_meta(WRFOUT_AFRICA_DUMP)


def test_resolve_input_variables():
    ds = mock_africa_ds()
    names = ['U', 'V', 'T', 'TKE_PBL']
    var = resolve_input_variables(ds, names)
    var_names = [v.name for v in var]
    assert var_names == names


def test_resolve_input_dimensions():
    in_ds = mock_africa_ds()
    out_var_names = ['U', 'V', 'T', 'TKE_PBL']
    in_vars = resolve_input_variables(in_ds, out_var_names)
    in_dim_names = resolve_input_dimensions(in_vars)
    assert set(in_dim_names) == {'Time', 'bottom_top', 'south_north', 'west_east',
                                 'bottom_top_stag', 'south_north_stag', 'west_east_stag'}


def test_resolve_output_dimensions():
    in_dim_names = ['Time', 'bottom_top', 'south_north', 'west_east_stag',
                    'west_east', 'south_north_stag', 'bottom_top_stag']
    out_dim_names = resolve_output_dimensions(in_dim_names)
    assert set(out_dim_names) == {'Time', 'bottom_top', 'south_north', 'west_east'}


def test_create_output_dimensions():
    in_ds = mock_africa_ds()
    out_ds = MagicMock(name='out_ds')
    out_dim_names = ['Time', 'bottom_top', 'south_north', 'west_east']
    create_output_dimensions(in_ds, out_ds, out_dim_names, 5)
    assert out_ds.createDimension.call_args_list == [
        call('Time', None),
        call('bottom_top', 5),
        call('south_north', 199),
        call('west_east', 184),
    ]


def test_create_output_variables():
    in_ds = mock_africa_ds()

    out_ds = MagicMock(name='out_ds')
    out_var_names = ['U', 'V', 'T', 'TKE_PBL']
    create_output_variables(in_ds, out_ds, out_var_names, 7, True, 5)
    assert out_ds.createVariable.call_args_list == [
        call('U', np.float32, chunksizes=(128, 5, 16, 16), complevel=7,
             dimensions=['Time', 'bottom_top', 'south_north', 'west_east'], shuffle=True, zlib=True),
        call('V', np.float32, chunksizes=(128, 5, 16, 16), complevel=7,
             dimensions=['Time', 'bottom_top', 'south_north', 'west_east'], shuffle=True, zlib=True),
        call('T', np.float32, chunksizes=(128, 5, 16, 16), complevel=7,
             dimensions=['Time', 'bottom_top', 'south_north', 'west_east'], shuffle=True, zlib=True),
        call('TKE_PBL', np.float32, chunksizes=(128, 5, 16, 16), complevel=7,
             dimensions=['Time', 'bottom_top', 'south_north', 'west_east'], shuffle=True, zlib=True)
    ]


def test_destagger_name():
    assert destagger_dim_name('west_east') == 'west_east'
    assert destagger_dim_name('west_east_stag') == 'west_east'
    assert destagger_dim_name('south_north') == 'south_north'
    assert destagger_dim_name('south_north_stag') == 'south_north'
    assert destagger_dim_name('bottom_top') == 'bottom_top'
    assert destagger_dim_name('bottom_top_stag') == 'bottom_top'
