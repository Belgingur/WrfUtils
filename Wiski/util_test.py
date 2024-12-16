import pytest
from datetime import datetime
from util import extract_date


@pytest.mark.parametrize(
    "file_name, date_format, expected_date",
    [
        ("prefix_2024-12-31T14:23:45_suffix.txt", "prefix_%Y-%m-%dT%H:%M:%S_suffix", datetime(2024, 12, 31, 14, 23, 45)),
        ("log_2024-12-31_info.txt", "log_%Y-%m-%d_info", datetime(2024, 12, 31)),
        ("data_2024-12_extra.txt", "data_%Y-%m_extra", datetime(2024, 12, 1)),
        ("file_2024_misc.txt", "file_%Y_misc", datetime(2024, 1, 1)),
    ]
)
def test_extract_date(file_name, date_format, expected_date):
    assert extract_date(file_name, date_format) == expected_date


@pytest.mark.parametrize(
    "file_name, date_format, expected_error",
    [
        ("no_date_in_this_file.txt", "log_%Y-%m-%d_info", ValueError),
        ("2024-12-31_rest_of_the_file.txt", "prefix_%Y-%m-%dT%H:%M:%S_suffix", ValueError),
    ]
)
def test_extract_date_errors(file_name, date_format, expected_error):
    with pytest.raises(expected_error):
        extract_date(file_name, date_format)
