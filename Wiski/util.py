import re
from datetime import datetime


def extract_date(file_name, date_format):
    """
    Extracts a date from a file name using the specified format string.

    Args:
        file_name (str): Name of the file.
        date_format (str): Date format string (e.g., "%Y-%m-%d").

    Returns:
        datetime: Extracted datetime object.

    Raises:
        ValueError: If no date is found in the file name.
    """
    # Convert the date_format to a regex pattern
    pattern = re.sub(r'%[YmdHMS]', r'(\\d+)', re.escape(date_format))
    match = re.search(pattern, file_name)
    if not match:
        raise ValueError(f"No date found in file name: {file_name}")

    try:
        return datetime.strptime(match.group(0), date_format)
    except Exception as e:
        raise ValueError(f"Failed to parse date from file name: {file_name}. Error: {e}")

