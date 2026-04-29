"""Generic tools library"""


def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def seconds_to_hms(seconds: float) -> str:
    """Convert a number of seconds to an H:MM:SS string."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def hours_to_hms(hours: float) -> str:
    """Convert hours to HH:MM:SS format."""
    return seconds_to_hms(hours * 3600)


def hms_to_hours(time_str: str) -> float:
    """Convert HH:MM:SS to hours."""
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600
