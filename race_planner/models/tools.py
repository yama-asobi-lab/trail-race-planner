"""Generic tools library"""


# Convert time strings to seconds
def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def hours_to_hms(hours: float) -> str:
    """Convert hours to HH:MM:SS format."""
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def hms_to_hours(time_str: str) -> float:
    """Convert HH:MM:SS to hours."""
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600
