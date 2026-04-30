"""Generic tools library"""

import numpy as np


def hhmm_to_hours(time_str: str) -> float:
    """Convert HH:MM 24-hour string to decimal hours since midnight."""
    parts = time_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format '{time_str}'. Expected HH:MM.")
    hours = int(parts[0])
    minutes = int(parts[1])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValueError(f"Invalid time value '{time_str}'.")
    return hours + minutes / 60.0


def race_offset_to_clock_hhmm(start_h: float, race_offset_h: float) -> str:
    """Return wall-clock time in HH:MM (24h) for a given race offset from start_h."""
    total_minutes = int(round((start_h + race_offset_h) * 60.0)) % (24 * 60)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def hms_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def pace_to_seconds_per_km(pace_str: str) -> float:
    """Convert a pace string like MM:SS/km or HH:MM:SS/km to sec/km."""
    text = pace_str.strip().lower().replace('/km', '')
    parts = text.split(':')
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported pace format: {pace_str}")


def seconds_to_hms(seconds: float) -> str:
    """Convert a number of seconds to an H:MM:SS string."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def seconds_per_km_to_pace(pace_sec_per_km: float) -> str:
    """Convert sec/km to an M:SS/km pace string."""
    total = int(round(pace_sec_per_km))
    minutes = total // 60
    seconds = total % 60
    return f"{minutes}:{seconds:02d}/km"


def pace_to_speed_kmh(pace_sec_per_km: np.ndarray | float) -> np.ndarray | float:
    """Convert sec/km pace to horizontal speed in km/h."""
    return 3600.0 / pace_sec_per_km


def vertical_speed_m_per_h(
    grades: np.ndarray | float,
    pace_sec_per_km: np.ndarray | float,
) -> np.ndarray | float:
    """Convert grade and sec/km pace to signed vertical speed in m/h."""
    return pace_to_speed_kmh(pace_sec_per_km) * 1000.0 * grades


def pace_from_constant_vertical_speed(
    grades: np.ndarray | float,
    vertical_speed: float,
) -> np.ndarray | float:
    """Return sec/km pace implied by a constant signed vertical speed."""
    return 3_600_000.0 * grades / vertical_speed


def hours_to_hms(hours: float) -> str:
    """Convert hours to HH:MM:SS format."""
    return seconds_to_hms(hours * 3600)


def hms_to_hours(time_str: str) -> float:
    """Convert HH:MM:SS to hours."""
    h, m, s = map(int, time_str.split(':'))
    return h + m / 60 + s / 3600
