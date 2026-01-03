"""
GPX loader utilities.

Main class: Course - Represents a trail race course loaded from GPX file.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import gpxpy
import numpy as np
import pandas as pd


# ---- Helpers ----


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in meters between two lat/lon points using haversine."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _to_timestamp(dt) -> Optional[float]:
    if dt is None:
        return None
    if isinstance(dt, (int, float)):
        return float(dt)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _accumulate_elevation(ele_prev: float, ele_curr: float) -> Tuple[float, float]:
    """Return (gain, loss) between two elevations in meters."""
    delta = ele_curr - ele_prev
    if delta > 0:
        return delta, 0.0
    else:
        return 0.0, -delta


# ---- Core parsing ----


def _parse_with_gpxpy(path: str) -> List[Dict]:
    """Parse GPX using gpxpy and return raw point list."""
    with open(path, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                points.append(
                    {
                        "lat": float(p.latitude),
                        "lon": float(p.longitude),
                        "ele_m": (
                            float(p.elevation) if p.elevation is not None else None
                        ),
                        "time": p.time,  # datetime or None
                    }
                )
    return points


def _compute_point_stats(raw_points: List[Dict]) -> List[Dict]:
    """Compute distances, cumulative distance, grade, elevation gain/loss, time deltas."""
    out = []
    cum = 0.0
    prev = None
    cum_gain = 0.0
    cum_loss = 0.0

    for i, p in enumerate(raw_points):
        lat = p.get("lat")
        lon = p.get("lon")
        ele = p.get("ele_m")
        time = p.get("time")
        ts = _to_timestamp(time)

        if prev is None:
            dist = 0.0
            grade = 0.0
            ele_gain = 0.0
            ele_loss = 0.0
            dt = None
        else:
            dist = _haversine_m(prev["lat"], prev["lon"], lat, lon)
            cum += dist
            dt = None
            if prev.get("time") is not None and time is not None:
                dt = _to_timestamp(time) - _to_timestamp(prev["time"])
            # grade: rise over run in percent (avoid division by zero)
            if dist > 0 and ele is not None and prev.get("ele_m") is not None:
                grade = 100.0 * (ele - prev["ele_m"]) / dist
            else:
                grade = 0.0
            if ele is not None and prev.get("ele_m") is not None:
                g, l = _accumulate_elevation(prev["ele_m"], ele)
                ele_gain = g
                ele_loss = l
                cum_gain += g
                cum_loss += l
            else:
                ele_gain = 0.0
                ele_loss = 0.0

        out.append(
            {
                "lat": lat,
                "lon": lon,
                "ele_m": ele,
                "time": time,
                "timestamp": ts,
                "dist_m": dist,
                "cum_dist_m": cum,
                "grade": grade,
                "ele_gain_m": ele_gain,
                "ele_loss_m": ele_loss,
                "cum_ele_gain_m": cum_gain,
                "cum_ele_loss_m": cum_loss,
                "dt_s": dt,
            }
        )
        prev = p
    return out


# ---- Resampling by distance ----


def _resample_by_distance(points: List[Dict], interval_m: float) -> List[Dict]:
    """
    Resample points to fixed distance interval using linear interpolation on lat/lon/ele/time.
    """
    if interval_m <= 0:
        return points

    cum = [p["cum_dist_m"] for p in points]
    total = cum[-1] if cum else 0.0
    if total == 0.0:
        return points

    targets = [i * interval_m for i in range(int(math.floor(total / interval_m)) + 1)]
    if targets[-1] < total:
        targets.append(total)

    # helper to interpolate between two points
    def interp(p0, p1, frac):
        lat = p0["lat"] + (p1["lat"] - p0["lat"]) * frac
        lon = p0["lon"] + (p1["lon"] - p0["lon"]) * frac
        ele0 = p0["ele_m"] if p0["ele_m"] is not None else 0.0
        ele1 = p1["ele_m"] if p1["ele_m"] is not None else ele0
        ele = ele0 + (ele1 - ele0) * frac
        t0 = _to_timestamp(p0["time"])
        t1 = _to_timestamp(p1["time"])
        time = None
        if t0 is not None and t1 is not None:
            ts = t0 + (t1 - t0) * frac
            time = datetime.fromtimestamp(ts, tz=timezone.utc)
        return {"lat": lat, "lon": lon, "ele_m": ele, "time": time}

    resampled = []
    j = 0
    for tgt in targets:
        # find segment where cum[j] <= tgt <= cum[j+1]
        while j + 1 < len(points) and points[j + 1]["cum_dist_m"] < tgt:
            j += 1
        if j + 1 >= len(points):
            resampled.append(
                {
                    "lat": points[-1]["lat"],
                    "lon": points[-1]["lon"],
                    "ele_m": points[-1]["ele_m"],
                    "time": points[-1]["time"],
                }
            )
            continue
        p0 = points[j]
        p1 = points[j + 1]
        seg_start = p0["cum_dist_m"]
        seg_end = p1["cum_dist_m"]
        if seg_end == seg_start:
            frac = 0.0
        else:
            frac = (tgt - seg_start) / (seg_end - seg_start)
        resampled.append(interp(p0, p1, frac))

    # compute stats for resampled points
    enriched = _compute_point_stats(resampled)
    return enriched


# ---- Public API ----


class Course:
    """
    Represents a trail race course loaded from a GPX file.

    The course data is stored as a pandas DataFrame with computed statistics
    including distances, elevations, grades, and cumulative metrics.

    Args:
        path: Path to GPX file.
        resample_m: If provided, resample to fixed spacing in meters (e.g., 10).
        smooth_elevation: Placeholder for future smoothing (not implemented).

    Attributes:
        path: Path to the source GPX file.
        df: DataFrame containing all course points and computed statistics.

    Example:
        >>> course = Course("race.gpx", resample_m=10)
        >>> print(f"Total distance: {course.total_distance_km:.1f} km")
        >>> print(f"Elevation gain: {course.total_elevation_gain_m:.0f} m")
        >>> segment = course.get_segment(start_km=10, end_km=20)
    """

    def __init__(
        self,
        path: Union[str, Path],
        resample_m: Optional[float] = None,
        smooth_elevation: bool = False,
    ):
        """Initialize Course by loading and processing GPX file."""
        self.path = Path(path)
        self._resample_m = resample_m

        # Load and process GPX
        raw = _parse_with_gpxpy(str(self.path))
        if not raw:
            raise ValueError(f"No points found in GPX file: {self.path}")

        enriched = _compute_point_stats(raw)

        if resample_m is not None and resample_m > 0:
            enriched = _resample_by_distance(enriched, float(resample_m))

        # Convert to DataFrame
        self.df = pd.DataFrame(enriched)
        # Ensure column order
        cols = [
            "lat",
            "lon",
            "ele_m",
            "time",
            "timestamp",
            "dist_m",
            "cum_dist_m",
            "grade",
            "ele_gain_m",
            "ele_loss_m",
            "cum_ele_gain_m",
            "cum_ele_loss_m",
            "dt_s",
        ]
        self.df = self.df[[c for c in cols if c in self.df.columns]]

    # ---- Properties ----

    @property
    def total_distance_m(self) -> float:
        """Total course distance in meters."""
        return float(self.df["cum_dist_m"].iloc[-1])

    @property
    def total_distance_km(self) -> float:
        """Total course distance in kilometers."""
        return self.total_distance_m / 1000.0

    @property
    def total_elevation_gain_m(self) -> float:
        """Total elevation gain in meters."""
        return float(self.df["cum_ele_gain_m"].iloc[-1])

    @property
    def total_elevation_loss_m(self) -> float:
        """Total elevation loss in meters."""
        return float(self.df["cum_ele_loss_m"].iloc[-1])

    @property
    def min_elevation_m(self) -> float:
        """Minimum elevation in meters."""
        return float(self.df["ele_m"].min())

    @property
    def max_elevation_m(self) -> float:
        """Maximum elevation in meters."""
        return float(self.df["ele_m"].max())

    @property
    def num_points(self) -> int:
        """Number of GPS points in the course."""
        return len(self.df)

    # ---- Query methods ----

    def get_point_at_distance(self, distance_m: float) -> pd.Series:
        """
        Get the course point nearest to a specific distance.

        Args:
            distance_m: Target distance in meters from start.

        Returns:
            Series containing the point data.
        """
        idx = self.find_index_by_distance(distance_m)
        return self.df.iloc[idx]

    def get_segment(
        self,
        start_m: Optional[float] = None,
        end_m: Optional[float] = None,
        start_km: Optional[float] = None,
        end_km: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Extract a segment of the course by distance.

        Args:
            start_m: Start distance in meters (or use start_km).
            end_m: End distance in meters (or use end_km).
            start_km: Start distance in kilometers.
            end_km: End distance in kilometers.

        Returns:
            DataFrame containing points within the specified range.

        Example:
            >>> segment = course.get_segment(start_km=10, end_km=20)
        """
        # Convert km to m if provided
        if start_km is not None:
            start_m = start_km * 1000.0
        if end_km is not None:
            end_m = end_km * 1000.0

        # Default to full course
        if start_m is None:
            start_m = 0.0
        if end_m is None:
            end_m = self.total_distance_m

        mask = (self.df["cum_dist_m"] >= start_m) & (self.df["cum_dist_m"] <= end_m)
        return self.df[mask].copy()

    def find_index_by_distance(self, target_distance_m: float) -> int:
        """
        Find the index of the point nearest to a target distance.

        Args:
            target_distance_m: Target distance in meters.

        Returns:
            Index of the nearest point.
        """
        arr = self.df["cum_dist_m"].values
        idx = int(np.abs(arr - float(target_distance_m)).argmin())
        return idx

    def get_elevation_at_distance(self, distance_m: float) -> float:
        """
        Get elevation at a specific distance along the course.

        Args:
            distance_m: Distance in meters from start.

        Returns:
            Elevation in meters.
        """
        point = self.get_point_at_distance(distance_m)
        return float(point["ele_m"])

    def get_grade_at_distance(self, distance_m: float) -> float:
        """
        Get grade (slope) at a specific distance along the course.

        Args:
            distance_m: Distance in meters from start.

        Returns:
            Grade as percentage (positive = uphill, negative = downhill).
        """
        point = self.get_point_at_distance(distance_m)
        return float(point["grade"])

    def __repr__(self) -> str:
        """String representation of Course."""
        return (
            f"Course('{self.path.name}', "
            f"{self.total_distance_km:.1f} km, "
            f"+{self.total_elevation_gain_m:.0f}m/-{self.total_elevation_loss_m:.0f}m, "
            f"{self.num_points} points)"
        )

    def __len__(self) -> int:
        """Return number of points in course."""
        return self.num_points
