"""Tests for Course class."""

import pytest
from race_planner.course.course import Course


class TestCourse:
    """Tests for the Course class."""

    def test_course_initialization(self, sample_gpx_path):
        """Test that a Course can be initialized from a GPX file."""
        course = Course(sample_gpx_path)
        print(f"\n\nCourse loaded with {course.num_points} points.")
        print(f"Total course distance is: {course.total_distance_km:.2f} km")
        print(f"Total elevation gain is: {course.total_elevation_gain_m:.2f} m")
        assert course.num_points > 0
        assert course.total_distance_m > 0
        assert course.df is not None
        assert len(course.df) == course.num_points

    def test_course_with_resampling(self, sample_gpx_path):
        """Test Course with resampling enabled."""
        course = Course(sample_gpx_path, resample_m=100)
        assert course.num_points > 0
        # Check that resampling worked - distances should be ~100m apart
        dist_diffs = course.df["dist_m"].iloc[1:]
        assert 110 > dist_diffs.median() > 90  # Should be close to 100m

    def test_total_distance_properties(self, sample_gpx_path):
        """Test distance properties."""
        course = Course(sample_gpx_path)
        assert course.total_distance_km == course.total_distance_m / 1000.0
        assert course.total_distance_km > 100

    def test_elevation_properties(self, sample_gpx_path):
        """Test elevation properties."""
        course = Course(sample_gpx_path)
        assert course.total_elevation_gain_m >= 7000
        assert course.total_elevation_loss_m >= 7000
        assert course.min_elevation_m < course.max_elevation_m
        assert course.min_elevation_m == course.df["ele_m"].min()
        assert course.max_elevation_m == course.df["ele_m"].max()

    def test_get_point_at_distance(self, sample_gpx_path):
        """Test retrieving a point at a specific distance."""
        course = Course(sample_gpx_path)
        target_distance = course.total_distance_m / 2  # Midpoint
        point = course.get_point_at_distance(target_distance)
        assert point is not None
        assert "lat" in point.index
        assert "lon" in point.index
        assert "ele_m" in point.index
        # Check that we got a point close to the target
        assert (
            abs(point["cum_dist_m"] - target_distance) < course.total_distance_m * 0.1
        )

    def test_get_segment(self, sample_gpx_path):
        """Test extracting a segment of the course."""
        course = Course(sample_gpx_path)
        total_km = course.total_distance_km

        # Get middle third of course
        start = total_km / 3
        end = 2 * total_km / 3
        segment = course.get_segment(start_km=start, end_km=end)

        assert len(segment) > 0
        assert segment["cum_dist_m"].min() >= start * 1000
        assert segment["cum_dist_m"].max() <= end * 1000

    def test_get_segment_default_values(self, sample_gpx_path):
        """Test segment extraction with default start/end values."""
        course = Course(sample_gpx_path)

        # No arguments should return full course
        segment = course.get_segment()
        assert len(segment) == len(course.df)

        # Only start specified
        segment = course.get_segment(start_km=10)
        assert segment["cum_dist_m"].min() >= 10000

        # Only end specified
        segment = course.get_segment(end_km=10)
        assert segment["cum_dist_m"].max() <= 10000

    def test_get_elevation_at_distance(self, sample_gpx_path):
        """Test getting elevation at a specific distance."""
        course = Course(sample_gpx_path)
        distance = course.total_distance_m / 2
        elevation = course.get_elevation_at_distance(distance)
        assert isinstance(elevation, float)
        assert course.min_elevation_m <= elevation <= course.max_elevation_m

    def test_get_grade_at_distance(self, sample_gpx_path):
        """Test getting grade at a specific distance."""
        course = Course(sample_gpx_path)
        distance = course.total_distance_m / 2
        grade = course.get_grade_at_distance(distance)
        assert isinstance(grade, float)
        # Grade should be reasonable (between -100% and +100%)
        assert -100 <= grade <= 100

    def test_find_index_by_distance_method(self, sample_gpx_path):
        """Test the find_index_by_distance method."""
        course = Course(sample_gpx_path)
        target_distance = course.total_distance_m / 2
        idx = course.find_index_by_distance(target_distance)
        assert 0 <= idx < len(course.df)
        assert abs(course.df.iloc[idx]["cum_dist_m"] - target_distance) < 1000

    def test_course_repr(self, sample_gpx_path):
        """Test string representation of Course."""
        course = Course(sample_gpx_path)
        repr_str = repr(course)
        assert "Course" in repr_str
        assert "km" in repr_str
        assert "points" in repr_str

    def test_course_len(self, sample_gpx_path):
        """Test len() on Course object."""
        course = Course(sample_gpx_path)
        assert len(course) == course.num_points
        assert len(course) == len(course.df)

    def test_dataframe_columns(self, sample_gpx_path):
        """Test that DataFrame has expected columns."""
        course = Course(sample_gpx_path)
        expected_cols = [
            "lat",
            "lon",
            "ele_m",
            "cum_dist_m",
            "grade",
            "ele_gain_m",
            "ele_loss_m",
            "cum_ele_gain_m",
            "cum_ele_loss_m",
        ]
        for col in expected_cols:
            assert col in course.df.columns
