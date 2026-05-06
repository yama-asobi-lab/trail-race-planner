"""Course module for loading and working with GPX trail race courses."""

from race_planner.course.course import Course
from race_planner.course.segment_analyzer import SegmentAnalyzer, analyze_course

__all__ = ["Course", "SegmentAnalyzer", "analyze_course"]
