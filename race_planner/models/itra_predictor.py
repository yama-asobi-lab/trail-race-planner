"""
ITRA Score Time Predictor

Predicts race times for different ITRA scores based on a single reference datapoint.
Uses pre-calculated ratios derived from empirical race data.
"""

from typing import Tuple
import numpy as np

from race_planner.models.itra_score_ratios import SCORE_TIME_RATIOS
from race_planner.models.tools import hours_to_hms, hms_to_hours


class ItraScorePredictor:
    """
    Predicts race times for different ITRA scores using ratio interpolation.

    Given one known (time, score) datapoint from a race, this class can predict
    the time for any other score by using empirically-derived ratios.

    Args:
        reference_time: Known finish time as "HH:MM:SS" string or hours (float)
        reference_score: The ITRA score corresponding to the reference time
    """

    def __init__(self, reference_time: str | float, reference_score: int):
        """Initialize predictor with a reference datapoint."""
        if reference_score not in SCORE_TIME_RATIOS:
            raise ValueError(
                f"Reference score {reference_score} not in available ratios "
                f"(range: {min(SCORE_TIME_RATIOS.keys())}-{max(SCORE_TIME_RATIOS.keys())})"
            )
        # Convert reference time to hours if needed
        if isinstance(reference_time, str):
            self.reference_time_hours = hms_to_hours(reference_time)
        else:
            self.reference_time_hours = reference_time

        self.reference_score = reference_score
        self.reference_ratio = SCORE_TIME_RATIOS[reference_score]

        # Calculate the base time for score 1000 for this race
        self.base_time_1000 = self.reference_time_hours / self.reference_ratio

    def predict_time(self, target_score: int) -> float:
        """
        Predict the time for a target ITRA score.

        Args:
            target_score: The ITRA score to predict time for

        Returns:
            Predicted time in hours
        """
        # Get ratio for target score (with interpolation if needed)
        target_ratio = self._get_ratio(target_score)

        # Calculate predicted time
        return self.base_time_1000 * target_ratio

    def predict_time_formatted(self, target_score: int) -> str:
        """
        Predict the time for a target score and return as HH:MM:SS string.

        Args:
            target_score: The ITRA score to predict time for

        Returns:
            Predicted time in HH:MM:SS format
        """
        hours = self.predict_time(target_score)
        return hours_to_hms(hours)

    def _get_ratio(self, score: int) -> float:
        """
        Get the ratio for a score, using linear interpolation if needed.

        Args:
            score: The ITRA score

        Returns:
            Time ratio relative to score 1000
        """
        # Direct lookup if available
        if score in SCORE_TIME_RATIOS:
            return SCORE_TIME_RATIOS[score]

        # Linear interpolation
        scores = sorted(SCORE_TIME_RATIOS.keys())
        min_score = min(scores)
        max_score = max(scores)

        if score < min_score or score > max_score:
            raise ValueError(
                f"Score {score} is outside the valid range "
                f"({min_score}-{max_score})"
            )

        # Find bracketing scores
        lower_score = max(s for s in scores if s <= score)
        upper_score = min(s for s in scores if s >= score)

        if lower_score == upper_score:
            return SCORE_TIME_RATIOS[lower_score]

        # Linear interpolation
        lower_ratio = SCORE_TIME_RATIOS[lower_score]
        upper_ratio = SCORE_TIME_RATIOS[upper_score]

        fraction = (score - lower_score) / (upper_score - lower_score)
        return lower_ratio + fraction * (upper_ratio - lower_ratio)


def predict_times_from_reference(
    reference_time: str | float, reference_score: int, target_scores: list[int]
) -> dict[int, str]:
    """
    Convenience function to predict multiple target times.

    Args:
        reference_time: Known time as "HH:MM:SS" string or hours (float)
        reference_score: ITRA score for the reference time
        target_scores: List of scores to predict times for

    Returns:
        Dictionary mapping target scores to predicted times (HH:MM:SS)
    """
    # Convert reference time to hours if needed
    if isinstance(reference_time, str):
        reference_time_hours = hms_to_hours(reference_time)
    else:
        reference_time_hours = reference_time

    # Create predictor
    predictor = ItraScorePredictor(reference_time_hours, reference_score)

    # Generate predictions
    return {score: predictor.predict_time_formatted(score) for score in target_scores}
