"""Test for ITRA score predictor"""

import pytest
import numpy as np
from loguru import logger
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path

from race_planner.models.itra_predictor import ItraScorePredictor
from race_planner.models.tools import hours_to_hms, hms_to_hours
from race_planner.models.itra_score_ratios import SCORE_TIME_RATIOS


@pytest.fixture
def reference_race():
    """Fixture for reference race data (ITJ 2024)"""
    return {
        "race_name": "ITJ 2024",
        "times": np.array(
            [
                "6:00:56",
                "6:02:09",
                "6:06:55",
                "6:19:52",
                "6:35:33",
                "6:39:43",
                "6:48:18",
                "6:56:42",
                "7:03:09",
                "7:09:07",
                "7:14:00",
                "7:20:25",
                "7:32:33",
                "7:45:22",
                "8:01:12",
                "8:14:40",
                "8:24:30",
                "8:38:59",
                "8:59:42",
                "9:19:11",
                "9:40:21",
                "10:04:0",
                "10:27:4",
                "10:54:3",
                "11:24:2",
                "11:56:08",
            ]
        ),
        "scores": np.array(
            [
                853,
                850,
                839,
                811,
                778,
                770,
                754,
                739,
                728,
                717,
                709,
                699,
                680,
                662,
                640,
                622,
                610,
                593,
                570,
                550,
                530,
                510,
                490,
                470,
                450,
                430,
            ]
        ),
    }


class TestItraScorePredictorComprehensive:
    """Comprehensive tests for ItraScorePredictor using all reference datapoints."""

    def test_predict_time_accuracy_from_all_datapoints(self, reference_race):
        """
        Test time prediction accuracy by using each datapoint as reference
        and predicting times for all other scores.
        """
        times = reference_race["times"]
        scores = reference_race["scores"]
        max_error_percent = 10.0

        errors = []
        error_data = []  # Store detailed error information for plotting

        # Use each datapoint as a reference
        for ref_idx in range(len(times)):
            ref_time = times[ref_idx]
            ref_score = scores[ref_idx]

            # Create predictor from this reference point
            predictor = ItraScorePredictor(ref_time, ref_score)

            # Predict times for all other scores
            for target_idx in range(len(times)):
                target_score = scores[target_idx]
                actual_time_hours = hms_to_hours(times[target_idx])

                # Predict time for this score
                predicted_time_hours = predictor.predict_time(target_score)

                # Calculate error percentage
                error_percent = (
                    abs(predicted_time_hours - actual_time_hours)
                    / actual_time_hours
                    * 100
                )
                errors.append(error_percent)

                # Store detailed error data
                error_data.append(
                    {
                        'ref_score': ref_score,
                        'target_score': target_score,
                        'actual_time': actual_time_hours,
                        'predicted_time': predicted_time_hours,
                        'error_percent': error_percent,
                        'score_diff': abs(target_score - ref_score),
                    }
                )

                # Assert error is within tolerance
                assert error_percent <= max_error_percent, (
                    f"Time prediction error too high: "
                    f"Reference: {ref_time} (score {ref_score}), "
                    f"Target score: {target_score}, "
                    f"Actual time: {times[target_idx]}, "
                    f"Predicted: {hours_to_hms(predicted_time_hours)}, "
                    f"Error: {error_percent:.2f}%"
                )

        # Print statistics
        logger.info(f"\nTime prediction error statistics:")
        logger.info(f"  Mean error: {np.mean(errors):.3f}%")
        logger.info(f"  Max error: {np.max(errors):.3f}%")
        logger.info(f"  Std dev: {np.std(errors):.3f}%")

        # Create visualization
        self._plot_time_prediction_errors(error_data, reference_race)

    def test_predict_score_accuracy_from_all_datapoints(self, reference_race):
        """
        Test score prediction accuracy by using each datapoint as reference
        and predicting scores for all other times.
        """
        times = reference_race["times"]
        scores = reference_race["scores"]
        max_error_percent = 10.0

        errors = []
        error_data = []  # Store detailed error information for plotting

        # Use each datapoint as a reference
        for ref_idx in range(len(times)):
            ref_time = times[ref_idx]
            ref_score = scores[ref_idx]

            # Create predictor from this reference point
            predictor = ItraScorePredictor(ref_time, ref_score)

            # Predict scores for all other times
            for target_idx in range(len(times)):
                target_time = times[target_idx]
                actual_score = scores[target_idx]

                # Predict score for this time
                predicted_score = predictor.predict_score(target_time)

                # Calculate error percentage
                error_percent = abs(predicted_score - actual_score) / actual_score * 100
                errors.append(error_percent)

                # Store detailed error data
                error_data.append(
                    {
                        'ref_score': ref_score,
                        'target_score': actual_score,
                        'predicted_score': predicted_score,
                        'error_percent': error_percent,
                        'score_diff': abs(actual_score - ref_score),
                    }
                )

                # Assert error is within tolerance
                assert error_percent <= max_error_percent, (
                    f"Score prediction error too high: "
                    f"Reference: {ref_time} (score {ref_score}), "
                    f"Target time: {target_time}, "
                    f"Actual score: {actual_score}, "
                    f"Predicted score: {predicted_score}, "
                    f"Error: {error_percent:.2f}%"
                )

        # Print statistics
        logger.info(f"\nScore prediction error statistics:")
        logger.info(f"  Mean error: {np.mean(errors):.3f}%")
        logger.info(f"  Max error: {np.max(errors):.3f}%")
        logger.info(f"  Std dev: {np.std(errors):.3f}%")

        # Create visualization
        self._plot_score_prediction_errors(error_data, reference_race)

    def _plot_time_prediction_errors(self, error_data, reference_race):
        """Create comprehensive visualization of time prediction errors."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Time Prediction Error Analysis - {reference_race["race_name"]}',
            fontsize=14,
            fontweight='bold',
        )

        # Convert to arrays for easier manipulation
        target_scores = np.array([d['target_score'] for d in error_data])
        errors = np.array([d['error_percent'] for d in error_data])
        score_diffs = np.array([d['score_diff'] for d in error_data])
        ref_scores = np.array([d['ref_score'] for d in error_data])

        # Plot 1: Error vs Target Score
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            target_scores, errors, alpha=0.3, s=20, c=score_diffs, cmap='viridis'
        )
        ax1.set_xlabel('Target Score', fontsize=11)
        ax1.set_ylabel('Prediction Error (%)', fontsize=11)
        ax1.set_title('Error vs Target Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='2% threshold')
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Score Difference')

        # Plot 2: Error vs Score Difference (from reference)
        ax2 = axes[0, 1]
        ax2.scatter(score_diffs, errors, alpha=0.3, s=20)
        ax2.set_xlabel('|Target Score - Reference Score|', fontsize=11)
        ax2.set_ylabel('Prediction Error (%)', fontsize=11)
        ax2.set_title('Error vs Distance from Reference', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='2% threshold')
        ax2.legend()

        # Plot 3: Error distribution by score bins
        ax3 = axes[1, 0]
        score_bins = [400, 500, 600, 700, 800, 900, 1000]
        binned_errors = []
        bin_labels = []
        for i in range(len(score_bins) - 1):
            mask = (target_scores >= score_bins[i]) & (
                target_scores < score_bins[i + 1]
            )
            if mask.sum() > 0:
                binned_errors.append(errors[mask])
                bin_labels.append(f'{score_bins[i]}-{score_bins[i+1]}')

        bp = ax3.boxplot(binned_errors, tick_labels=bin_labels)
        ax3.set_xlabel('Score Range', fontsize=11)
        ax3.set_ylabel('Prediction Error (%)', fontsize=11)
        ax3.set_title('Error Distribution by Score Range', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)

        # Plot 4: Error heatmap (Reference Score vs Target Score)
        ax4 = axes[1, 1]
        unique_ref_scores = sorted(set(ref_scores))
        unique_target_scores = sorted(set(target_scores))

        # Create heatmap matrix
        heatmap_data = np.full(
            (len(unique_ref_scores), len(unique_target_scores)), np.nan
        )
        for i, ref_score in enumerate(unique_ref_scores):
            for j, target_score in enumerate(unique_target_scores):
                mask = (ref_scores == ref_score) & (target_scores == target_score)
                if mask.sum() > 0:
                    heatmap_data[i, j] = errors[mask].mean()

        im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=10)
        ax4.set_xticks(
            np.arange(
                0, len(unique_target_scores), max(1, len(unique_target_scores) // 5)
            )
        )
        ax4.set_xticklabels(
            [
                unique_target_scores[i]
                for i in range(
                    0, len(unique_target_scores), max(1, len(unique_target_scores) // 5)
                )
            ]
        )
        ax4.set_yticks(
            np.arange(0, len(unique_ref_scores), max(1, len(unique_ref_scores) // 5))
        )
        ax4.set_yticklabels(
            [
                unique_ref_scores[i]
                for i in range(
                    0, len(unique_ref_scores), max(1, len(unique_ref_scores) // 5)
                )
            ]
        )
        ax4.set_xlabel('Target Score', fontsize=11)
        ax4.set_ylabel('Reference Score', fontsize=11)
        ax4.set_title('Mean Error Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Error (%)')

        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "itra_predictor_time_error_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved time error analysis plot to {output_path}")
        plt.close()

    def _plot_score_prediction_errors(self, error_data, reference_race):
        """Create comprehensive visualization of score prediction errors."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Score Prediction Error Analysis - {reference_race["race_name"]}',
            fontsize=14,
            fontweight='bold',
        )

        # Convert to arrays for easier manipulation
        target_scores = np.array([d['target_score'] for d in error_data])
        predicted_scores = np.array([d['predicted_score'] for d in error_data])
        errors = np.array([d['error_percent'] for d in error_data])
        score_diffs = np.array([d['score_diff'] for d in error_data])
        ref_scores = np.array([d['ref_score'] for d in error_data])

        # Plot 1: Predicted vs Actual Score
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            target_scores, predicted_scores, alpha=0.3, s=20, c=errors, cmap='viridis'
        )
        min_score, max_score = min(target_scores), max(target_scores)
        ax1.plot(
            [min_score, max_score],
            [min_score, max_score],
            'r--',
            alpha=0.5,
            label='Perfect prediction',
        )
        ax1.set_xlabel('Actual Score', fontsize=11)
        ax1.set_ylabel('Predicted Score', fontsize=11)
        ax1.set_title('Predicted vs Actual Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label='Error (%)')

        # Plot 2: Error vs Target Score
        ax2 = axes[0, 1]
        ax2.scatter(target_scores, errors, alpha=0.3, s=20)
        ax2.set_xlabel('Target Score', fontsize=11)
        ax2.set_ylabel('Prediction Error (%)', fontsize=11)
        ax2.set_title('Error vs Target Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='2% threshold')
        ax2.legend()

        # Plot 3: Error vs Score Difference
        ax3 = axes[1, 0]
        ax3.scatter(score_diffs, errors, alpha=0.3, s=20)
        ax3.set_xlabel('|Target Score - Reference Score|', fontsize=11)
        ax3.set_ylabel('Prediction Error (%)', fontsize=11)
        ax3.set_title('Error vs Distance from Reference', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='2% threshold')
        ax3.legend()

        # Plot 4: Error distribution by score bins
        ax4 = axes[1, 1]
        score_bins = [400, 500, 600, 700, 800, 900, 1000]
        binned_errors = []
        bin_labels = []
        for i in range(len(score_bins) - 1):
            mask = (target_scores >= score_bins[i]) & (
                target_scores < score_bins[i + 1]
            )
            if mask.sum() > 0:
                binned_errors.append(errors[mask])
                bin_labels.append(f'{score_bins[i]}-{score_bins[i+1]}')

        bp = ax4.boxplot(binned_errors, tick_labels=bin_labels)
        ax4.set_xlabel('Score Range', fontsize=11)
        ax4.set_ylabel('Prediction Error (%)', fontsize=11)
        ax4.set_title('Error Distribution by Score Range', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=2, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Save plot
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "itra_predictor_score_error_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved score error analysis plot to {output_path}")
        plt.close()

    def test_time_score_roundtrip_consistency(self, reference_race):
        """
        Test that predicting time from score, then score from that time
        returns approximately the original score.
        """
        times = reference_race["times"]
        scores = reference_race["scores"]

        # Use middle datapoint as reference
        ref_idx = len(times) // 2
        ref_time = times[ref_idx]
        ref_score = scores[ref_idx]

        predictor = ItraScorePredictor(ref_time, ref_score)

        for target_score in scores:
            # Predict time from score
            predicted_time = predictor.predict_time(target_score)

            # Predict score from that time
            roundtrip_score = predictor.predict_score(predicted_time)

            # Should be very close to original score
            error = abs(roundtrip_score - target_score)
            assert error <= 2, (
                f"Roundtrip error too high for score {target_score}: "
                f"got {roundtrip_score} (error: {error})"
            )


class TestItraScorePredictorBasic:
    """Basic functionality tests for ItraScorePredictor."""

    def test_initialization_with_string_time(self, reference_race):
        """Test initialization with time as string."""
        predictor = ItraScorePredictor("7:00:00", 700)
        assert predictor.reference_time_hours == 7.0
        assert predictor.reference_score == 700

    def test_initialization_with_float_time(self, reference_race):
        """Test initialization with time as float."""
        predictor = ItraScorePredictor(7.5, 700)
        assert predictor.reference_time_hours == 7.5
        assert predictor.reference_score == 700

    def test_invalid_score_raises_error(self):
        """Test that invalid reference score raises ValueError."""
        invalid_score = 350  # Below the minimum valid score
        with pytest.raises(ValueError, match="not in available ratios"):
            ItraScorePredictor("7:00:00", invalid_score)

    def test_predict_time_returns_float(self, reference_race):
        """Test that predict_time returns a float."""
        predictor = ItraScorePredictor("7:00:00", 700)
        result = predictor.predict_time(600)
        assert isinstance(result, float)
        assert result > 0

    def test_predict_time_formatted_returns_string(self, reference_race):
        """Test that predict_time_formatted returns a string in HH:MM:SS format."""
        predictor = ItraScorePredictor("7:00:00", 700)
        result = predictor.predict_time_formatted(600)
        assert isinstance(result, str)
        assert result.count(':') == 2

    def test_predict_score_with_string_time(self, reference_race):
        """Test predict_score with time as string."""
        predictor = ItraScorePredictor("7:00:00", 700)
        result = predictor.predict_score("8:00:00")
        assert isinstance(result, int)
        assert 400 <= result <= 1000

    def test_predict_score_with_float_time(self, reference_race):
        """Test predict_score with time as float."""
        predictor = ItraScorePredictor(7.0, 700)
        result = predictor.predict_score(8.0)
        assert isinstance(result, int)
        assert 400 <= result <= 1000

    def test_higher_score_means_faster_time(self, reference_race):
        """Test that higher scores correspond to faster times."""
        predictor = ItraScorePredictor("7:00:00", 700)

        time_900 = predictor.predict_time(900)
        time_800 = predictor.predict_time(800)
        time_700 = predictor.predict_time(700)
        time_600 = predictor.predict_time(600)

        assert time_900 < time_800 < time_700 < time_600

    def test_score_out_of_range_raises_error(self, reference_race):
        """Test that score outside valid range raises ValueError."""
        predictor = ItraScorePredictor("7:00:00", 700)

        with pytest.raises(ValueError, match="outside the valid range"):
            predictor.predict_time(350)

        with pytest.raises(ValueError, match="outside the valid range"):
            predictor.predict_time(1001)


class TestItraScorePredictorEdgeCases:
    """Edge case tests for ItraScorePredictor."""

    def test_reference_score_prediction_returns_reference_time(self, reference_race):
        """Test that predicting the reference score returns the reference time."""
        ref_time = "7:00:00"
        ref_score = 700

        predictor = ItraScorePredictor(ref_time, ref_score)
        predicted_time = predictor.predict_time(ref_score)

        # Should be very close (within rounding errors)
        assert abs(predicted_time - 7.0) < 0.001

    def test_interpolation_between_scores(self, reference_race):
        """Test that interpolation works for scores not in the ratio table."""
        predictor = ItraScorePredictor("7:00:00", 700)

        # These scores might not be exact keys in SCORE_TIME_RATIOS
        score_1 = 705
        score_2 = 695

        time_1 = predictor.predict_time(score_1)
        time_2 = predictor.predict_time(score_2)

        # Higher score should have faster time
        assert time_1 < time_2

    def test_extreme_scores(self, reference_race):
        """Test predictions at the extremes of the valid range."""
        predictor = ItraScorePredictor("7:00:00", 700)

        min_score = min(SCORE_TIME_RATIOS.keys())
        max_score = max(SCORE_TIME_RATIOS.keys())

        # Should not raise errors
        time_min = predictor.predict_time(min_score)
        time_max = predictor.predict_time(max_score)

        # Extreme scores should give extreme times
        assert time_min > time_max
        assert time_max > 0

    def test_consistency_across_different_reference_points(self, reference_race):
        """
        Test that predictions are consistent regardless of which
        reference point is used (within tolerance).
        """
        times = reference_race["times"]
        scores = reference_race["scores"]

        # Use first and last datapoints as references
        predictor_first = ItraScorePredictor(times[0], scores[0])
        predictor_last = ItraScorePredictor(times[-1], scores[-1])

        # Target score in the middle
        target_score = scores[len(scores) // 2]

        time_from_first = predictor_first.predict_time(target_score)
        time_from_last = predictor_last.predict_time(target_score)

        # Should be very similar (within 5%)
        error_percent = abs(time_from_first - time_from_last) / time_from_first * 100
        assert error_percent < 5.0, (
            f"Predictions from different references too different: "
            f"{error_percent:.2f}% error"
        )


class TestItraScorePredictorRatioIntegrity:
    """Tests for the integrity of the ratio data itself."""

    def test_ratio_data_exists(self):
        """Test that SCORE_TIME_RATIOS is populated."""
        assert len(SCORE_TIME_RATIOS) > 0
        assert len(SCORE_TIME_RATIOS) >= 100  # Should have many scores

    def test_ratio_data_sorted_correctly(self):
        """Test that higher scores have lower ratios (faster times)."""
        scores = sorted(SCORE_TIME_RATIOS.keys())

        for i in range(len(scores) - 1):
            score_low = scores[i]
            score_high = scores[i + 1]

            # Higher score should have smaller ratio (faster relative time)
            assert SCORE_TIME_RATIOS[score_high] < SCORE_TIME_RATIOS[score_low], (
                f"Ratio ordering incorrect: score {score_high} has ratio "
                f"{SCORE_TIME_RATIOS[score_high]}, but score {score_low} has ratio "
                f"{SCORE_TIME_RATIOS[score_low]}"
            )

    def test_score_1000_has_ratio_1(self):
        """Test that score 1000 has ratio exactly 1.0."""
        assert 1000 in SCORE_TIME_RATIOS
        assert abs(SCORE_TIME_RATIOS[1000] - 1.0) < 0.0001

    def test_ratios_are_positive(self):
        """Test that all ratios are positive."""
        for score, ratio in SCORE_TIME_RATIOS.items():
            assert ratio > 0, f"Score {score} has non-positive ratio {ratio}"

    def test_ratios_are_reasonable(self):
        """Test that ratios are within reasonable bounds."""
        for score, ratio in SCORE_TIME_RATIOS.items():
            # Ratios should be between 0.5 and 3.0 for reasonable scores
            assert 0.5 < ratio < 3.0, f"Score {score} has unreasonable ratio {ratio}"
