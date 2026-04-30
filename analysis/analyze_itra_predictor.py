"""
Diagnostic analysis of ItraScorePredictor accuracy.

Uses the ITJ 2024 leaderboard as a reference dataset.  For every pair of
datapoints (reference → target) the predictor is evaluated in both
directions (time prediction and score prediction).  The resulting error
distributions are saved as PNG figures under analysis/results/analyze_itra_predictor/.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from race_planner.models.itra_predictor import ItraScorePredictor
from race_planner.models.tools import hms_to_hours, hours_to_hms

# ---------------------------------------------------------------------------
# ITJ 2024 reference data
# ---------------------------------------------------------------------------

ITJ_2024_TIMES = np.array(
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
)

ITJ_2024_SCORES = np.array(
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
)

RACE_NAME = "ITJ 2024"
OUTPUT_DIR = Path(__file__).parent / "results" / "analyze_itra_predictor"


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def _compute_time_prediction_errors(times, scores):
    """Return list of per-pair dicts for time prediction cross-validation."""
    error_data = []
    for ref_idx in range(len(times)):
        predictor = ItraScorePredictor(times[ref_idx], scores[ref_idx])
        for target_idx in range(len(times)):
            actual_time_hours = hms_to_hours(times[target_idx])
            predicted_time_hours = predictor.predict_time(scores[target_idx])
            error_percent = abs(predicted_time_hours - actual_time_hours) / actual_time_hours * 100
            error_data.append(
                {
                    'ref_score': scores[ref_idx],
                    'target_score': scores[target_idx],
                    'actual_time': actual_time_hours,
                    'predicted_time': predicted_time_hours,
                    'error_percent': error_percent,
                    'score_diff': abs(scores[target_idx] - scores[ref_idx]),
                }
            )
    return error_data


def _compute_score_prediction_errors(times, scores):
    """Return list of per-pair dicts for score prediction cross-validation."""
    error_data = []
    for ref_idx in range(len(times)):
        predictor = ItraScorePredictor(times[ref_idx], scores[ref_idx])
        for target_idx in range(len(times)):
            actual_score = scores[target_idx]
            predicted_score = predictor.predict_score(times[target_idx])
            error_percent = abs(predicted_score - actual_score) / actual_score * 100
            error_data.append(
                {
                    'ref_score': scores[ref_idx],
                    'target_score': actual_score,
                    'predicted_score': predicted_score,
                    'error_percent': error_percent,
                    'score_diff': abs(actual_score - scores[ref_idx]),
                }
            )
    return error_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_time_prediction_errors(error_data, race_name, output_dir):
    """Save a 4-panel figure summarising time prediction errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Time Prediction Error Analysis - {race_name}',
        fontsize=14,
        fontweight='bold',
    )

    target_scores = np.array([d['target_score'] for d in error_data])
    errors = np.array([d['error_percent'] for d in error_data])
    score_diffs = np.array([d['score_diff'] for d in error_data])
    ref_scores = np.array([d['ref_score'] for d in error_data])

    # Plot 1: Error vs Target Score
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(target_scores, errors, alpha=0.3, s=20, c=score_diffs, cmap='viridis')
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
        mask = (target_scores >= score_bins[i]) & (target_scores < score_bins[i + 1])
        if mask.sum() > 0:
            binned_errors.append(errors[mask])
            bin_labels.append(f'{score_bins[i]}-{score_bins[i+1]}')
    ax3.boxplot(binned_errors, tick_labels=bin_labels)
    ax3.set_xlabel('Score Range', fontsize=11)
    ax3.set_ylabel('Prediction Error (%)', fontsize=11)
    ax3.set_title('Error Distribution by Score Range', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)

    # Plot 4: Error heatmap (Reference Score vs Target Score)
    ax4 = axes[1, 1]
    unique_ref_scores = sorted(set(ref_scores))
    unique_target_scores = sorted(set(target_scores))
    heatmap_data = np.full((len(unique_ref_scores), len(unique_target_scores)), np.nan)
    for i, ref_score in enumerate(unique_ref_scores):
        for j, target_score in enumerate(unique_target_scores):
            mask = (ref_scores == ref_score) & (target_scores == target_score)
            if mask.sum() > 0:
                heatmap_data[i, j] = errors[mask].mean()
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=10)
    step = max(1, len(unique_target_scores) // 5)
    ax4.set_xticks(np.arange(0, len(unique_target_scores), step))
    ax4.set_xticklabels(
        [unique_target_scores[i] for i in range(0, len(unique_target_scores), step)]
    )
    step = max(1, len(unique_ref_scores) // 5)
    ax4.set_yticks(np.arange(0, len(unique_ref_scores), step))
    ax4.set_yticklabels([unique_ref_scores[i] for i in range(0, len(unique_ref_scores), step)])
    ax4.set_xlabel('Target Score', fontsize=11)
    ax4.set_ylabel('Reference Score', fontsize=11)
    ax4.set_title('Mean Error Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Error (%)')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "itra_predictor_time_error_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved time error analysis plot to {output_path}")
    plt.close()


def _plot_score_prediction_errors(error_data, race_name, output_dir):
    """Save a 4-panel figure summarising score prediction errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Score Prediction Error Analysis - {race_name}',
        fontsize=14,
        fontweight='bold',
    )

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
        [min_score, max_score], [min_score, max_score], 'r--', alpha=0.5, label='Perfect prediction'
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
        mask = (target_scores >= score_bins[i]) & (target_scores < score_bins[i + 1])
        if mask.sum() > 0:
            binned_errors.append(errors[mask])
            bin_labels.append(f'{score_bins[i]}-{score_bins[i+1]}')
    ax4.boxplot(binned_errors, tick_labels=bin_labels)
    ax4.set_xlabel('Score Range', fontsize=11)
    ax4.set_ylabel('Prediction Error (%)', fontsize=11)
    ax4.set_title('Error Distribution by Score Range', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=2, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "itra_predictor_score_error_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved score error analysis plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    logger.info("Computing time prediction cross-validation errors...")
    time_error_data = _compute_time_prediction_errors(ITJ_2024_TIMES, ITJ_2024_SCORES)
    errors = np.array([d['error_percent'] for d in time_error_data])
    logger.info(
        f"  Mean error: {np.mean(errors):.3f}%  Max: {np.max(errors):.3f}%  Std: {np.std(errors):.3f}%"
    )
    _plot_time_prediction_errors(time_error_data, RACE_NAME, OUTPUT_DIR)

    logger.info("Computing score prediction cross-validation errors...")
    score_error_data = _compute_score_prediction_errors(ITJ_2024_TIMES, ITJ_2024_SCORES)
    errors = np.array([d['error_percent'] for d in score_error_data])
    logger.info(
        f"  Mean error: {np.mean(errors):.3f}%  Max: {np.max(errors):.3f}%  Std: {np.std(errors):.3f}%"
    )
    _plot_score_prediction_errors(score_error_data, RACE_NAME, OUTPUT_DIR)


if __name__ == "__main__":
    main()
