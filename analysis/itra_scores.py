"""Sandbox to derive Race score versus time curve.
The goal is not to construct the full curve just based on course information, but to reconstruct it based on some reference time.
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from race_planner.models.tools import hms_to_seconds, hours_to_hms  # noqa: E402

OUT_DIR = Path(__file__).parent / "results" / "itra_scores"

reference_races = {
    "oku_long": {
        "race_name": "Okumusashi Long",
        "times": np.array(
            [
                "16:34:08",
                "16:37:23",
                "16:55:12",
                "18:34:32",
                "19:17:39",
                "19:43:33",
                "19:48:28",
                "20:10:30",
                "20:16:51",
                "20:39:19",
                "20:43:20",
                "20:46:33",
                "21:00:14",
                "21:05:42",
                "21:18:28",
                "21:35:30",
                "21:38:40",
                "21:41:23",
                "21:51:41",
                "21:53:30",
                "21:57:07",
                "22:06:27",
                "22:13:26",
                "22:29:45",
                "22:31:17",
                "17:08:14",
                "17:31:38",
                "20:03:55",
                "20:15:36",
                "20:42:49",
                "20:44:33",
                "20:49:43",
                "20:54:40",
                "21:01:37",
                "21:01:41",
                "21:28:09",
                "21:39:11",
                "21:49:08",
                "21:52:17",
                "22:12:50",
                "22:14:08",
                "22:24:38",
            ]
        ),
        "scores": np.array(
            [
                817,
                815,
                800,
                729,
                702,
                687,
                684,
                671,
                668,
                656,
                653,
                652,
                645,
                642,
                636,
                627,
                626,
                624,
                619,
                619,
                617,
                612,
                609,
                602,
                601,
                790,
                773,
                675,
                668,
                654,
                653,
                650,
                648,
                644,
                644,
                631,
                625,
                621,
                619,
                610,
                609,
                604,
            ]
        ),
    },
    "utmb": {
        "race_name": "UTMB",
        "times": np.array(
            [
                "19:18:58",
                "19:51:37",
                "20:15:05",
                "20:40:34",
                "21:11:59",
                "21:42:54",
                "22:11:32",
                "22:43:34",
                "23:17:24",
                "23:38:06",
                "23:58:15",
                "24:10:17",
                "24:37:21",
                "24:51:32",
                "25:15:51",
                "25:33:04",
                "25:54:50",
                "26:15:21",
                "26:37:57",
                "27:22:30",
                "28:14:48",
                "29:10:07",
                "30:06:42",
                "31:04:48",
                "32:10:04",
                "33:18:51",
                "34:34:49",
                "35:53:30",
                "37:18:23",
                "38:52:42",
                "40:32:34",
                "42:24:25",
                "44:24:06",
            ]
        ),
        "scores": np.array(
            [
                966,
                940,
                922,
                903,
                880,
                859,
                841,
                821,
                801,
                790,
                779,
                772,
                758,
                751,
                739,
                730,
                720,
                711,
                701,
                682,
                661,
                640,
                620,
                600,
                580,
                560,
                540,
                520,
                500,
                480,
                460,
                440,
                420,
            ]
        ),
    },
}

# Save plot references
oku_long_fig = None
oku_long_ax = None

OUT_DIR.mkdir(parents=True, exist_ok=True)

for race, data in reference_races.items():
    times_seconds = np.array([hms_to_seconds(t) for t in data["times"]])

    # (a) Calculate A and B using two datapoints
    # Given score(t) = A - B * ln(t), we need two equations to solve for two unknowns
    # Using the first and last datapoints:
    t1, s1 = times_seconds[0], data["scores"][0]
    t2, s2 = times_seconds[-1], data["scores"][-1]

    # s1 = A - B * ln(t1)
    # s2 = A - B * ln(t2)
    # Subtracting: s1 - s2 = -B * (ln(t1) - ln(t2))
    # B = (s2 - s1) / (ln(t1) - ln(t2))
    B_single = (s2 - s1) / (np.log(t1) - np.log(t2))
    A_single = s1 + B_single * np.log(t1)

    # Calculate R² for two datapoint fit
    y_pred_single = A_single - B_single * np.log(times_seconds)
    ss_res_single = np.sum((data["scores"] - y_pred_single) ** 2)
    ss_tot_single = np.sum((data["scores"] - np.mean(data["scores"])) ** 2)
    r_squared_single = 1 - (ss_res_single / ss_tot_single)

    # (b) Calculate A and B using all datapoints via least squares regression
    # score = A - B * ln(t)
    # This is linear regression: y = A + B*x where y = score, x = -ln(t)
    X = -np.log(times_seconds)  # predictor: -ln(t)
    y = data["scores"]  # response: score

    # Using least squares: solve for [A, B] in y = A + B*x
    # Add column of ones for intercept
    X_matrix = np.column_stack([np.ones(len(X)), X])
    coeffs = np.linalg.lstsq(X_matrix, y, rcond=None)[0]
    A_full, B_full = coeffs[0], coeffs[1]

    # Calculate R² to assess fit quality
    y_pred = A_full + B_full * X
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print("\nComparison of curves calculated for two datapoints and all datapoints:")
    print(f"Reference race: {data['race_name']}")
    print(f"    R² (two datapoints) = {r_squared_single:.4f}")
    print(f"    R² (all datapoints) = {r_squared:.4f}")
    print(f"    A (two datapoints) = {A_single:.2f}, B (two datapoints) = {B_single:.2f}")
    print(f"    A (all datapoints) = {A_full:.2f}, B (all datapoints) = {B_full:.2f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot actual datapoints
    ax.scatter(
        times_seconds / 3600,
        data["scores"],
        alpha=0.6,
        s=50,
        label="Actual data",
        color="black",
        zorder=3,
    )

    # Generate smooth curves
    t_range = np.linspace(times_seconds.min(), times_seconds.max(), 500)

    # Curve from method (a) - two datapoints
    score_single = A_single - B_single * np.log(t_range)
    ax.plot(
        t_range / 3600,
        score_single,
        label=f"(a) Two datapoints: A={A_single:.1f}, B={B_single:.1f} (R²={r_squared_single:.4f})",
        linewidth=2,
        linestyle="--",
        color="blue",
    )

    # Curve from method (b) - all datapoints (least squares)
    score_full = A_full - B_full * np.log(t_range)
    ax.plot(
        t_range / 3600,
        score_full,
        label=f"(b) Least squares: A={A_full:.1f}, B={B_full:.1f} (R²={r_squared:.4f})",
        linewidth=2,
        color="red",
    )

    # Calculate target score predictions for every 1 point
    all_scores = range(1000, 399, -1)
    all_times_hours = np.exp((A_full - np.array(list(all_scores))) / B_full) / 3600

    # Calculate ratios relative to score 1000
    t_1000 = all_times_hours[0]
    ratios = {score: t_hours / t_1000 for score, t_hours in zip(all_scores, all_times_hours)}
    # Save to dictionary
    reference_races[race]["ratios_rel_to_1000"] = ratios

    # Plot only every 50 points for clarity
    plot_scores = range(1000, 350, -50)
    plot_times_hours = np.exp((A_full - np.array(list(plot_scores))) / B_full) / 3600
    ax.scatter(
        plot_times_hours,
        list(plot_scores),
        marker="x",
        s=100,
        linewidths=2.5,
        label="Target scores",
        color="green",
        zorder=4,
    )

    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("ITRA Score", fontsize=12)
    ax.set_title(
        f"ITRA Score vs Time for {data['race_name']}: Fitting score(t) = A − B·ln(t)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="major")
    ax.minorticks_on()
    ax.grid(True, alpha=0.15, which="minor", linestyle=":")

    plt.tight_layout()

    # Save reference to oku_long plot for later use
    if race == "oku_long":
        oku_long_fig = fig
        oku_long_ax = ax

    race_plot_path = OUT_DIR / f"itra_score_curve_{race}.png"
    fig.savefig(race_plot_path, dpi=170)
    print(f"Saved plot: {race_plot_path}")

    # Print time predictions table for every 50 points
    print("\n\nTime predictions for target ITRA scores (using least squares fit):")
    print(f"{'Score':<8} {'Time (h:mm:ss)':<15} {'Ratio vs 1000':<15}")
    print("-" * 40)

    for score, t_hours in zip(all_scores, all_times_hours):
        # Convert to h:mm:ss format
        total_seconds = int(t_hours * 3600)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        time_str = f"{hours}:{minutes:02d}:{seconds:02d}"

        ratio = ratios[score]
        if score % 50 == 0:
            print(f"{score:<8} {time_str:<15} {ratio:<15.3f}")

# Save ratios from UTMB to configuration file
ratios = reference_races["utmb"]["ratios_rel_to_1000"]
config_path = Path(__file__).parent.parent / "race_planner" / "models" / "itra_score_ratios.py"

ratios_export_path = OUT_DIR / "utmb_score_time_ratios.csv"
ratios_df = np.array([[s, ratios[s]] for s in sorted(ratios.keys(), reverse=True)])
np.savetxt(
    ratios_export_path,
    ratios_df,
    delimiter=",",
    header="score,ratio_vs_1000",
    comments="",
    fmt=["%d", "%.10f"],
)
print(f"Saved ratios table: {ratios_export_path}")

with open(config_path, "r") as f:
    lines = f.readlines()

# Find where to insert the ratios
with open(config_path, "w") as f:
    for line in lines:
        f.write(line)
        if line.strip() == "SCORE_TIME_RATIOS = {}":
            # Write the ratios dictionary
            f.write("SCORE_TIME_RATIOS = {\n")
            for score in sorted(ratios.keys(), reverse=True):
                f.write(f"    {score}: {ratios[score]:.10f},\n")
            f.write("}\n")
            # Skip the docstring that follows
            break
    # Write the docstring
    f.write(
        '"""dict[int, float]: Mapping from ITRA score to time ratio relative to score 1000"""\n'
    )

print(f"\nSaved {len(ratios)} score-ratio mappings to {config_path}")

# Test the predictor built from UTMB on the Oku long data
print("\n" + "=" * 60)
print("TESTING THE PREDICTOR (UTMB ratios on Oku Long data)")
print("=" * 60)

# Import the predictor
from race_planner.models.itra_predictor import ItraScorePredictor  # noqa: E402

# Test with a reference datapoint from the Okumusashi Long race
test_reference_time = reference_races["oku_long"]["times"][10]
test_reference_score = reference_races["oku_long"]["scores"][10]
test_reference_time_hours = hms_to_seconds(test_reference_time) / 3600

print(
    f"\nTest case: Given one datapoint from Oku Long ({test_reference_time}, score {test_reference_score})"
)
print("Using UTMB-derived ratios to predict times for various scores:")

target_scores = range(1000, 350, -50)
# Create predictor (with okukumsashi race reference times, but which uses UTMB reference ratios)
predictor = ItraScorePredictor(
    reference_time=test_reference_time, reference_score=test_reference_score
)

predictions = {score: predictor.predict_time(score) for score in target_scores}

print(f"\n{'Score':<8} {'Predicted Time':<15}")
print("-" * 25)
for score in target_scores:
    print(f"{score:<8} {hours_to_hms(predictions[score]):<15}")

# Add predictor results to the Okumusashi Long plot
if oku_long_fig is not None and oku_long_ax is not None:
    # Predict times for all scores in range
    predicted_scores = list(predictions.keys())
    predicted_times_hours = list(predictions.values())

    # Plot predictions on the saved oku_long axes
    oku_long_ax.plot(
        predicted_times_hours,
        predicted_scores,
        label=f"Predictor (UTMB ratios, from 1 point: {test_reference_score}@{test_reference_time})",
        linewidth=2.5,
        linestyle=":",
        color="purple",
        zorder=5,
    )

    # Highlight the reference point used
    oku_long_ax.scatter(
        [test_reference_time_hours],
        [test_reference_score],
        marker="*",
        s=500,
        color="purple",
        edgecolors="black",
        linewidths=2,
        label="Predictor reference point",
        zorder=6,
    )

    oku_long_ax.legend(fontsize=10, loc="best")
    oku_long_fig.tight_layout()

    predictor_plot_path = OUT_DIR / "itra_score_curve_oku_long_with_predictor.png"
    oku_long_fig.savefig(predictor_plot_path, dpi=170)
    print(f"Saved plot: {predictor_plot_path}")

plt.close("all")
