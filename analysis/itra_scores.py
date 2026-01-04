"""Sandbox to derive Race score versus time curve.
The goal is not to construct the full curve just based on course information, but to reconstruct it based on some reference time.
"""

import numpy as np
import matplotlib.pyplot as plt

oku_long_times = np.array(
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
)
oku_long_scores = np.array(
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
)

# Solve for the equation score(t) = A − B * ln(t)


# Convert time strings to seconds
def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


oku_long_times_seconds = np.array([time_to_seconds(t) for t in oku_long_times])

# (a) Calculate A and B using two datapoints
# Given score = A - B * ln(t), we need two equations to solve for two unknowns
# Using the first and last datapoints:
t1, s1 = oku_long_times_seconds[0], oku_long_scores[0]
t2, s2 = oku_long_times_seconds[-1], oku_long_scores[-1]

# s1 = A - B * ln(t1)
# s2 = A - B * ln(t2)
# Subtracting: s1 - s2 = -B * (ln(t1) - ln(t2))
# B = (s2 - s1) / (ln(t1) - ln(t2))
B_single = (s2 - s1) / (np.log(t1) - np.log(t2))
A_single = s1 + B_single * np.log(t1)

print(f"(a) Using two datapoints:")
print(f"    A = {A_single:.2f}")
print(f"    B = {B_single:.2f}")
print(
    f"    First point: score({t1}s) = {A_single - B_single * np.log(t1):.2f} (actual: {s1})"
)
print(
    f"    Second point: score({t2}s) = {A_single - B_single * np.log(t2):.2f} (actual: {s2})"
)

# (b) Calculate A and B using all datapoints via least squares regression
# score = A - B * ln(t)
# This is linear regression: y = A + B*x where y = score, x = -ln(t)
X = -np.log(oku_long_times_seconds)  # predictor: -ln(t)
y = oku_long_scores  # response: score

# Using least squares: solve for [A, B] in y = A + B*x
# Add column of ones for intercept
X_matrix = np.column_stack([np.ones(len(X)), X])
coeffs = np.linalg.lstsq(X_matrix, y, rcond=None)[0]
A_full, B_full = coeffs[0], coeffs[1]

print(f"\n(b) Using all datapoints (least squares):")
print(f"    A = {A_full:.2f}")
print(f"    B = {B_full:.2f}")

# Calculate R² to assess fit quality
y_pred = A_full + B_full * X
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"    R² = {r_squared:.4f}")

# Show some example predictions
print(f"\n    Example predictions:")
for i in [0, len(oku_long_times_seconds) // 2, -1]:
    t = oku_long_times_seconds[i]
    actual = oku_long_scores[i]
    predicted = A_full - B_full * np.log(t)
    print(f"      score({oku_long_times[i]}) = {predicted:.2f} (actual: {actual})")

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot actual datapoints
ax.scatter(
    oku_long_times_seconds / 3600,
    oku_long_scores,
    alpha=0.6,
    s=50,
    label='Actual data',
    color='black',
    zorder=3,
)

# Generate smooth curves
t_range = np.linspace(oku_long_times_seconds.min(), oku_long_times_seconds.max(), 500)

# Curve from method (a) - two datapoints
score_single = A_single - B_single * np.log(t_range)
ax.plot(
    t_range / 3600,
    score_single,
    label=f'(a) Two datapoints: A={A_single:.1f}, B={B_single:.1f}',
    linewidth=2,
    linestyle='--',
    color='blue',
)

# Curve from method (b) - all datapoints (least squares)
score_full = A_full - B_full * np.log(t_range)
ax.plot(
    t_range / 3600,
    score_full,
    label=f'(b) Least squares: A={A_full:.1f}, B={B_full:.1f} (R²={r_squared:.4f})',
    linewidth=2,
    color='red',
)

# Calculate target score predictions for every 1 point
all_scores = range(1000, 399, -1)
all_times_hours = np.exp((A_full - np.array(list(all_scores))) / B_full) / 3600

# Plot only every 50 points for clarity
plot_scores = range(1000, 350, -50)
plot_times_hours = np.exp((A_full - np.array(list(plot_scores))) / B_full) / 3600
ax.scatter(
    plot_times_hours,
    list(plot_scores),
    marker='x',
    s=100,
    linewidths=2.5,
    label='Target scores',
    color='green',
    zorder=4,
)

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('ITRA Score', fontsize=12)
ax.set_title(
    'ITRA Score vs Time: Fitting score(t) = A − B·ln(t)', fontsize=14, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='major')
ax.minorticks_on()
ax.grid(True, alpha=0.15, which='minor', linestyle=':')

plt.tight_layout()
plt.show()

# Print time predictions table for every 1 point
print(f"\n\nTime predictions for target ITRA scores (using least squares fit):")
print(f"{'Score':<8} {'Time (h:mm:ss)':<15} {'Ratio vs 1000':<15}")
print("-" * 40)

t_1000 = all_times_hours[0]  # Reference time for score 1000

for score, t_hours in zip(all_scores, all_times_hours):
    # Convert to h:mm:ss format
    total_seconds = int(t_hours * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    time_str = f"{hours}:{minutes:02d}:{seconds:02d}"

    ratio = t_hours / t_1000
    if score % 50 == 0:
        print(f"{score:<8} {time_str:<15} {ratio:<15.3f}")

# Next step: validate that the "ratio" holds for any other race
