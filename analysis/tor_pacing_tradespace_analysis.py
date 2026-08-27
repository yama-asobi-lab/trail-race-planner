import os
import re
import subprocess
import tempfile
from typing import List, Tuple
import pandas as pd
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from race_planner.models.tools import seconds_per_km_to_pace


BASE_CONFIG_PATH = "config/races/tor330.yaml"
OUT_DIR = Path(__file__).parent / "results" / "analyze_tor330_pacing_strategies"

# Define sleep strategies:
SLEEP_STRATEGIES = {
    "no_sleep": [],
    "strategy_1": [
        ("Rhemes-Notre-Dame", 1080),
        ("Rifugio Dondena", 1080),
        ("Rifugio Della Barma", 5100),
        ("GRESSONEY", 1080),
        ("Rifugio Lo Magià", 5100),
        ("OLLOMONT", 1080),
        ("Bosses", 1080),
    ],
    "strategy_2": [
        ("Rifugio Dondena", 1200),
        ("Rifugio Della Barma", 1200),
        ("GRESSONEY", 10800),
        ("OLLOMONT", 5100),
    ],
    "strategy_3": [
        ("Rifugio Dondena", 1200),
        ("Rifugio Della Barma", 1200),
        ("GRESSONEY", 5100),
        ("OLLOMONT", 5100),
    ],
}

PACE_RANGE = (310, 480)  # in seconds/km
PACE_STEP_SIZE = 15  # in seconds/km


def generate_paces(start_s: int, end_s: int, step_s: int) -> List[int]:
    """Generate starting paces in seconds/km."""
    return list(range(start_s, end_s + 1, step_s))


def create_temp_config(base_path: str, sleep_plan: List[Tuple[str, int]]) -> str:
    """Modifies the base YAML in-memory with stop durations (sleep + 600s)
    and writes out a temporary file.
    """
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    # Normalize lookup keys: strip markdown '*' and force lowercase
    sleep_map = {name.replace("*", "").strip().lower(): sleep_s for name, sleep_s in sleep_plan}

    # config["aid_stations"] is a list of dicts
    if "aid_stations" in config and isinstance(config["aid_stations"], list):
        for station in config["aid_stations"]:
            raw_name = station.get("name", "")
            clean_name = raw_name.replace("*", "").strip().lower()

            if clean_name in sleep_map:
                sleep_s = sleep_map[clean_name]
                station["stop_time_s"] = sleep_s + 600
                station["sleep_duration_s"] = sleep_s

    # Write out temporary YAML
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, temp_file)
    temp_file.close()
    return temp_file.name


def plot_tradespace(df: pd.DataFrame):
    """Generates a line plot of total race time vs starting pace per strategy."""
    df_clean = df.dropna(subset=["Final_Time_h"]).copy()
    if df_clean.empty:
        print("[WARNING] No valid race times captured to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot a curve for each strategy
    for strat_key, group in df_clean.groupby("Strategy_ID"):
        group = group.sort_values("Pace_s")
        sleep_hours = group["Total_Sleep_h"].iloc[0]
        label = f"{strat_key} ({sleep_hours:.1f}h sleep)"

        plt.plot(
            group["Pace_s"],
            group["Final_Time_h"],
            marker="o",
            linewidth=2,
            label=label,
        )

    # Format X-axis with human-readable paces (e.g., 5:10/km)
    unique_paces = sorted(df_clean["Pace_s"].unique())
    pace_labels = [seconds_per_km_to_pace(p) for p in unique_paces]
    plt.xticks(ticks=unique_paces, labels=pace_labels)

    plt.xlabel("Starting Grade-Adjusted Pace (min/km)", fontsize=11)
    plt.ylabel("Total Race Time (hours)", fontsize=11)
    plt.title("TOR330 Pacing & Sleep Strategy Tradespace", fontsize=13, pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Sleep Strategy", fontsize=10)
    plt.tight_layout()

    output_plot_path = OUT_DIR / "tor330_strategy_tradespace_plot.png"
    plt.savefig(output_plot_path, dpi=300)
    print(f"\n[INFO] Tradespace plot saved to {output_plot_path}")
    plt.show()


def run_cli_tradespace():
    paces = generate_paces(PACE_RANGE[0], PACE_RANGE[1], PACE_STEP_SIZE)
    results = []

    for strategy_key, sleep_plan in SLEEP_STRATEGIES.items():
        # Create temp config for this specific sleep strategy
        temp_config_path = create_temp_config(BASE_CONFIG_PATH, sleep_plan)

        try:
            for pace_s in paces:
                pace_str = seconds_per_km_to_pace(pace_s)

                cmd = [
                    sys.executable,
                    "-m",
                    "race_planner.main",
                    temp_config_path,
                    "--athlete",
                    "carlos",
                    "--mode",
                    "grade_adjusted_pace",
                    "--target-grade-adjusted-pace",
                    pace_str,
                    "--fatigue-mode",
                    "sigmoid",
                ]

                try:
                    # Run process and capture stdout
                    response = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"\n[ERROR] Command failed for strategy '{strategy_key}' at pace '{pace_str}':"
                    )
                    print(f"Command: {' '.join(cmd)}")  # Printable string for easy debugging
                    print("--- Subprocess Stderr ---")
                    print(e.stderr)
                    print("-------------------------")
                    raise e

                # Parse race time from CLI stdout (Adjust regex to match main.py output pattern)
                # Strip ANSI terminal escape sequences (colors, bold text, etc.)
                clean_stdout = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', response.stdout)

                # Flexible regex: handles 2-digit to 3-digit hour counts (e.g., 102:45:12)
                match = re.search(
                    r"finish time:\s*(\d+):(\d{2}):(\d{2})", clean_stdout, re.IGNORECASE
                )

                if match:
                    hours, minutes, seconds = map(int, match.groups())
                    final_time_h = round(hours + (minutes / 60.0) + (seconds / 3600.0), 2)
                else:
                    final_time_h = None

                results.append(
                    {
                        "Strategy_ID": strategy_key,
                        "Pace_s": pace_s,
                        "Start_GAP": pace_str,
                        "Total_Sleep_h": sum(s for _, s in sleep_plan) / 3600.0,
                        "Final_Time_h": final_time_h,
                        "Sleep_Plan": str(sleep_plan),
                    }
                )

        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    df = pd.DataFrame(results)

    print("\n--- Tradespace Results ---")
    print(
        df[
            [
                "Strategy_ID",
                "Start_GAP",
                "Total_Sleep_h",
                "Final_Time_h",
            ]
        ].to_string(index=False)
    )

    # Plot total time vs starting pace
    plot_tradespace(df)


if __name__ == "__main__":
    run_cli_tradespace()
