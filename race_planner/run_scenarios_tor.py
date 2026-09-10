'''
TOR330 Race Strategy Scenario Runner.

This module builds and manages named variations of the race configuration map.
It allows running individual pacing/sleep scenarios against the race planner
without duplicating physical YAML files.

Usage:
    python run_scenarios.py [--scenario SCENARIO_NAME] [--target-grade-adjusted-pace PACE]
'''

import argparse
import copy
import os
import sys
import tempfile
import subprocess
import yaml
from typing import List, Tuple

BASE_CONFIG_PATH = "config/races/tor330.yaml"


def apply_sleep_plan(
    base_config: dict,
    sleep_plan: List[Tuple[str, int]],
    stop_plan: List[Tuple[str, int]] | None = None,
) -> dict:
    """Helper to apply a sleep plan to the base configuration dictionary."""
    sleep_map = {name.replace("*", "").strip().lower(): sleep_s for name, sleep_s in sleep_plan}
    if stop_plan:
        stop_map = {name.replace("*", "").strip().lower(): stop_s for name, stop_s in stop_plan}
    else:
        stop_map = {}

    if "aid_stations" in base_config and isinstance(base_config["aid_stations"], list):
        for station in base_config["aid_stations"]:
            clean_name = station.get("name", "").replace("*", "").strip().lower()
            if clean_name in sleep_map:
                sleep_s = sleep_map[clean_name]
                station["stop_time_s"] = sleep_s + stop_map.get(clean_name, 600)
                station["sleep_duration_s"] = sleep_s
    return base_config


# Load the baseline configuration
with open(BASE_CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

# ==========================
# Create Variations
# ==========================

# No sleep strategy
no_sleep_config = copy.deepcopy(base_config)
no_sleep_config = apply_sleep_plan(no_sleep_config, [])
no_sleep_config["race"]["name"] = "TOR330 - No sleep"

late_90_min_sleep_config = copy.deepcopy(base_config)
late_90_min_sleep_config = apply_sleep_plan(
    late_90_min_sleep_config,
    [
        ("GRESSONEY", 5100),
    ],
)
late_90_min_sleep_config["race"]["name"] = "TOR330 - 90-min sleep Gressoney"

valtournenche_90_min_sleep_config = copy.deepcopy(base_config)
valtournenche_90_min_sleep_config = apply_sleep_plan(
    valtournenche_90_min_sleep_config,
    [
        ("VALTOURNENCHE", 5100),
    ],
)
valtournenche_90_min_sleep_config["race"]["name"] = "TOR330 - 90-min sleep Valtournenche"

ollomont_90_min_sleep_config = copy.deepcopy(base_config)
ollomont_90_min_sleep_config = apply_sleep_plan(
    ollomont_90_min_sleep_config,
    [
        ("OLLOMONT", 5100),
    ],
)
ollomont_90_min_sleep_config["race"]["name"] = "TOR330 - 90-min sleep Ollomont"

# late_180_min_sleep_config = copy.deepcopy(base_config)
# late_180_min_sleep_config = apply_sleep_plan(
#     late_180_min_sleep_config,
#     [
#         ("GRESSONEY", 10200),
#     ],
# )
# late_180_min_sleep_config["race"]["name"] = "TOR330 - 180-min sleep Gressoney"

twice_90_min_sleep_config = copy.deepcopy(base_config)
twice_90_min_sleep_config = apply_sleep_plan(
    twice_90_min_sleep_config,
    [
        ("Rifugio Della Barma", 5100),
        ("Rifugio Lo Magia", 5100),
    ],
)
twice_90_min_sleep_config["race"]["name"] = "TOR330 - 2x90-min sleep Rifugio Della Barma & Lo Magià"


# Strategy 1 (Balanced)
strat_1 = copy.deepcopy(base_config)
strat_1 = apply_sleep_plan(
    strat_1,
    [
        ("Rifugio Dondena", 1080),
        ("Rifugio Della Barma", 5100),
        ("GRESSONEY", 1080),
        ("Rifugio Lo Magia", 5100),
        ("OLLOMONT", 1080),
        ("Bosses", 1080),
    ],
)
strat_1["race"]["name"] = "TOR330 - Strategy 1 (Balanced)"
# strat_1["race"]["planning"]["fatigue_parameters"]["start_threshold_fraction"] = 0.75

strat_2 = copy.deepcopy(base_config)
strat_2 = apply_sleep_plan(
    strat_2,
    [
        ("Rifugio Dondena", 1080),
        ("Rifugio Della Barma", 1080),
        ("GRESSONEY", 5100),
        ("Rifugio Lo Magia", 1080),
        ("Oyace", 5100),
        ("Bosses", 1080),
    ],
)
strat_2["race"]["name"] = "TOR330 - Strategy 2 (Sleep Low)"

strat_3 = copy.deepcopy(base_config)
strat_3 = apply_sleep_plan(
    strat_3,
    [
        ("Rifugio Dondena", 1080),
        ("Niel - Dortoir La Gruba", 5100),
        ("Rifugio Lo Magia", 1080),
        ("Oyace", 5100),
        ("Bosses", 1080),
    ],
)
strat_3["race"]["name"] = "TOR330 - Strategy 3 (delayed sleep)"

strat_4 = copy.deepcopy(base_config)
strat_4 = apply_sleep_plan(
    strat_4,
    sleep_plan=[
        ("Rifugio Dondena", 1080),  # 18 minutes
        ("Rifugio Della Barma", 1080),
        ("Niel - Dortoir La Gruba", 1080),
        ("Champluc", 1080),
        ("Rifugio Lo Magia", 5100),
        ("Oyace", 1080),
        ("Bosses", 1080),
    ],
    stop_plan=[
        ("Rifugio Dondena", 420),  # 7 minutes
        ("Rifugio Della Barma", 420),
        ("Niel - Dortoir La Gruba", 420),
        ("Champluc", 420),
        ("Rifugio Lo Magia", 720),  # 12 minutes
        ("Oyace", 420),
        ("Bosses", 420),
    ],
)
strat_4["race"]["name"] = "TOR330 - Strategy 4 (short sleeps, more frequent)"

# Create configuration map
race_config_map = {
    "baseline": copy.deepcopy(base_config),
    "no_sleep": no_sleep_config,
    "strategy_1": strat_1,
    "strategy_2": strat_2,
    "strategy_3": strat_3,
    "strategy_4": strat_4,
    "late_90_min_sleep": late_90_min_sleep_config,
    "valtournenche_90_min_sleep": valtournenche_90_min_sleep_config,
    "ollomont_90_min_sleep": ollomont_90_min_sleep_config,
    # "late_180_min_sleep": late_180_min_sleep_config,
    "twice_90_min_sleep": twice_90_min_sleep_config,
}

# ==========================


def main():
    parser = argparse.ArgumentParser(description="Run named TOR330 pacing scenarios.")

    # Custom scenario argument
    parser.add_argument(
        "--scenario",
        choices=race_config_map.keys(),
        default="baseline",
        help="Which named configuration variation to run from race_config_map",
    )

    # Passthrough arguments for race_planner.main
    parser.add_argument("--athlete", default="carlos")
    parser.add_argument("--mode", default="grade_adjusted_pace")
    parser.add_argument("--target-grade-adjusted-pace", default="6:00", dest="target_gap")
    parser.add_argument("--fatigue-mode", default="sigmoid")

    args = parser.parse_args()
    selected_config = race_config_map[args.scenario]

    # Route output to folder named after the scenario
    from pathlib import Path

    orig_output_str = selected_config.get("race", {}).get("output_file")
    if orig_output_str:
        orig_path = Path(orig_output_str)
        # Inject the scenario name into the path (e.g., outputs/<scenario>/filename.xlsx)
        new_out_dir = orig_path.parent / args.scenario
        new_out_path = new_out_dir / orig_path.name

        # Ensure the directory exists so pandas/main.py doesn't crash on write
        new_out_dir.mkdir(parents=True, exist_ok=True)

        # Override the path in the in-memory dictionary
        # Use as_posix() to ensure standard forward-slashes in the YAML
        selected_config["race"]["output_file"] = new_out_path.as_posix()
    # =================================================================

    # Write the chosen variation to a temporary YAML
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(selected_config, temp_file)
    temp_file.close()

    # Write the chosen variation to a temporary YAML
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(selected_config, temp_file)
    temp_file.close()

    print(f"\n[INFO] Executing Scenario: {args.scenario}")
    print(f"[INFO] Race Name: {selected_config['race'].get('name', 'N/A')}\n")

    # Build the exact command you are used to running
    cmd = [
        sys.executable,
        "-m",
        "race_planner.main",
        temp_file.name,
        "--athlete",
        args.athlete,
        "--mode",
        args.mode,
        "--target-grade-adjusted-pace",
        args.target_gap,
        "--fatigue-mode",
        args.fatigue_mode,
    ]

    try:
        # Run standard output to console so you can see the loguru prints
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Simulation failed for scenario '{args.scenario}'")
        sys.exit(1)
    finally:
        # Always clean up the temporary config file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


if __name__ == "__main__":
    main()
