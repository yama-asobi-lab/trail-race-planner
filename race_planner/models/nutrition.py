"""Nutrition models for race fueling planning."""

from math import floor, log
from typing import Any

import numpy as np

from race_planner.models.tools import (
    canonical_point_name,
    extract_volume_ml,
    hms_to_seconds,
    seconds_to_hms,
)


CAFFEINE_DEFAULT_ABSORPTION_LAG_H = 0.5
CAFFEINE_DEFAULT_HALF_LIFE_H = 5.5
HYDRATION_DEFAULT_MAX_BODY_WEIGHT_LOSS_PCT = 1.5
HYDRATION_WATER_FLASK_ML = 500.0


def caffeine_concentration_mg_per_kg(
    time_h: np.ndarray,
    ingestion_plan: list[tuple[float, float]],
    weight_kg: float,
    absorption_lag_h: float,
    half_life_h: float,
) -> np.ndarray:
    """Return caffeine concentration curve in mg/kg over race time.

    Models first-order absorption (step at absorption_lag_h after each dose)
    and first-order elimination (exponential decay with the given half-life).

    Args:
        time_h: Array of race-time points in hours.
        ingestion_plan: List of (dose_time_h, dose_mg) tuples.
        weight_kg: Athlete body weight in kg.
        absorption_lag_h: Time after ingestion at which full dose enters circulation.
        half_life_h: Caffeine plasma half-life in hours.

    Returns:
        Array of caffeine concentration values in mg/kg, same shape as time_h.
    """
    elimination_k = log(2.0) / half_life_h
    concentration = np.zeros_like(time_h, dtype=float)

    for dose_time_h, dose_mg in ingestion_plan:
        absorbed_time_h = dose_time_h + absorption_lag_h
        dt = time_h - absorbed_time_h
        active = dt >= 0.0
        concentration[active] += (dose_mg / weight_kg) * np.exp(-elimination_k * dt[active])

    return concentration


def load_food_catalog(catalog_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalize a food catalog config dictionary.

    Expected shape:
        foods:
          - name: "Food Name"
            reference_size: "500 ml"   # optional
            carbs_g: 25                 # required (or carbs_g_per_unit)
            protein_g: 0                # optional
            fat_g: 0                    # optional
            sodium_mg: 600              # optional
    """
    foods = catalog_config.get("foods", [])
    if not isinstance(foods, list) or not foods:
        raise ValueError("Food catalog must define a non-empty 'foods' list")

    catalog: dict[str, dict[str, Any]] = {}
    for index, food in enumerate(foods):
        name = str(food.get("name", "")).strip()
        if not name:
            raise ValueError(f"foods[{index}].name is required")
        if name in catalog:
            raise ValueError(f"Duplicate food name in catalog: {name}")

        carbs = food.get("carbs_g_per_unit", food.get("carbs_g"))
        if carbs is None:
            raise ValueError(f"foods[{index}] ({name}) must define carbs_g or carbs_g_per_unit")
        carbs_g_per_unit = float(carbs)
        if carbs_g_per_unit <= 0:
            raise ValueError(f"foods[{index}] ({name}) carbs must be > 0")

        reference_size = str(food.get("reference_size", "1 unit")).strip() or "1 unit"
        fluid_ml_per_unit = food.get("fluid_ml_per_unit")
        if fluid_ml_per_unit is None:
            fluid_ml_per_unit = extract_volume_ml(reference_size)

        accepted_fractions_cfg = food.get("accepted_fractions")
        accepted_fractions: list[float] | None = None
        if accepted_fractions_cfg is not None:
            if not isinstance(accepted_fractions_cfg, list) or not accepted_fractions_cfg:
                raise ValueError(
                    f"foods[{index}] ({name}) accepted_fractions must be a non-empty list"
                )
            parsed = sorted({float(value) for value in accepted_fractions_cfg})
            if any(value <= 0 or value > 1 for value in parsed):
                raise ValueError(
                    f"foods[{index}] ({name}) accepted_fractions values must be in (0, 1]"
                )
            if 1.0 not in parsed:
                parsed.append(1.0)
            accepted_fractions = sorted(set(parsed))

        catalog[name] = {
            "name": name,
            "reference_size": reference_size,
            "unit_label": str(food.get("unit_label", "serving")).strip() or "serving",
            "carbs_g_per_unit": carbs_g_per_unit,
            "caffeine_mg_per_unit": float(food.get("caffeine_mg", 0.0) or 0.0),
            "protein_g_per_unit": float(food.get("protein_g", 0.0) or 0.0),
            "fat_g_per_unit": float(food.get("fat_g", 0.0) or 0.0),
            "sodium_mg_per_unit": float(food.get("sodium_mg", 0.0) or 0.0),
            "fluid_ml_per_unit": float(fluid_ml_per_unit or 0.0),
            "notes": str(food.get("notes", "")).strip(),
            "accepted_fractions": accepted_fractions,
        }

    return catalog


def _normalize_food_choices(raw_choices: list[Any] | None) -> list[dict[str, Any]]:
    """Normalize food choices to [{'food': str, 'ratio': float}, ...]."""
    if not raw_choices:
        return []

    normalized: list[dict[str, Any]] = []
    for index, choice in enumerate(raw_choices):
        if isinstance(choice, str):
            food_name = choice.strip()
            ratio = 1.0
        elif isinstance(choice, dict):
            food_name = str(choice.get("food", choice.get("name", ""))).strip()
            ratio = float(choice.get("ratio", 1.0))
        else:
            raise ValueError(f"Unsupported food choice type at index {index}: {type(choice)}")

        if not food_name:
            raise ValueError(f"Food choice at index {index} is missing a name")
        if ratio <= 0:
            raise ValueError(f"Food choice ratio for '{food_name}' must be > 0")

        normalized.append({"food": food_name, "ratio": ratio})

    return normalized


def _choices_for_point(
    point_name: str,
    choices_cfg: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Resolve default/by_segment choices for a checkpoint."""
    if not choices_cfg:
        return []

    by_segment = choices_cfg.get("by_segment", {})
    canonical = canonical_point_name(point_name)

    raw = None
    if isinstance(by_segment, dict):
        raw = by_segment.get(point_name)
        if raw is None:
            raw = by_segment.get(canonical)
    if raw is None:
        raw = choices_cfg.get("default", [])

    return _normalize_food_choices(raw)


def _manual_intake_for_point(
    point_name: str,
    intake_cfg: dict[str, Any] | None,
    food_catalog: dict[str, dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Resolve explicit intake (food + units) for a checkpoint.

    Returns:
        (is_configured, allocations)
        is_configured is True when the point (or default) has an explicit intake
        entry, even if the entry is an empty list.
    """
    if not intake_cfg:
        return False, []

    by_segment = intake_cfg.get("by_segment", {})
    canonical = canonical_point_name(point_name)

    raw_entries = None
    is_configured = False
    if isinstance(by_segment, dict):
        if point_name in by_segment:
            raw_entries = by_segment[point_name]
            is_configured = True
        elif canonical in by_segment:
            raw_entries = by_segment[canonical]
            is_configured = True

    if not is_configured and "default" in intake_cfg:
        raw_entries = intake_cfg.get("default")
        is_configured = True

    if not is_configured:
        return False, []

    if raw_entries is None:
        return True, []
    if not isinstance(raw_entries, list):
        raise ValueError("Manual aid intake entries must be a list")

    allocations: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manual aid intake entry at index {index} must be a mapping")

        food_name = str(entry.get("food", entry.get("name", ""))).strip()
        if not food_name:
            raise ValueError(f"Manual aid intake entry at index {index} is missing 'food'")

        food = food_catalog.get(food_name)
        has_units = "units" in entry
        has_carbs = "carbs_g" in entry
        if not has_units and not has_carbs:
            raise ValueError(
                f"Manual aid intake for '{food_name}' must define either units or carbs_g"
            )

        units = float(entry.get("units", 0.0)) if has_units else 0.0
        if units < 0:
            raise ValueError(f"Manual aid intake units for '{food_name}' must be >= 0")

        if has_carbs:
            actual_carbs_g = float(entry.get("carbs_g", 0.0))
            if actual_carbs_g < 0:
                raise ValueError(f"Manual aid intake carbs_g for '{food_name}' must be >= 0")
            if not has_units and food is not None:
                units = actual_carbs_g / food["carbs_g_per_unit"]
            if not has_units and food is None:
                units = 1.0
        else:
            if food is None:
                raise ValueError(
                    f"Unknown food in manual aid intake: '{food_name}'. "
                    "Use carbs_g for non-catalog foods."
                )
            actual_carbs_g = units * food["carbs_g_per_unit"]

        if "caffeine_mg" in entry:
            actual_caffeine_mg = float(entry.get("caffeine_mg", 0.0))
        elif food is not None:
            actual_caffeine_mg = units * float(food.get("caffeine_mg_per_unit", 0.0))
        else:
            actual_caffeine_mg = 0.0

        reference_size = food["reference_size"] if food is not None else "custom"
        unit_label = food["unit_label"] if food is not None else "serving"
        allocations.append(
            {
                "food": food_name,
                "ratio": 1.0,
                "target_carbs_g": actual_carbs_g,
                "actual_carbs_g": actual_carbs_g,
                "actual_caffeine_mg": actual_caffeine_mg,
                "raw_units": units,
                "units": units,
                "reference_size": reference_size,
                "unit_label": unit_label,
            }
        )

    return True, allocations


def _quantize_units(units: float, accepted_fractions: list[float] | None) -> float:
    """Round units to allowed fractional servings when configured."""
    if units <= 0:
        return 0.0
    if not accepted_fractions:
        return units

    base = int(floor(units))
    fracs = [0.0] + [f for f in accepted_fractions if 0 < f < 1]
    candidates = {max(0.0, n + f) for n in range(max(0, base - 1), base + 2) for f in fracs} | {
        float(n) for n in range(max(0, base - 1), base + 3)
    }

    return min(candidates, key=lambda c: (abs(c - units), -c))


def _allocate_carbs_by_ratio(
    target_carbs_g: float,
    choices: list[dict[str, Any]],
    food_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allocate target carbs across selected foods according to their ratios."""
    if target_carbs_g <= 0 or not choices:
        return []

    total_ratio = sum(choice["ratio"] for choice in choices)
    if total_ratio <= 0:
        raise ValueError("Food choices must have a positive total ratio")

    allocations: list[dict[str, Any]] = []
    for choice in choices:
        food_name = choice["food"]
        if food_name not in food_catalog:
            raise ValueError(f"Unknown food in nutrition plan: '{food_name}'")

        food = food_catalog[food_name]
        food_target_carbs_g = target_carbs_g * (choice["ratio"] / total_ratio)
        raw_units = food_target_carbs_g / food["carbs_g_per_unit"]
        units = _quantize_units(raw_units, food.get("accepted_fractions"))
        actual_carbs_g = units * food["carbs_g_per_unit"]
        actual_caffeine_mg = units * float(food.get("caffeine_mg_per_unit", 0.0))
        allocations.append(
            {
                "food": food_name,
                "ratio": choice["ratio"],
                "target_carbs_g": food_target_carbs_g,
                "actual_carbs_g": actual_carbs_g,
                "actual_caffeine_mg": actual_caffeine_mg,
                "raw_units": raw_units,
                "units": units,
                "reference_size": food["reference_size"],
                "unit_label": food["unit_label"],
            }
        )

    return allocations


def _quantize_supplemental_water_ml(target_ml: float) -> float:
    """Quantize supplemental carried water to integer 500 ml flasks."""
    if target_ml <= 0:
        return 0.0
    return floor(target_ml / HYDRATION_WATER_FLASK_ML) * HYDRATION_WATER_FLASK_ML


def build_race_nutrition_plan(
    pacing_rows: list[dict[str, Any]],
    target_carbs_g_per_h: float,
    food_catalog: dict[str, dict[str, Any]],
    segment_foods_cfg: dict[str, Any] | None,
    aid_station_intake_cfg: dict[str, Any] | None = None,
    caffeine_ingestion_plan: list[tuple[float, float]] | None = None,
    caffeine_weight_kg: float | None = None,
    caffeine_absorption_lag_h: float = CAFFEINE_DEFAULT_ABSORPTION_LAG_H,
    caffeine_half_life_h: float = CAFFEINE_DEFAULT_HALF_LIFE_H,
    sweat_rate_ml_per_h: float | None = None,
    hydration_weight_kg: float | None = None,
    max_body_weight_loss_pct: float = HYDRATION_DEFAULT_MAX_BODY_WEIGHT_LOSS_PCT,
) -> dict[str, Any]:
    """Build a race nutrition plan from segment/aid food choices.

    The user specifies *what* foods are allowed in each segment and explicit
    aid-station intake (food + units). This function computes *how much* of each
    segment food is needed to satisfy carb targets with carry-over compensation.
    """
    target_carbs_g_per_h = float(target_carbs_g_per_h)
    if target_carbs_g_per_h <= 0:
        raise ValueError("target_carbs_g_per_h must be > 0")

    hydration_enabled = sweat_rate_ml_per_h is not None and float(sweat_rate_ml_per_h) > 0
    if hydration_enabled:
        sweat_rate_ml_per_h = float(sweat_rate_ml_per_h)
        if hydration_weight_kg is None or float(hydration_weight_kg) <= 0:
            raise ValueError(
                "Hydration plan is configured but hydration_weight_kg is missing or invalid"
            )
        hydration_weight_kg = float(hydration_weight_kg)
        max_body_weight_loss_pct = float(max_body_weight_loss_pct)
        if max_body_weight_loss_pct < 0:
            raise ValueError("max_body_weight_loss_pct must be >= 0")
        max_hydration_deficit_ml = hydration_weight_kg * 1000.0 * (max_body_weight_loss_pct / 100.0)
    else:
        sweat_rate_ml_per_h = 0.0
        hydration_weight_kg = float(hydration_weight_kg or 0.0)
        max_body_weight_loss_pct = float(max_body_weight_loss_pct)
        max_hydration_deficit_ml = 0.0

    rows: list[dict[str, Any]] = []
    totals_by_food: dict[str, dict[str, float]] = {}
    prev_elapsed_s = 0
    cumulative_target_g = 0.0
    cumulative_actual_g = 0.0
    carry_carbs_g = 0.0
    cumulative_hydration_balance_ml = 0.0
    hydration_total_planned_fluids_ml = 0.0
    hydration_total_supplemental_fluids_ml = 0.0
    hydration_total_sweat_loss_ml = 0.0

    for index, pacing_row in enumerate(pacing_rows):
        point_name = str(pacing_row.get("Point Name", f"Point {index}"))
        elapsed_hms = str(pacing_row.get("Elapsed Time", "0:00:00"))
        elapsed_s = hms_to_seconds(elapsed_hms)
        segment_start_s = prev_elapsed_s
        segment_s = max(0, elapsed_s - prev_elapsed_s)
        prev_elapsed_s = elapsed_s

        segment_hours = segment_s / 3600.0
        moving_segment_target_g = segment_hours * target_carbs_g_per_h
        desired_row_carbs_g = max(0.0, moving_segment_target_g + carry_carbs_g)

        has_manual_aid, manual_aid_allocations = _manual_intake_for_point(
            point_name,
            aid_station_intake_cfg,
            food_catalog,
        )

        aid_allocations = manual_aid_allocations if has_manual_aid else []
        aid_target_g = sum(a["actual_carbs_g"] for a in aid_allocations)
        segment_target_g = max(0.0, desired_row_carbs_g - aid_target_g)

        segment_choices = _choices_for_point(point_name, segment_foods_cfg)
        segment_allocations = _allocate_carbs_by_ratio(
            segment_target_g,
            segment_choices,
            food_catalog,
        )
        all_allocations = segment_allocations + aid_allocations

        row_actual_carbs_g = sum(a["actual_carbs_g"] for a in all_allocations)
        row_caffeine_intake_mg = sum(a.get("actual_caffeine_mg", 0.0) for a in all_allocations)
        row_planned_fluids_ml = sum(
            float(a.get("units", 0.0))
            * float(food_catalog[a["food"]].get("fluid_ml_per_unit", 0.0))
            for a in all_allocations
            if a.get("food") in food_catalog
        )

        row_sweat_loss_ml = segment_hours * sweat_rate_ml_per_h
        provisional_hydration_balance_ml = (
            cumulative_hydration_balance_ml + row_planned_fluids_ml - row_sweat_loss_ml
        )
        row_supplemental_fluids_ml = 0.0
        if hydration_enabled:
            if provisional_hydration_balance_ml < 0:
                row_supplemental_fluids_ml = _quantize_supplemental_water_ml(
                    -provisional_hydration_balance_ml
                )
                provisional_hydration_balance_ml += row_supplemental_fluids_ml

        cumulative_hydration_balance_ml = provisional_hydration_balance_ml
        if hydration_enabled and cumulative_hydration_balance_ml > 1e-9:
            raise ValueError(
                "Hydration plan would over-drink (positive sweat imbalance) "
                f"at point '{point_name}'"
            )
        if hydration_enabled and cumulative_hydration_balance_ml < (
            -max_hydration_deficit_ml - 1e-9
        ):
            raise ValueError(
                "Hydration plan exceeds maximum allowed sweat-loss imbalance "
                f"at point '{point_name}': {abs(cumulative_hydration_balance_ml):.1f} ml "
                f"> {max_hydration_deficit_ml:.1f} ml"
            )

        row_total_fluids_ml = row_planned_fluids_ml + row_supplemental_fluids_ml
        hydration_total_planned_fluids_ml += row_planned_fluids_ml
        hydration_total_supplemental_fluids_ml += row_supplemental_fluids_ml
        hydration_total_sweat_loss_ml += row_sweat_loss_ml

        row_carbs_per_h = row_actual_carbs_g / segment_hours if segment_hours > 0 else 0.0

        total_row_target_g = moving_segment_target_g
        cumulative_target_g += total_row_target_g
        cumulative_actual_g += row_actual_carbs_g
        carry_carbs_g = cumulative_target_g - cumulative_actual_g

        for allocation in all_allocations:
            food_name = allocation["food"]
            if food_name not in totals_by_food:
                totals_by_food[food_name] = {"units": 0.0, "carbs_g": 0.0}
            totals_by_food[food_name]["units"] += allocation["units"]
            totals_by_food[food_name]["carbs_g"] += allocation["actual_carbs_g"]

        rows.append(
            {
                "point_name": point_name,
                "elapsed_hms": elapsed_hms,
                "segment_time_hms": seconds_to_hms(segment_s),
                "segment_start_h": segment_start_s / 3600.0,
                "segment_end_h": elapsed_s / 3600.0,
                "segment_target_carbs_g": segment_target_g,
                "aid_target_carbs_g": aid_target_g,
                "row_target_carbs_g": total_row_target_g,
                "row_actual_carbs_g": row_actual_carbs_g,
                "row_carbs_per_h": row_carbs_per_h,
                "row_caffeine_intake_mg": row_caffeine_intake_mg,
                "row_sweat_loss_ml": row_sweat_loss_ml,
                "row_planned_fluids_ml": row_planned_fluids_ml,
                "row_supplemental_fluids_ml": row_supplemental_fluids_ml,
                "row_total_fluids_ml": row_total_fluids_ml,
                "cumulative_hydration_balance_ml": cumulative_hydration_balance_ml,
                "cumulative_hydration_balance_pct_bw": (
                    (cumulative_hydration_balance_ml / (hydration_weight_kg * 1000.0)) * 100.0
                    if hydration_weight_kg > 0
                    else 0.0
                ),
                "cumulative_target_carbs_g": cumulative_target_g,
                "cumulative_actual_carbs_g": cumulative_actual_g,
                "carry_carbs_g": carry_carbs_g,
                "segment_allocations": segment_allocations,
                "aid_allocations": aid_allocations,
                "all_allocations": all_allocations,
            }
        )

    checkpoint_time_h = np.array([hms_to_seconds(row["elapsed_hms"]) / 3600.0 for row in rows])
    if caffeine_ingestion_plan and caffeine_weight_kg and caffeine_weight_kg > 0:
        concentration = caffeine_concentration_mg_per_kg(
            time_h=checkpoint_time_h,
            ingestion_plan=caffeine_ingestion_plan,
            weight_kg=float(caffeine_weight_kg),
            absorption_lag_h=float(caffeine_absorption_lag_h),
            half_life_h=float(caffeine_half_life_h),
        )
        dose_by_time: dict[float, float] = {}
        for dose_time_h, dose_mg in caffeine_ingestion_plan:
            dose_by_time[float(dose_time_h)] = dose_by_time.get(float(dose_time_h), 0.0) + float(
                dose_mg
            )

        for row, t_h, c_mgkg in zip(rows, checkpoint_time_h, concentration):
            segment_events = [
                {"time_h": float(dose_time_h), "dose_mg": float(dose_mg)}
                for dose_time_h, dose_mg in caffeine_ingestion_plan
                if row["segment_start_h"] < float(dose_time_h) <= row["segment_end_h"]
            ]
            segment_events.sort(key=lambda event: event["time_h"])
            row["segment_caffeine_events"] = segment_events
            row["caffeine_dose_mg_at_point"] = dose_by_time.get(float(t_h), 0.0)
            row["caffeine_concentration_mg_per_kg"] = float(c_mgkg)

        caffeine_totals = {
            "total_dose_mg": float(sum(dose for _, dose in caffeine_ingestion_plan)),
            "peak_concentration_mg_per_kg": (
                float(np.max(concentration)) if len(concentration) else 0.0
            ),
            "mean_concentration_mg_per_kg": (
                float(np.mean(concentration)) if len(concentration) else 0.0
            ),
        }
    else:
        for row in rows:
            row["segment_caffeine_events"] = []
            row["caffeine_dose_mg_at_point"] = 0.0
            row["caffeine_concentration_mg_per_kg"] = 0.0
        caffeine_totals = {
            "total_dose_mg": 0.0,
            "peak_concentration_mg_per_kg": 0.0,
            "mean_concentration_mg_per_kg": 0.0,
        }

    total_time_h = prev_elapsed_s / 3600.0
    return {
        "rows": rows,
        "food_catalog": food_catalog,
        "totals": {
            "target_carbs_g_per_h": target_carbs_g_per_h,
            "total_time_h": total_time_h,
            "moving_target_carbs_g": total_time_h * target_carbs_g_per_h,
            "planned_total_carbs_g": cumulative_actual_g,
            "by_food": totals_by_food,
            "caffeine": caffeine_totals,
            "hydration": {
                "enabled": hydration_enabled,
                "sweat_rate_ml_per_h": sweat_rate_ml_per_h,
                "max_body_weight_loss_pct": max_body_weight_loss_pct,
                "max_allowed_deficit_ml": max_hydration_deficit_ml,
                "planned_fluids_from_food_ml": hydration_total_planned_fluids_ml,
                "planned_supplemental_fluids_ml": hydration_total_supplemental_fluids_ml,
                "planned_total_fluids_ml": hydration_total_planned_fluids_ml
                + hydration_total_supplemental_fluids_ml,
                "estimated_total_sweat_loss_ml": hydration_total_sweat_loss_ml,
                "final_sweat_imbalance_ml": cumulative_hydration_balance_ml,
                "final_sweat_imbalance_pct_bw": (
                    (cumulative_hydration_balance_ml / (hydration_weight_kg * 1000.0)) * 100.0
                    if hydration_weight_kg > 0
                    else 0.0
                ),
            },
        },
    }
