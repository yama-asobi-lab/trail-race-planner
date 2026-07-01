"""Tests for race_planner.models.nutrition"""

import numpy as np
import pytest

from race_planner.models.nutrition import (
    build_race_nutrition_plan,
    caffeine_concentration_mg_per_kg,
    load_food_catalog,
)

WEIGHT_KG = 70.0
HALF_LIFE_H = 5.5
ABSORPTION_LAG_H = 0.5
SINGLE_DOSE_MG = 200.0


def test_concentration_is_zero_before_absorption():
    """No caffeine in blood before the absorption lag has elapsed."""
    time_h = np.array([0.0, 0.25, 0.49])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert np.all(conc == 0.0)


def test_concentration_peaks_at_absorption_lag():
    """Concentration is highest exactly at the absorption lag (no decay yet)."""
    # Sample time points: just before, at, and after absorption
    lag = ABSORPTION_LAG_H
    time_h = np.array([lag - 1e-9, lag, lag + 1.0])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    peak = SINGLE_DOSE_MG / WEIGHT_KG
    assert conc[1] == pytest.approx(peak)
    assert conc[2] < conc[1]


def test_concentration_halves_after_one_half_life():
    """Concentration at absorption_lag + half_life should be half the peak."""
    peak_time = ABSORPTION_LAG_H
    half_life_time = peak_time + HALF_LIFE_H
    time_h = np.array([peak_time, half_life_time])
    ingestion_plan = [(0.0, SINGLE_DOSE_MG)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert conc[1] == pytest.approx(conc[0] / 2.0, rel=1e-6)


def test_concentration_superposition_of_two_doses():
    """Two simultaneous doses should produce exactly double the concentration."""
    time_h = np.linspace(0.0, 10.0, 50)
    single = caffeine_concentration_mg_per_kg(
        time_h, [(0.0, SINGLE_DOSE_MG)], WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    double = caffeine_concentration_mg_per_kg(
        time_h,
        [(0.0, SINGLE_DOSE_MG), (0.0, SINGLE_DOSE_MG)],
        WEIGHT_KG,
        ABSORPTION_LAG_H,
        HALF_LIFE_H,
    )
    np.testing.assert_allclose(double, 2.0 * single)


def test_concentration_is_non_negative_for_any_plan():
    """Concentration must never go negative regardless of dosing plan."""
    time_h = np.linspace(0.0, 30.0, 300)
    ingestion_plan = [(0.0, 100.0), (5.0, 200.0), (12.0, 150.0)]
    conc = caffeine_concentration_mg_per_kg(
        time_h, ingestion_plan, WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H
    )
    assert np.all(conc >= 0.0)


def test_empty_ingestion_plan_yields_zero_concentration():
    time_h = np.linspace(0.0, 10.0, 100)
    conc = caffeine_concentration_mg_per_kg(time_h, [], WEIGHT_KG, ABSORPTION_LAG_H, HALF_LIFE_H)
    assert np.all(conc == 0.0)


def test_load_food_catalog_parses_reference_foods():
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Pocari doble", "reference_size": "500 ml", "carbs_g": 70},
                {"name": "Self-made gel", "reference_size": "200 gr", "carbs_g": 125},
            ]
        }
    )

    assert catalog["Pocari doble"]["carbs_g_per_unit"] == pytest.approx(70)
    assert catalog["Self-made gel"]["reference_size"] == "200 gr"


def test_race_nutrition_plan_totals_follow_target_rate():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "2:00:00"},
        {"Point Name": "A2", "Elapsed Time": "3:30:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Pocari", "reference_size": "500 ml", "carbs_g": 70},
                {"name": "Gel", "reference_size": "200 gr", "carbs_g": 125},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Pocari", "Gel"]},
    )

    assert plan["totals"]["total_time_h"] == pytest.approx(3.5)
    assert plan["totals"]["moving_target_carbs_g"] == pytest.approx(315.0)
    assert plan["totals"]["planned_total_carbs_g"] == pytest.approx(315.0)
    assert plan["rows"][1]["segment_target_carbs_g"] == pytest.approx(180.0)
    assert plan["rows"][2]["segment_target_carbs_g"] == pytest.approx(135.0)


def test_race_nutrition_plan_supports_manual_segment_and_aid_choices():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 70},
                {"name": "Gel", "reference_size": "200 gr", "carbs_g": 30},
                {"name": "Banana", "reference_size": "1 piece", "carbs_g": 20},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=80,
        food_catalog=catalog,
        segment_foods_cfg={
            "default": [
                {"food": "Drink", "ratio": 3},
                {"food": "Gel", "ratio": 1},
            ],
            "by_segment": {"A1": [{"food": "Drink", "ratio": 3}, {"food": "Gel", "ratio": 1}]},
        },
        aid_station_intake_cfg={"by_segment": {"A1": [{"food": "Banana", "units": 2.0}]}},
    )

    row = plan["rows"][1]
    segment = {a["food"]: a for a in row["segment_allocations"]}
    aid = {a["food"]: a for a in row["aid_allocations"]}

    assert segment["Drink"]["target_carbs_g"] == pytest.approx(30.0)
    assert segment["Gel"]["target_carbs_g"] == pytest.approx(10.0)
    assert aid["Banana"]["units"] == pytest.approx(2.0)
    assert row["row_target_carbs_g"] == pytest.approx(80.0)
    assert row["row_actual_carbs_g"] == pytest.approx(80.0)
    assert row["row_carbs_per_h"] == pytest.approx(80.0)
    assert plan["totals"]["planned_total_carbs_g"] == pytest.approx(80.0)


def test_explicit_aid_intake_can_exceed_segment_budget():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "0:30:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 60},
                {"name": "Banana", "reference_size": "1 piece", "carbs_g": 20},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Drink"]},
        aid_station_intake_cfg={"by_segment": {"A1": [{"food": "Banana", "units": 3.0}]}},
    )

    row = plan["rows"][1]
    assert row["row_target_carbs_g"] == pytest.approx(45.0)
    assert row["segment_target_carbs_g"] == pytest.approx(0.0)
    assert row["aid_target_carbs_g"] == pytest.approx(60.0)
    assert plan["totals"]["planned_total_carbs_g"] == pytest.approx(60.0)


def test_fraction_constraints_and_carry_over_between_segments():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "S1", "Elapsed Time": "1:00:00"},
        {"Point Name": "S2", "Elapsed Time": "2:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {
                    "name": "Self-made gel",
                    "reference_size": "200 gr",
                    "carbs_g": 125,
                    "accepted_fractions": [0.33, 0.5, 0.67, 1.0],
                }
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Self-made gel"]},
    )

    row1 = plan["rows"][1]
    row2 = plan["rows"][2]

    assert row1["row_target_carbs_g"] == pytest.approx(90.0)
    assert row1["row_actual_carbs_g"] == pytest.approx(83.75)
    assert row1["carry_carbs_g"] == pytest.approx(6.25)

    assert row2["row_target_carbs_g"] == pytest.approx(90.0)
    assert row2["row_actual_carbs_g"] == pytest.approx(83.75)
    assert row2["carry_carbs_g"] == pytest.approx(12.5)


def test_fraction_constraints_can_limit_to_whole_units_only():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "S1", "Elapsed Time": "0:30:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {
                    "name": "Youkan",
                    "reference_size": "60 gr",
                    "carbs_g": 38,
                    "accepted_fractions": [1.0],
                }
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=30,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Youkan"]},
    )

    row = plan["rows"][1]
    allocation = row["segment_allocations"][0]
    assert allocation["units"] in {0.0, 1.0}


def test_manual_aid_intake_overrides_aid_target_and_food_choices():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 70},
                {"name": "Banana", "reference_size": "1 piece", "carbs_g": 22},
                {"name": "Shio onigiri", "reference_size": "1 piece", "carbs_g": 27},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Drink"]},
        aid_station_intake_cfg={
            "by_segment": {
                "A1": [
                    {"food": "Banana", "units": 1.0},
                    {"food": "Shio onigiri", "units": 1.0},
                ]
            }
        },
    )

    row = plan["rows"][1]
    assert row["aid_target_carbs_g"] == pytest.approx(49.0)
    assert row["segment_target_carbs_g"] == pytest.approx(41.0)
    assert row["row_actual_carbs_g"] == pytest.approx(90.0)


def test_manual_aid_intake_can_exceed_segment_target_and_be_compensated_later():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00"},
        {"Point Name": "A2", "Elapsed Time": "2:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 30},
                {"name": "Banana", "reference_size": "1 piece", "carbs_g": 22},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Drink"]},
        aid_station_intake_cfg={"by_segment": {"A1": [{"food": "Banana", "units": 5}]}},
    )

    row1 = plan["rows"][1]
    row2 = plan["rows"][2]

    assert row1["aid_target_carbs_g"] == pytest.approx(110.0)
    assert row1["segment_target_carbs_g"] == pytest.approx(0.0)
    assert row1["carry_carbs_g"] == pytest.approx(-20.0)
    assert row2["row_target_carbs_g"] == pytest.approx(90.0)
    assert row2["segment_target_carbs_g"] == pytest.approx(70.0)


def test_manual_aid_intake_supports_non_catalog_food_by_carbs():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 70},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Drink"]},
        aid_station_intake_cfg={"by_segment": {"A1": [{"food": "Rice curry", "carbs_g": 65}]}},
    )

    row = plan["rows"][1]
    aid = row["aid_allocations"][0]
    assert aid["food"] == "Rice curry"
    assert aid["actual_carbs_g"] == pytest.approx(65.0)
    assert aid["reference_size"] == "custom"


def test_caffeine_plan_computes_checkpoint_concentration():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00"},
        {"Point Name": "A2", "Elapsed Time": "2:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Drink", "reference_size": "500 ml", "carbs_g": 30},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=60,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Drink"]},
        caffeine_ingestion_plan=[(1.0, 200.0)],
        caffeine_weight_kg=70.0,
        caffeine_absorption_lag_h=0.5,
        caffeine_half_life_h=5.5,
    )

    row_at_dose = plan["rows"][1]
    row_after_absorption = plan["rows"][2]
    assert row_at_dose["caffeine_dose_mg_at_point"] == pytest.approx(200.0)
    assert row_at_dose["caffeine_concentration_mg_per_kg"] == pytest.approx(0.0)
    assert row_after_absorption["caffeine_concentration_mg_per_kg"] > 0.0
    assert plan["totals"]["caffeine"]["total_dose_mg"] == pytest.approx(200.0)


def test_hydration_counts_drink_food_volume_without_short_segment_catch_up():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00", "Stop Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00", "Stop Time": "0:03:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Pocari", "reference_size": "500 ml", "carbs_g": 70},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=70,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Pocari"]},
        sweat_rate_ml_per_h=700,
        hydration_weight_kg=65,
        max_body_weight_loss_pct=1.5,
    )

    row = plan["rows"][1]
    assert row["row_planned_fluids_ml"] == pytest.approx(500.0)
    assert row["row_supplemental_fluids_ml"] == pytest.approx(0.0)
    assert row["row_total_fluids_ml"] == pytest.approx(500.0)
    assert row["cumulative_hydration_balance_ml"] == pytest.approx(-200.0)


def test_hydration_never_overdrinks_when_segment_drink_already_exceeds_sweat():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00", "Stop Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00", "Stop Time": "0:03:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Pocari", "reference_size": "500 ml", "carbs_g": 70},
            ]
        }
    )

    with pytest.raises(ValueError, match="over-drink"):
        build_race_nutrition_plan(
            pacing_rows=pacing_rows,
            target_carbs_g_per_h=140,
            food_catalog=catalog,
            segment_foods_cfg={"default": ["Pocari"]},
            sweat_rate_ml_per_h=700,
            hydration_weight_kg=65,
            max_body_weight_loss_pct=1.5,
        )


def test_hydration_uses_supplemental_drink_to_stay_within_maximum_loss_bound():
    pacing_rows = [
        {"Point Name": "START", "Elapsed Time": "0:00:00", "Stop Time": "0:00:00"},
        {"Point Name": "A1", "Elapsed Time": "1:00:00", "Stop Time": "0:03:00"},
        {"Point Name": "A2", "Elapsed Time": "2:00:00", "Stop Time": "0:03:00"},
        {"Point Name": "A3", "Elapsed Time": "3:00:00", "Stop Time": "0:00:00"},
    ]
    catalog = load_food_catalog(
        {
            "foods": [
                {"name": "Gel", "reference_size": "1 unit", "carbs_g": 90},
            ]
        }
    )

    plan = build_race_nutrition_plan(
        pacing_rows=pacing_rows,
        target_carbs_g_per_h=90,
        food_catalog=catalog,
        segment_foods_cfg={"default": ["Gel"]},
        sweat_rate_ml_per_h=700,
        hydration_weight_kg=60,
        max_body_weight_loss_pct=1.5,
    )

    final_row = plan["rows"][-1]
    assert final_row["row_supplemental_fluids_ml"] > 0
    assert final_row["cumulative_hydration_balance_ml"] >= -900.0 - 1e-6
    assert final_row["cumulative_hydration_balance_ml"] <= 1e-6
