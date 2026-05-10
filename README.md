# trail-race-planner

Segment-by-segment trail ultra race planner. Give it a GPX file, an athlete profile,
and a race config — get a pacing plan, nutrition/hydration schedule, and dropbag list
exported to Excel, plus an interactive elevation profile and a mobile race-plan HTML.

---

## Models

| Model | What it does |
|---|---|
| Piecewise Riegel + FED | Predicts finish time from a flat-race PB, adjusting for elevation via Flat Equivalent Distance |
| Grade Adjusted Pace (GAP) | Distributes pace point-by-point using a terrain correction curve |
| ITRA Score Predictor | Converts between finish time and ITRA score (fitted from UTMB 2025 data) |
| Nutrition planner | Allocates food per segment to hit a g/h carb target with carry-over |
| Hydration planner | Prescribes supplemental water (500 ml flasks) to stay within a sweat-loss limit |
| Caffeine model | First-order absorption + elimination curve from an exact-time dose plan |

---

## Installation

```bash
git clone https://github.com/your-user/trail-race-planner.git
cd trail-race-planner
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Quick Start

```bash
# Predict time from athlete PB
python -m race_planner.main config/races/tgt_2026.yaml --athlete carlos

# Plan to a target finish time, with fatigue
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos --mode target_time --target-time 29:00:00 --fatigue-mode race

# Plan to a target ITRA score
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos --mode target_itra --target-itra-score 750
```

Outputs go to `results/` as configured in the race YAML.

---

## CLI

```
python -m race_planner.main <race.yaml> [--athlete NAME]
    [--mode athlete_pb|target_time|target_itra|grade_adjusted_pace]
    [--target-time HH:MM:SS]
    [--target-itra-score N]
    [--target-grade-adjusted-pace MM:SS]
    [--fatigue-mode none|race|athlete]
    [--fatigue-total-decay-pct PCT]
```

---

## Configuration

### Athlete (`config/athletes/*.yaml`)

```yaml
athlete:
  name: "Carlos"
  weight_kg: 65
  itra_points: 720
  hydration:
    sweat_rate_ml_per_h: 750
  reference_performance:
    distance_km: 42.195
    time: "2:45:00"
  gap_curve:
    points: []   # empty = use built-in default
```

### Race (`config/races/*.yaml`)

```yaml
race:
  name: "Tokyo Grand Trail 2026"
  start_time: "15:50"
  gpx_file: "config/gpx_repo/tgt_2026.gpx"
  output_file: "results/tgt_2026_segment_analysis.xlsx"
  itra_reference_points:
    - { reference_time: "29:52:00", reference_score: 750 }
  planning:
    fatigue_total_decay_pct: 14

aid_stations:
  - { name: "Start", distance_km: 0.0, stop_time_s: 0 }
  - { name: "CP1", distance_km: 22.5, elevation_m: 1820, stop_time_s: 300, jap_name: "第1CP", gmaps_link: "..." }
  - { name: "Finish", distance_km: 161.0, stop_time_s: 0 }
```

### Nutrition (inside race YAML, optional)

```yaml
nutrition:
  target_carbs_g_per_h: 90
  food_catalog_file: "config/nutrition/foods.yaml"

  segment_foods:
    default:
      - { food: "Pocari doble", ratio: 1 }
      - { food: "Self-made gel", ratio: 1.2 }
    by_segment:
      "A3 Sanogawa":
        - { food: "Pocari doble", ratio: 1 }
        - { food: "Self-made gel", ratio: 2 }

  aid_station_intake:
    by_segment:
      "A1 Mitake Aid":
        - { food: "Banana", units: 0.5 }
        - { food: "Rice curry", carbs_g: 30 }

  dropbag_points: ["A2 Jurigi (1)", "A6 Jurigi (2)"]

  caffeine_plan:
    ingestion_plan:
      - { time_h: 5.0, dose_mg: 200 }
      - { time_h: 13.0, dose_mg: 200 }
```

Sweat rate comes from `athlete.hydration.sweat_rate_ml_per_h`. Supplemental water
is added in 500 ml flask increments to stay within 1.5% BW loss.

### Food Catalog (`config/nutrition/foods.yaml`)

```yaml
foods:
  - name: "Self-made gel"
    reference_size: "1 unit"
    carbs_g: 25
    accepted_fractions: [0.5, 1.0]
  - name: "Pocari doble"
    reference_size: "500 ml"   # ml -> fluid_ml_per_unit auto-detected
    carbs_g: 70
    sodium_mg: 520
```

---

## Outputs

| File | Description |
|---|---|
| `*_segment_analysis.xlsx` | Segment stats, race plan, nutrition plan, dropbag plan |
| `*_elevation_profile.html` | Interactive Plotly elevation chart with aid station markers |
| `*_race_plan.html` | Mobile-friendly table with wall-clock times, splits, and the elevation chart |

---

## Tests

```bash
pytest
```
