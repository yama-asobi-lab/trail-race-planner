# trail-race-planner

A Python tool for trail ultra-marathon race planning. Given a GPX course file, an
athlete profile, and a race configuration, it computes a segment-by-segment pacing
plan that accounts for elevation, terrain, and cumulative fatigue — then exports the
results to an Excel workbook, a standalone interactive elevation profile, and a
smartphone-friendly race plan HTML report.

---

## Table of Contents

- [Overview](#overview)
- [Key Models](#key-models)
- [Installation](#installation)
- [MVP First Run (Minimum Required)](#mvp-first-run-minimum-required)
- [Comprehensive Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
  - [Athlete Profile](#athlete-profile-configathletesyaml)
  - [Race Config](#race-config-configracesyaml)
  - [Nutrition Config](#nutrition-config-inside-race-yaml)
  - [Food Catalog](#food-catalog-confignutritionfoodsyaml)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)

---

## Overview

Planning a trail ultra means dealing with two compounding effects that flat-road
pace models ignore:

1. **Terrain**: uphills slow you down; moderate downhills speed you up, but steep 
   downhills slow you down too (braking effect).  
2. **Fatigue**: the effective difficulty of an extra kilometre grows non-linearly
   beyond marathon distance.

This tool combines three validated models to address both:

| Model | Role |
|---|---|
| Piecewise Riegel | Predicts total race time from a flat-race reference PB |
| Flat Equivalent Distance (FED) | Converts elevation gain into extra "flat" effort |
| Grade Adjusted Pace (GAP) | Distributes pace across the course point-by-point |

A **nutrition and hydration planner** runs on top of the pacing plan. It allocates
food and water segment by segment to meet carbohydrate targets and replace sweat loss,
and models caffeine concentration over the race.

---

## Key Models

### Piecewise Riegel

Riegel's power-law formula predicts race time from a known reference:

$$T(D) = T_{\text{ref}} \times \left(\frac{D_{\text{eff}}}{D_{\text{ref}}}\right)^{k(D)}$$

For ultra distances the exponent grows with a square-root penalty above the
reference distance:

$$k(D) = 1.06 + 0.013422 \times \sqrt{\max(D_{\text{eff}} - D_{\text{ref}},\ 0)}$$

The coefficient was fitted to **flat road and track world records** across
distances from 5 km to 48 hours events, then validated against the fixed-exponent Riegel,
linear, quadratic, and hybrid alternatives on the same flat-record dataset —
the sqrt form offers the best balance of accuracy and generality.

### Flat Equivalent Distance (FED)

Elevation gain is converted to virtual flat distance before applying Riegel:

$$FED = D + \frac{G}{100}$$

where $D$ is horizontal distance (km) and $G$ is total elevation gain (m).
The factor of 100 m/km follows Naismith's rule and the ITRA convention.

*Example*: A 160 km / 11,000 m course has FED ≈ 270 km — the athlete's body
works as hard as running 270 km flat, not 160 km.

### Grade Adjusted Pace (GAP)

A per-point correction factor maps terrain gradient to pace multiplier. The
default calibration table is:

| Grade | Correction | Effect |
|---|---|---|
| +20% | 2.60 × | Anchor — constant-vspeed tail above this |
| +1% | 1.08 × | 8% slower |
| 0% | 1.00 × | Flat reference |
| −1% | 0.97 × | 3% faster |
| −5% | 0.85 × | 15% faster (good runnable descent) |
| −6% | 0.90 × | 10% faster (braking begins) |
| −20% | 1.60 × | Anchor — constant-vspeed tail below this |

Between knots the curve is piecewise-linear. Beyond ±20 % grade, a
**constant-vertical-speed** tail is applied instead of unbounded extrapolation:

$$c(g) = c(g_{\text{cutoff}}) \times \frac{g}{g_{\text{cutoff}}}$$

Both the knot table and the cutoff grade can be overridden per athlete.

### ITRA Score Prediction

ITRA scores follow a logarithmic relationship with finish time:

$$\text{score}(t) = A - B \ln(t)$$

The coefficients $A$ and $B$ are fitted from UTMB 2025 finisher data. The
resulting curve is stored as a pre-computed ratio table
(`itra_score_ratios.py`), indexed by score. At runtime, one reference
`(time, score)` pair from the race config anchors the curve to the specific
race, and `ItraScorePredictor` interpolates to convert between score and time.
This enables `--mode target_itra` and provides a score estimate for any
computed finish time.

### Nutrition and Hydration

The planner builds a **segment-by-segment fueling plan** on top of the pacing output.

**Carbohydrates**: The user specifies which foods are available in each segment
(and their relative carb-share ratios); the planner allocates exact quantities to
hit a `target_carbs_g_per_h` target. Aid-station intake can be defined manually
by food name and units (or carbs_g for on-course foods without a catalog entry).
Carb shortfalls and surpluses carry over between segments.

**Hydration**: A sweat-loss model (sweat rate from the athlete profile × segment
duration) is compared to fluids from planned foods plus supplemental carried water.
Supplemental water is quantized to integer 500 ml flasks. The model tracks a
running cumulative sweat imbalance and validates it stays within the configured
maximum body-weight-loss threshold (default 1.5 % BW).

**Caffeine**: An exact-time ingestion plan (dose + race-time hours) is modelled
with first-order absorption (absorption lag) and first-order elimination (half-life).
The resulting concentration curve is reported at every checkpoint.

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/yama-asobi-lab/trail-race-planner.git
cd trail-race-planner
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## MVP First Run (Minimum Required)

If you are using this tool for the first time and only want a **working smartphone HTML race plan**,
you only need:

1. A GPX file for the course.
2. One race YAML that points to that GPX.

You can skip nutrition, ITRA references, custom GAP curves, and even custom aid stations for your
first run.

### 1) Put your GPX in the repo

Example path:

`config/gpx_repo/my_course.gpx`

### 2) Use the minimal race config template

Copy `config/races/my_course.yaml`:

```yaml
race:
  name: "My Course"
  gpx_file: "config/gpx_repo/my_course.gpx"
  output_file: "results/my_course_segment_analysis.xlsx"

# Empty list is valid for MVP.
# The planner will automatically use Start (0 km) and Finish (course end).
aid_stations: []
```

Minimum fields required for `race` are:
- `name`
- `gpx_file`
- `output_file`

### 3) Run the planner

```bash
python -m race_planner.main config/races/my_course.yaml
```

This uses the default athlete profile (`yet_another_sato`) and generates:

- `results/my_course_segment_analysis.xlsx`
- `results/my_course_segment_analysis_elevation_profile.html`
- `results/my_course_race_plan.html`  (smartphone-friendly report)

### 4) (Optional) Add basic aid stations next

Once MVP works, you can add simple checkpoints for better split planning:

```yaml
aid_stations:
  - name: "Start"
    distance_km: 0.0
    stop_time_s: 0
  - name: "Aid 1"
    distance_km: 25.0
    stop_time_s: 180
  - name: "Finish"
    distance_km: 52.4
    stop_time_s: 0
```

Only `name` and `distance_km` are required per aid station; other fields are optional.

---

## Comprehensive Quick Start

```bash
# Predict Carlos's time for TGT 2026 based on his marathon PB
python -m race_planner.main config/races/tgt_2026.yaml --athlete carlos

# Plan to a specific finish time
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos \
    --mode target_time \
    --target-time 28:30:00

# Plan to a target ITRA score (requires itra_reference_points in race config)
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos \
    --mode target_itra \
    --target-itra-score 750

# Plan from a target grade-adjusted pace (MM:SS)
python -m race_planner.main config/races/tgt_2026.yaml \
  --athlete carlos \
  --mode grade_adjusted_pace \
  --target-grade-adjusted-pace 5:15

# Include the nutrition/fueling column in the main race-plan HTML
python -m race_planner.main config/races/tgt_2026.yaml \
  --athlete carlos \
  --mode target_time \
  --target-time 30:00:00 \
  --nutrition yes
```

Outputs land in `results/`:

```
results/
  tgt_2026_segment_analysis.xlsx
  tgt_2026_segment_analysis_elevation_profile.html
  tgt_2026_race_plan.html
```

By default, the main HTML report does not include nutrition/fueling details. Pass
`--nutrition yes` to add the rightmost fuel column to the race-plan table.

---

## CLI Reference

```
python -m race_planner.main <race_config.yaml> [options]

Positional:
  race_config.yaml          Path to the race YAML (relative to project root)

Options:
  --athlete NAME            Athlete name, matches a file in config/athletes/
                            (default: yet_another_sato)
  --mode MODE               Planning mode (default: athlete_pb)
    athlete_pb              Use athlete's reference PB to predict race time
    target_time             Back-solve to a given total finish time
    target_itra             Back-solve to a given ITRA performance score
    grade_adjusted_pace    Back-solve from a target GAP-weighted pace
  --target-time HH:MM:SS    Required when --mode target_time
  --target-itra-score N     Required when --mode target_itra
  --target-grade-adjusted-pace MM:SS
                            Required when --mode grade_adjusted_pace
  --fatigue-mode {none|athlete|race}
                            Fatigue model source (default: none)
                              none              No fatigue modeling
                              race              Use race YAML planning defaults
                              athlete           Use athlete physiology (TBD)
  --fatigue-total-decay-pct PCT
                            Override fatigue with linear decay PCT (0–100)
                            Takes precedence over --fatigue-mode
  --nutrition {yes|no}      Include nutrition/fueling column in main HTML report
                            (default: no)
```

Terminal output includes a summary block:

```
====================================================
RACE PLAN SUMMARY
  Mode:          athlete_pb
  Athlete:       Carlos
  Riegel approx running time: 26:12:05
  Grade-adjusted running time: 28:01:44
  Stop time:     1:00:00
  Fatigue model:  Linear decay 10.0%
  Finish time:   29:01:44
  ITRA score:    752
====================================================
```

---

## Configuration

### Athlete Profile (`config/athletes/*.yaml`)

```yaml
athlete:
  name: "Carlos"
  weight_kg: 65              # body weight in kg — used for hydration modelling

  hydration:
    sweat_rate_ml_per_h: 750  # sweat loss rate in ml/h (race conditions)

  itra_points: 720

  reference_performance:
    distance_km: 42.195        # flat race distance (marathon recommended)
    time: "2:45:00"            # finish time for that flat race
    notes: "Road marathon PB"  # optional

  gap_curve:
    points: []                 # empty = use default table
    # Custom example (overrides entire default table):
    # - [ 0.20,  2.50  ]  # +20% cutoff anchor
    # - [ 0.01,  1.075 ]
    # - [ 0.00,  1.00  ]
    # - [-0.01,  0.965 ]
    # - [-0.06,  0.8   ]
    # - [-0.07,  0.865 ]
    # - [-0.20,  1.71  ]  # -20% cutoff anchor

  preferences:                 # optional — informational only at present
    threshold_flat_pace_per_km: "3:50/km"
    aerobic_threshold_flat_pace_per_km: "4:40/km"

  fatigue_physiology: {}       # optional; reserved for future process-based model
```

**Notes on the GAP curve**:
- Points are `[grade_decimal, correction_factor]`, e.g. `[0.10, 1.50]` means
  a 10% uphill is 1.5× slower than flat.
- If `points` is empty, `DEFAULT_GAP_CURVE` from `pace_calculator.py` is used.
- The two anchor points at ±20% mark where the constant-vertical-speed tail
  begins; include them in your custom curve.
- Points do **not** need to be sorted — the calculator sorts them internally.

**Notes on fatigue modeling**:
- Fatigue can be controlled via CLI (`--fatigue-mode` and `--fatigue-total-decay-pct`).
- CLI override `--fatigue-total-decay-pct` takes precedence over all config defaults.
- `--fatigue-mode none` (default): no fatigue modeling.
- `--fatigue-mode race`: uses `race.planning.fatigue_total_decay_pct` if present.
- `--fatigue-mode athlete`: reserved for future process-based physiological model.
- Linear model: pace multiplier rises from 1.0 at start to 1.0 + (decay_pct/100) at finish.
- Example: `--fatigue-total-decay-pct 10` means 10% slower pace at the finish.

---

### Race Config (`config/races/*.yaml`)

```yaml
race:
  name: "Tokyo Grand Trail 2026"
  start_time: 15:50            # Race start time in `HH:MM` or `HH:MM:SS`; defaults to `00:00:00`
  distance_km: 161             # informational — actual distance comes from GPX
  elevation_gain_m: 11000      # informational

  gpx_file: "config/gpx_repo/TGT_2025.gpx"   # path relative to project root
  output_file: "results/tgt_2026_segment_analysis.xlsx"

  resample_m: 5                # resample GPX to this interval (metres)
  elevation_tolerance_m: 50    # tolerance for aid station elevation validation

  itra_reference_points:       # optional; enables ITRA prediction
    - reference_time: "21:52:00"
      reference_score: 829
    - reference_time: "32:00:00"
      reference_score: 700
      notes: "estimated"
  
  planning:                    # optional; default planning assumptions
    fatigue_total_decay_pct: 0 # 0 = no fatigue (delegated to athlete config or CLI)

# An empty list is valid — the whole course is treated as one segment.
aid_stations:
  - name: "Start"
    distance_km: 0.0
    elevation_m: 1350
    stop_time_s: 0

  - name: "Checkpoint 1"
    jap_name: "第1CP"          # optional Japanese name (shown in plot)
    distance_km: 22.5
    elevation_m: 1820
    stop_time_s: 300           # 5-minute stop
    notes: "Drop bag available"
    gmaps_link: "https://maps.google.com/?q=..."   # shown in elevation plot

  # ... more stations ...

  - name: "Finish"
    distance_km: 161.0
    elevation_m: 1350
    stop_time_s: 0
```

**Aid station fields**:

| Field | Required | Description |
|---|---|---|
| `name` | yes | English display name |
| `distance_km` | yes | Distance from start |
| `jap_name` | no | Japanese name (displayed in HTML plot) |
| `elevation_m` | no | Reference elevation for validation |
| `stop_time_s` | no | Planned stop in seconds (default 0) |
| `notes` | no | Free text, shown when clicking the marker in the plot |
| `gmaps_link` | no | Google Maps URL linked in the plot |

---

### Nutrition Config (inside race YAML)

Add a `nutrition` block to the race YAML to enable the Nutrition Plan and Dropbag
Plan sheets in the Excel output.

```yaml
nutrition:
  target_carbs_g_per_h: 90
  food_catalog_file: "config/nutrition/foods.yaml"  # default path if omitted

  # Defines WHAT foods are available per segment; the planner computes HOW MUCH.
  # Ratio is a carb-share weight. Example: ratio 1 vs 2 means 1/3 vs 2/3 of carb target.
  segment_foods:
    default:
      - { food: "Pocari doble", ratio: 1 }
      - { food: "Self-made gel", ratio: 1.2 }
      - { food: "Youkan", ratio: 0.33 }

    by_segment:            # optional per-checkpoint overrides (key = aid station name)
      "A3 Sanogawa":
        - { food: "Pocari doble", ratio: 1 }
        - { food: "Self-made gel", ratio: 2 }
        - { food: "Hoshi imo", ratio: 1 }

  # Manual intake at specific aid stations (fixed amounts, not ratio-allocated).
  aid_station_intake:
    by_segment:
      "A1 Mitake Aid":
        - { food: "Banana", units: 0.5 }
        - { food: "Rice curry", carbs_g: 30 }   # carbs_g for non-catalog foods
      "Hinode Oguno 7-eleven":
        - { food: "Coke", units: 1.0 }

  # Dropbag access points. The Dropbag Plan sheet groups required foods
  # between each dropbag, with a START stash for the opening leg.
  dropbag_points:
    - "A2 Jurigi (1)"
    - "A6 Jurigi (2)"

  # Caffeine: exact-time ingestion plan (hours from race start).
  caffeine_plan:
    ingestion_plan:
      - { time_h: 5.0,  dose_mg: 200 }
      - { time_h: 8.0,  dose_mg: 200 }
      - { time_h: 13.0, dose_mg: 200 }
      - { time_h: 22.0, dose_mg: 200 }
      - { time_h: 27.0, dose_mg: 200 }
```

**How segment food allocation works**:

1. The planner tracks a cumulative carb carry-over (shortfall or surplus vs. target).
2. For each segment, it computes the carb target = segment moving time × `target_carbs_g_per_h` + carry-over.
3. Manual aid-station intake (from `aid_station_intake`) is applied first and subtracted from that target.
4. The remainder is distributed across `segment_foods` proportional to their `ratio` values.
5. Units are rounded to `accepted_fractions` if defined in the food catalog.

**Hydration** is driven by `athlete.hydration.sweat_rate_ml_per_h` (set in the
athlete profile). The planner adds supplemental carried water in 500 ml flask
increments as needed to stay within the 1.5 % body-weight-loss limit.

---

### Food Catalog (`config/nutrition/foods.yaml`)

```yaml
foods:
  - name: "Self-made gel"
    reference_size: "1 unit"
    carbs_g: 25
    caffeine_mg: 0
    accepted_fractions: [0.5, 1.0]   # only half or whole units allowed

  - name: "Pocari doble"
    reference_size: "500 ml"         # ml in reference_size → fluid_ml_per_unit auto-detected
    carbs_g: 70
    sodium_mg: 520

  - name: "Coke"
    reference_size: "350 ml"
    carbs_g: 38
    caffeine_mg: 35
```

**Food catalog fields**:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique identifier (used as key in race YAML) |
| `reference_size` | no | Serving size string, e.g. `"500 ml"` or `"1 unit"` |
| `carbs_g` | yes* | Carbs per serving (`carbs_g_per_unit` also accepted) |
| `caffeine_mg` | no | Caffeine per serving |
| `protein_g` | no | Protein per serving |
| `fat_g` | no | Fat per serving |
| `sodium_mg` | no | Sodium per serving |
| `fluid_ml_per_unit` | no | Override fluid volume; auto-detected from `reference_size` if omitted |
| `accepted_fractions` | no | List of allowed fractional units, e.g. `[0.5, 1.0]` |
| `unit_label` | no | Display label for units (default: `"serving"`) |
| `notes` | no | Free text |

---

## Outputs

### Excel Workbook (`.xlsx`)

**Sheet: Segment Statistics**

| Column | Description |
|---|---|
| Point Name | Aid station name |
| Total Distance (km) | Cumulative distance from start |
| Elevation (m) | Current elevation |
| Accum. Elevation Gain (m) | Total gain to this point |
| Segment Distance (km) | Distance since previous station |
| Segment Elevation Gain (m) | Gain in this segment |
| Segment Elevation Loss (m) | Loss in this segment |
| Average Gradient (%) | Average positive gradient |

**Sheet: Race Plan (`<mode>`)**

Adds running-time and stop-time columns to the above, plus a summary block:

| Column | Description |
|---|---|
| Segment Running Time | Running time for this segment (H:MM:SS) |
| Avg Pace (mm:ss/km) | Segment average pace by horizontal distance |
| Avg Grade-Adjusted Pace (mm:ss/km) | Segment average pace on GAP-weighted distance |
| Stop Time | Planned stop (H:MM:SS) |
| Elapsed Time | Cumulative elapsed race time at station (H:MM:SS) |

**Sheet: Nutrition Plan**

One row per checkpoint. Key columns:

| Column | Description |
|---|---|
| Segment Carb Target (g) | Moving-time carb target including carry-over |
| Moving Plan Carbs (g) | Carbs from segment foods |
| Aid Plan Carbs (g) | Carbs from manual aid station intake |
| Total Segment Carbs per Hour (g/h) | Effective carb rate for the segment |
| Estimated Sweat Loss (ml) | Sweat loss for the segment |
| Total Planned Drink (ml) | Fluids from food + carried water |
| Cum Sweat Imbalance (ml / %BW) | Running hydration balance |
| Segment Caffeine Intake (mg [time]) | Doses with ingestion time |
| Caffeine Concentration (mg/kg) | Model concentration at checkpoint |
| Carried Water (ml) | Supplemental flask water (500 ml increments) |
| Sports Drink (ml) | Volume from sports drinks (Pocari etc.) |
| Gels (qty) | Gel/jelly allocations |
| Others (qty) | Other foods and custom intakes |

A summary block below the table reports totals, hydration balance, and caffeine peak.

**Sheet: Dropbag Plan**

One row per dropbag point (plus START stash). Columns:

| Column | Description |
|---|---|
| Dropbag Point | Bag name (`START`, or aid station name) |
| Covers Segments Until | Last checkpoint covered by this bag |
| Stash Food Plan (qty) | Segment foods to pack |
| Stash Caffeine Plan (mg [time]) | Caffeine doses within this leg |

### Interactive Elevation Profile (`.html`)

A self-contained Plotly HTML file with:
- Elevation line with shaded area under the curve
- Aid station markers (gold diamonds) with Japanese names
- Hover tooltips: distance, elevation, grade, cumulative gain
- Click a marker to see its notes and a Google Maps link
- Course statistics box (total distance, gain, loss)
- No external dependencies — opens in any browser offline

### Smartphone Race Plan Report (`_race_plan.html`)

A mobile-oriented HTML report written as `<race>_race_plan.html`, for example
`results/tgt_2026_race_plan.html`.

It includes:
- Sticky top header row and sticky left aid-station column
- English and Japanese aid-station names in the same cell
- Total and split distance in one cell
- Accumulated gain plus split gain/loss columns
- Elapsed, split, and wall-clock time in one timing cell
- Wall-clock times derived from `race.start_time` in the race YAML
- Average split pace and combined comments block
- Embedded elevation profile below the table
- Summary cards for finish time, distance, gain, and start time

---

## Project Structure

```
trail-race-planner/
│
├── race_planner/              # Main package
│   ├── main.py                # CLI entry point + Excel sheet writers
│   ├── course/
│   │   ├── course.py          # GPX loading, resampling, Course class
│   │   └── segment_analyzer.py# Segment stats, elevation validation, Excel export
│   ├── models/
│   │   ├── itra_predictor.py  # ITRA score ↔ time prediction
│   │   ├── itra_score_ratios.py# Empirical ITRA ratio table (UTMB 2025)
│   │   ├── nutrition.py       # Nutrition, hydration, and caffeine planner
│   │   └── tools.py           # Pace/speed/time/unit conversion utilities
│   ├── planner/
│   │   └── pace_calculator.py # PaceCalculator: Riegel + FED + GAP
│   └── visualization/
│       ├── course_profile.py  # Plotly elevation profile generator
│       └── race_plan_table.py # Smartphone race-plan HTML report generator
│
├── config/
│   ├── athletes/              # Athlete YAML profiles
│   │   ├── carlos.yaml
│   │   └── yet_another_sato.yaml
│   ├── gpx_repo/              # GPX course files
│   ├── nutrition/
│   │   └── foods.yaml         # Food catalog
│   └── races/                 # Race YAML configs
│       ├── tgt_2026.yaml
│       └── oku_long_2025.yaml
│
├── analysis/                  # Standalone analysis scripts (not part of main pipeline)
│   ├── itra_scores.py         # Fit ITRA score vs. time curve
│   ├── validate_ultra_fatigue_model.py  # Compare Riegel variants against race records
│   └── analyze_gap_vertical_speed_cutoff.py  # GAP curve diagnostic plots
│
├── tests/
│   ├── conftest.py            # Shared fixtures (isolated from live configs)
│   ├── fixtures/              # Test-only YAML copies
│   ├── test_course.py
│   ├── test_nutrition.py
│   ├── test_pace_calculator.py
│   ├── test_itra_predictor.py
│   ├── test_segment_analyzer.py
│   ├── test_config.py
│   └── test_tools.py
│
├── results/                   # Generated outputs (xlsx, html)
├── requirements.txt
├── pyproject.toml             # Black config (line-length 88, py39+)
└── pytest.ini
```

---

## Running Tests

```bash
pytest
```

The suite uses isolated test fixtures in `tests/fixtures/` so that edits to live
athlete and race configs do not break tests.

Code style is enforced with [Black](https://github.com/psf/black) and
[Ruff](https://github.com/astral-sh/ruff):

```bash
black .
ruff check .
```


---

## Table of Contents

- [Overview](#overview)
- [Key Models](#key-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
  - [Athlete Profile](#athlete-profile-configathletesyaml)
  - [Race Config](#race-config-configracesyaml)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)

---

## Overview

Planning a trail ultra means dealing with two compounding effects that flat-road
pace models ignore:

1. **Terrain**: uphills slow you down; moderate downhills speed you up, but steep 
   downhills slow you down too (braking effect).  
2. **Fatigue**: the effective difficulty of an extra kilometre grows non-linearly
   beyond marathon distance.

This tool combines three validated models to address both:

| Model | Role |
|---|---|
| Piecewise Riegel | Predicts total race time from a flat-race reference PB |
| Flat Equivalent Distance (FED) | Converts elevation gain into extra "flat" effort |
| Grade Adjusted Pace (GAP) | Distributes pace across the course point-by-point |

---

## Key Models

### Piecewise Riegel

Riegel's power-law formula predicts race time from a known reference:

$$T(D) = T_{\text{ref}} \times \left(\frac{D_{\text{eff}}}{D_{\text{ref}}}\right)^{k(D)}$$

For ultra distances the exponent grows with a square-root penalty above the
reference distance:

$$k(D) = 1.06 + 0.013422 \times \sqrt{\max(D_{\text{eff}} - D_{\text{ref}},\ 0)}$$

The coefficient was fitted to **flat road and track world records** across
distances from 5 km to 48 hours events, then validated against the fixed-exponent Riegel,
linear, quadratic, and hybrid alternatives on the same flat-record dataset —
the sqrt form offers the best balance of accuracy and generality.

### Flat Equivalent Distance (FED)

Elevation gain is converted to virtual flat distance before applying Riegel:

$$FED = D + \frac{G}{100}$$

where $D$ is horizontal distance (km) and $G$ is total elevation gain (m).
The factor of 100 m/km follows Naismith's rule and the ITRA convention.

*Example*: A 160 km / 11,000 m course has FED ≈ 270 km — the athlete's body
works as hard as running 270 km flat, not 160 km.

### Grade Adjusted Pace (GAP)

A per-point correction factor maps terrain gradient to pace multiplier. The
default calibration table is:

| Grade | Correction | Effect |
|---|---|---|
| +20% | 2.60 × | Anchor — constant-vspeed tail above this |
| +1% | 1.08 × | 8% slower |
| 0% | 1.00 × | Flat reference |
| −1% | 0.97 × | 3% faster |
| −5% | 0.85 × | 15% faster (good runnable descent) |
| −6% | 0.90 × | 10% faster (braking begins) |
| −20% | 1.60 × | Anchor — constant-vspeed tail below this |

Between knots the curve is piecewise-linear. Beyond ±20 % grade, a
**constant-vertical-speed** tail is applied instead of unbounded extrapolation:

$$c(g) = c(g_{\text{cutoff}}) \times \frac{g}{g_{\text{cutoff}}}$$

Both the knot table and the cutoff grade can be overridden per athlete.

### ITRA Score Prediction

ITRA scores follow a logarithmic relationship with finish time:

$$\text{score}(t) = A - B \ln(t)$$

The coefficients $A$ and $B$ are fitted from UTMB 2025 finisher data. The
resulting curve is stored as a pre-computed ratio table
(`itra_score_ratios.py`), indexed by score. At runtime, one reference
`(time, score)` pair from the race config anchors the curve to the specific
race, and `ItraScorePredictor` interpolates to convert between score and time.
This enables `--mode target_itra` and provides a score estimate for any
computed finish time.

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/yama-asobi-lab/trail-race-planner.git
cd trail-race-planner
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick Start

```bash
# Predict Carlos's time for TGT 2026 based on his marathon PB
python -m race_planner.main config/races/tgt_2026.yaml --athlete carlos

# Plan to a specific finish time
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos \
    --mode target_time \
    --target-time 28:30:00

# Plan to a target ITRA score (requires itra_reference_points in race config)
python -m race_planner.main config/races/tgt_2026.yaml \
    --athlete carlos \
    --mode target_itra \
    --target-itra-score 750

# Plan from a target grade-adjusted pace (MM:SS)
python -m race_planner.main config/races/tgt_2026.yaml \
  --athlete carlos \
  --mode grade_adjusted_pace \
  --target-grade-adjusted-pace 5:15
```

Outputs land in `results/`:

```
results/
  tgt_2026_segment_analysis.xlsx
  tgt_2026_segment_analysis_elevation_profile.html
  tgt_2026_race_plan.html
```

---

## CLI Reference

```
python -m race_planner.main <race_config.yaml> [options]

Positional:
  race_config.yaml          Path to the race YAML (relative to project root)

Options:
  --athlete NAME            Athlete name, matches a file in config/athletes/
                            (default: yet_another_sato)
  --mode MODE               Planning mode (default: athlete_pb)
    athlete_pb              Use athlete's reference PB to predict race time
    target_time             Back-solve to a given total finish time
    target_itra             Back-solve to a given ITRA performance score
    grade_adjusted_pace    Back-solve from a target GAP-weighted pace
  --target-time HH:MM:SS    Required when --mode target_time
  --target-itra-score N     Required when --mode target_itra
  --target-grade-adjusted-pace MM:SS
                            Required when --mode grade_adjusted_pace
  --fatigue-mode {none|athlete|race}
                            Fatigue model source (default: none)
                              none              No fatigue modeling
                              race              Use race YAML planning defaults
                              athlete           Use athlete physiology (TBD)
  --fatigue-total-decay-pct PCT
                            Override fatigue with linear decay PCT (0–100)
                            Takes precedence over --fatigue-mode
```

Terminal output includes a summary block:

```
====================================================
RACE PLAN SUMMARY
  Mode:          athlete_pb
  Athlete:       Carlos
  Riegel approx running time: 26:12:05
  Grade-adjusted running time: 28:01:44
  Stop time:     1:00:00
  Fatigue model:  Linear decay 10.0%
  Finish time:   29:01:44
  ITRA score:    752
====================================================
```

---

## Configuration

### Athlete Profile (`config/athletes/*.yaml`)

```yaml
athlete:
  name: "Carlos"
  itra_points: 750

  reference_performance:
    distance_km: 42.195        # flat race distance (marathon recommended)
    time: "2:45:00"            # finish time for that flat race
    notes: "Berlin 2024"       # optional

  gap_curve:
    points: []                 # empty = use default table
    # Custom example (overrides entire default table):
    # - [ 0.01,  1.075 ]
    # - [ 0.20,  2.50  ]  # +20% cutoff anchor
    # - [ 0.00,  1.00  ]
    # - [-0.01,  0.965 ]
    # - [-0.06,  0.8   ]
    # - [-0.07,  0.865 ]
    # - [-0.20,  1.71  ]  # -20% cutoff anchor

  preferences:                 # optional — informational only at present
    threshold_flat_pace_per_km: "3:50/km"
    aerobic_threshold_flat_pace_per_km: "4:40/km"
  
  fatigue_physiology: {}       # optional; reserved for process-based model (TBD)
```

**Notes on the GAP curve**:
- Points are `[grade_decimal, correction_factor]`, e.g. `[0.10, 1.50]` means
  a 10% uphill is 1.5× slower than flat.
- If `points` is empty, `DEFAULT_GAP_CURVE` from `pace_calculator.py` is used.
- The two anchor points at ±20% mark where the constant-vertical-speed tail
  begins; include them in your custom curve.
- Points do **not** need to be sorted — the calculator sorts them internally.

**Notes on fatigue modeling**:
- Fatigue can be controlled via CLI (`--fatigue-mode` and `--fatigue-total-decay-pct`).
- CLI override `--fatigue-total-decay-pct` takes precedence over all config defaults.
- `--fatigue-mode none` (default): no fatigue modeling.
- `--fatigue-mode race`: uses `race.planning.fatigue_total_decay_pct` if present.
- `--fatigue-mode athlete`: reserved for future process-based physiological model.
- Linear model: pace multiplier rises from 1.0 at start to 1.0 + (decay_pct/100) at finish.
- Example: `--fatigue-total-decay-pct 10` means 10% slower pace at the finish.

---

### Race Config (`config/races/*.yaml`)

```yaml
race:
  name: "Tokyo Grand Trail 2026"
  start_time: 15:50            # Race start time in `HH:MM` or `HH:MM:SS`; defaults to `00:00:00`
  distance_km: 161             # informational — actual distance comes from GPX
  elevation_gain_m: 11000      # informational

  gpx_file: "config/gpx_repo/TGT_2025.gpx"   # path relative to project root
  output_file: "results/tgt_2026_segment_analysis.xlsx"

  resample_m: 5                # resample GPX to this interval (metres)
  elevation_tolerance_m: 50    # tolerance for aid station elevation validation

  itra_reference_points:       # optional; enables ITRA prediction
    - reference_time: "21:52:00"
      reference_score: 829
    - reference_time: "32:00:00"
      reference_score: 700
      notes: "estimated"
  
  planning:                    # optional; default planning assumptions
    fatigue_total_decay_pct: 0 # 0 = no fatigue (delegated to athlete config or CLI); or 1–100 for linear decay model

# An empty list is valid — the whole course is treated as one segment.
aid_stations:
  - name: "Start"
    distance_km: 0.0
    elevation_m: 1350
    stop_time_s: 0

  - name: "Checkpoint 1"
    jap_name: "第1CP"          # optional Japanese name (shown in plot)
    distance_km: 22.5
    elevation_m: 1820
    stop_time_s: 300           # 5-minute stop
    notes: "Drop bag available"
    gmaps_link: "https://maps.google.com/?q=..."   # shown in elevation plot

  # ... more stations ...

  - name: "Finish"
    distance_km: 161.0
    elevation_m: 1350
    stop_time_s: 0
```

**Aid station fields**:

| Field | Required | Description |
|---|---|---|
| `name` | yes | English display name |
| `distance_km` | yes | Distance from start |
| `jap_name` | no | Japanese name (displayed in HTML plot) |
| `elevation_m` | no | Reference elevation for validation |
| `stop_time_s` | no | Planned stop in seconds (default 0) |
| `notes` | no | Free text, shown when clicking the marker in the plot |
| `gmaps_link` | no | Google Maps URL linked in the plot |

If `aid_stations` is an empty list, the planner still runs: pacing is computed for the
full course and the smartphone report falls back to the pacing rows it generated.

---

## Outputs

### Excel Workbook (`.xlsx`)

**Sheet: Segment Statistics**

| Column | Description |
|---|---|
| Point Name | Aid station name |
| Total Distance (km) | Cumulative distance from start |
| Elevation (m) | Current elevation |
| Accum. Elevation Gain (m) | Total gain to this point |
| Segment Distance (km) | Distance since previous station |
| Segment Elevation Gain (m) | Gain in this segment |
| Segment Elevation Loss (m) | Loss in this segment |
| Average Gradient (%) | Average positive gradient |

**Sheet: Race Plan (`<mode>`)**

Adds running-time and stop-time columns to the above, plus a summary block:

| Column | Description |
|---|---|
| Segment Running Time | Running time for this segment (H:MM:SS) |
| Avg Pace (mm:ss/km) | Segment average pace by horizontal distance |
| Avg Grade-Adjusted Pace (mm:ss/km) | Segment average pace on GAP-weighted distance |
| Stop Time | Planned stop (H:MM:SS) |
| Elapsed Time | Cumulative elapsed race time at station (H:MM:SS) |

Summary block below the table:

```
Planning mode        athlete_pb
Athlete              Carlos
Total running time   28:01:44
Overall avg pace     10:31
Overall avg grade-adjusted pace 5:14
Total stop time      1:00:00
Total finish time    29:01:44
Predicted ITRA score 752
```

### Interactive Elevation Profile (`.html`)

A self-contained Plotly HTML file with:
- Elevation line with shaded area under the curve
- Aid station markers (gold diamonds) with Japanese names
- Hover tooltips: distance, elevation, grade, cumulative gain
- Click a marker to see its notes and a Google Maps link
- Course statistics box (total distance, gain, loss)
- No external dependencies — opens in any browser offline

### Smartphone Race Plan Report (`_race_plan.html`)

A mobile-oriented HTML report written as `<race>_race_plan.html`, for example
`results/tgt_2026_race_plan.html`.

It includes:
- Sticky top header row and sticky left aid-station column
- English and Japanese aid-station names in the same cell
- Total and split distance in one cell
- Accumulated gain plus split gain/loss columns
- Elapsed, split, and wall-clock time in one timing cell
- Wall-clock times derived from `race.start_time` in the race YAML
- Average split pace and combined comments block
- Embedded elevation profile below the table
- Summary cards for finish time, distance, gain, and start time

---

## Project Structure

```
trail-race-planner/
│
├── race_planner/              # Main package
│   ├── main.py                # CLI entry point
│   ├── course/
│   │   ├── course.py          # GPX loading, resampling, Course class
│   │   └── segment_analyzer.py# Segment stats, elevation validation, Excel export
│   ├── models/
│   │   ├── itra_predictor.py  # ITRA score ↔ time prediction
│   │   ├── itra_score_ratios.py# Empirical ITRA ratio table (UTMB 2025)
│   │   └── tools.py           # Pace/speed/time conversion utilities
│   ├── planner/
│   │   └── pace_calculator.py # PaceCalculator: Riegel + FED + GAP
│   └── visualization/
│       ├── course_profile.py  # Plotly elevation profile generator
│       └── race_plan_table.py # Smartphone race-plan HTML report generator
│
├── config/
│   ├── athletes/              # Athlete YAML profiles
│   │   ├── carlos.yaml
│   │   └── yet_another_sato.yaml
│   ├── gpx_repo/              # GPX course files
│   └── races/                 # Race YAML configs
│       ├── tgt_2026.yaml
│       └── oku_long_2025.yaml
│
├── analysis/                  # Standalone analysis scripts (not part of main pipeline)
│   ├── itra_scores.py         # Fit ITRA score vs. time curve
│   ├── validate_ultra_fatigue_model.py  # Compare Riegel variants against race records
│   └── analyze_gap_vertical_speed_cutoff.py  # GAP curve diagnostic plots
│
├── tests/
│   ├── conftest.py            # Shared fixtures (isolated from live configs)
│   ├── fixtures/              # Test-only YAML copies
│   ├── test_course.py
│   ├── test_pace_calculator.py
│   ├── test_itra_predictor.py
│   ├── test_segment_analyzer.py
│   ├── test_config.py
│   ├── test_race_plan_table.py
│   └── test_tools.py
│
├── results/                   # Generated outputs (xlsx, html)
├── requirements.txt
├── pyproject.toml             # Black config (line-length 88, py39+)
└── pytest.ini
```

---

## Running Tests

```bash
pytest
```

The suite uses isolated test fixtures in `tests/fixtures/` so that edits to live
athlete and race configs do not break tests.

Code style is enforced with [Black](https://github.com/psf/black) and
[Ruff](https://github.com/astral-sh/ruff):

```bash
black .
ruff check .
```
