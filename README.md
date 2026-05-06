# trail-race-planner

A Python tool for trail ultra-marathon race planning. Given a GPX course file, an
athlete profile, and a race configuration, it computes a segment-by-segment pacing
plan that accounts for elevation, terrain, and cumulative fatigue — then exports the
results to an interactive HTML elevation profile and an Excel workbook.

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
git clone https://github.com/your-user/trail-race-planner.git
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
| Elapsed Time | Cumulative wall-clock time at station (H:MM:SS) |

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
│       └── course_profile.py  # Plotly elevation profile generator
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

```
81 passed in ~3s
```

Code style is enforced with [Black](https://github.com/psf/black) and
[Ruff](https://github.com/astral-sh/ruff):

```bash
black .
ruff check .
```
