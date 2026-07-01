# Copilot Instructions for Trail Race Planner

## Core Principles

### 1. Reuse Utilities from `race_planner.models.tools`
Before writing new helper functions, **always check** `race_planner.models.tools` for existing utilities. Common conversions already exist:
- Time conversions: `hms_to_seconds()`, `seconds_to_hms()`, `clock_time_to_seconds()`, `elapsed_hms_to_clock_time()`
- Pace conversions: `pace_to_seconds_per_km()`, `seconds_per_km_to_pace()`, `seconds_per_km_to_mmss()`
- Hour conversions: `hours_to_hms()`, `hms_to_hours()`
- Speed/pace calculations: `pace_to_speed_kmh()`, `vertical_speed_m_per_h()`, `pace_from_constant_vertical_speed()`

Do not duplicate these functions.

### 2. Variable Naming: Always Include Units
All numeric variables **must include their unit as a suffix**. This prevents unit confusion bugs.

**Examples:**
- `distance_km` (not `distance`)
- `time_s` (seconds; not `duration` or `time`)
- `elevation_m` (meters)
- `pace_mmss` (mm:ss format string)
- `pace_sec_per_km` (seconds per kilometer)
- `speed_kmh` (kilometers per hour)
- `weight_kg`, `sweat_loss_ml`, `energy_kcal`, `decay_pct`
- `cumulative_hydration_balance_ml`, `caffeine_concentration_mg_per_kg`

### 3. Type Hints
Always use type hints on function parameters and return values. Use modern Python syntax:
```python
def my_function(value: int, name: str | None = None) -> dict[str, float]:
```

### 4. Docstrings
Include concise docstrings for public functions:
```python
def calculate_pace(distance_km: float, time_s: float) -> float:
    """Calculate pace in seconds per km."""
    if time_s <= 0:
        return 0.0
    return time_s / distance_km
```

### 5. Dataclass ViewModels
Use frozen dataclasses for presentation models (especially UI ViewModels):
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SummaryItemViewModel:
    label: str
    value: str
```

### 6. Private Functions
Functions not meant for external use should start with `_`:
```python
def _helper_function() -> None:
    ...
```

## Code Organization

### Module Structure
- **`course/`** — GPX parsing, course analysis, segment analysis
- **`models/`** — Business logic: pacing, nutrition, ITRA prediction, tools (utilities)
- **`planner/`** — Pace calculator and race planning
- **`visualization/`** — HTML reports, plots, table generation

### Import Order
1. `from __future__ import annotations`
2. Standard library imports
3. Third-party imports (pandas, loguru, openpyxl, etc.)
4. Local race_planner imports

## Common Patterns

### Configuration Loading
Race and athlete configs are loaded from YAML in the project root:
- `config/races/*.yaml` — race definitions (gpx_file, output_file, aid stations, itra_reference_points, etc.)
- `config/athletes/*.yaml` — athlete profiles (weight_kg, reference performance, hydration, etc.)

### Logging
Use loguru for all logging:
```python
from loguru import logger
logger.info("Message")
logger.success("Operation completed")
logger.error("Error occurred")
```

### DataFrames (Pandas)
Pacing plans are computed as DataFrames with columns like:
- `"Point Name"`, `"Total Distance (km)"`, `"Elapsed Time"`
- `"Avg Pace (mm:ss/km)"`, `"Avg Grade-Adjusted Pace (mm:ss/km)"`
- Frame attributes: `pacing_df.attrs` stores metadata (total_time_s, total_running_time_s, fatigue_total_decay_pct, etc.)

### Excel Export
Use openpyxl to append sheets to existing workbooks. Always set column widths for readability.

## Testing

### Test Location
Tests are in `tests/` with fixtures in `tests/fixtures/`:
- `tests/test_*.py` — unit tests corresponding to `race_planner/` modules
- `tests/conftest.py` — pytest configuration and shared fixtures

### Naming Convention
Test functions: `test_<function_or_feature>_<condition>`
```python
def test_pace_calculator_with_fatigue():
    ...
```

## Performance Considerations

- **Course resampling** is configurable (default 5 m intervals) via `resample_m` in race config
- **DataFrames** should use proper dtypes to minimize memory usage
- **Plotly figures** are embedded in HTML — keep embedded profiles reasonably sized (~430px height)

## Special Notes

### Fatigue Modeling
Fatigue decay is a percentage applied linearly to running pace. It's configurable per race or athlete:
- `fatigue_total_decay_pct` in config or CLI
- Applied via `fatigueRatio(progress, decayPct)` in pacing calculations

### HTML Reports
The smartphone-friendly race plan report includes:
- Embedded Plotly elevation profile (dark theme)
- Interactive time/pace adjustment via JavaScript
- Sticky table header and first column
- Responsive design for mobile

## When Adding New Features

1. **Check existing utilities** in `tools.py` before writing helpers
2. **Use units in variable names** — no ambiguity
3. **Add type hints** — full coverage
4. **Write docstrings** for public functions
5. **Test with fixtures** — use athlete/race configs in `tests/fixtures/`
6. **Log progress** — use loguru
7. **Update this file** if introducing new patterns

## LLM Behavioral Guidelines (Karpathy-Inspired)

These behavior rules are adapted from Karpathy-inspired `CLAUDE.md` guidance and merged for this project.
Source reference: https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md

### 1. Think Before Coding
- State assumptions explicitly before implementation.
- If there are multiple valid interpretations, present them instead of choosing silently.
- If requirements are unclear, stop and ask a targeted clarifying question.
- If a simpler approach is available, call it out.

### 2. Simplicity First
- Implement only what was requested.
- Avoid speculative abstractions, options, and configuration knobs.
- Avoid handling impossible scenarios unless requested.
- Prefer minimal code paths that are easy to verify.

### 3. Surgical Changes
- Keep diffs tightly scoped to the request.
- Do not refactor unrelated areas while implementing a task.
- Match existing file style and local conventions.
- Remove only unused code/imports introduced by your own changes.

### 4. Goal-Driven Execution
- Convert each task into explicit, testable success criteria.
- For fixes, reproduce first (or define a failing condition), then verify resolution.
- For multi-step tasks, keep a short step/check plan and verify each step.
- Finish only after validation (tests, checks, or concrete output evidence).

### Practical Tradeoff
- These rules prioritize correctness and clarity over raw speed.
- For trivial changes, apply judgment and keep overhead low.
