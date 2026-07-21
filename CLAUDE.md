# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Battery Optimizer for Growatt WIT Inverter - a Home Assistant AppDaemon application that uses Nord Pool electricity price forecasts to optimize battery charging/discharging schedules.

## Architecture

### File Structure
```
appdaemon/apps/
├── battery_optimizer.py           # Main AppDaemon app (orchestrator)
├── battery_optimizer_lib/         # Python package for helper modules
│   ├── __init__.py                # Re-exports all public classes
│   ├── config.py                  # BatteryOptimizerConfig dataclass
│   ├── models.py                  # Data classes and enums
│   ├── dp_optimizer.py            # Dynamic programming optimizer
│   ├── learning_engine.py         # Self-learning charge rate tracker
│   ├── load_profile.py            # Statistical load forecasting
│   ├── pv_profile.py              # Statistical PV production profile
│   ├── pv_forecast_service.py     # PV forecast fetching (Solcast / Forecast.Solar)
│   ├── price_service.py           # Nord Pool price fetching
│   ├── direct_control.py          # Direct inverter control via set_wit_mode
│   ├── cost_tracker.py            # Battery cost tracking
│   ├── schedule_formatter.py      # Schedule logging/formatting
│   ├── soc_deviation.py           # SOC deviation detection
│   ├── load_prediction_tracker.py # Predicted vs actual load accuracy
│   ├── slot_outcome_tracker.py    # Per-slot outcome/compliance tracking
│   ├── timezone_utils.py          # Timezone-aware datetime helpers
│   ├── ha_helpers.py              # HA state reading helpers
│   └── charge_rate_utils.py       # Temperature-aware rate computation
├── apps.yaml                      # AppDaemon configuration (contains secrets!)
homeassistant/packages/
└── battery_optimizer.yaml         # HA entities, automations, sensors
tests/
├── conftest.py                    # Pytest fixtures + mock AppDaemon setup
└── test_*.py                      # Test modules
```

### Package Modules (battery_optimizer_lib/)

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `config.py` | BatteryOptimizerConfig | Typed config dataclass with `from_args()` loader |
| `models.py` | BatteryMode, PricePoint, ScheduleEntry | Pure data structures and enums |
| `dp_optimizer.py` | DPOptimizer, DPOptimizerConfig, DPOptimizerResult | Dynamic programming SOC-aware scheduling |
| `learning_engine.py` | BatteryLearningEngine | Self-learning charge rate and efficiency tracking |
| `load_profile.py` | LoadProfile | Statistical load forecasting by time-of-day |
| `pv_profile.py` | PvProfile | Statistical PV production profile by time-of-day |
| `pv_forecast_service.py` | PvForecastService, PvForecastServiceConfig | PV forecast fetching (Solcast / Forecast.Solar) |
| `price_service.py` | NordPoolPriceService | Nord Pool price fetching (built-in HA + HACS) |
| `direct_control.py` | DirectControl | Direct inverter control via `growatt_modbus/set_wit_mode` |
| `cost_tracker.py` | BatteryCostTracker, BatteryCostConfig | Battery cost tracking with weighted averages |
| `schedule_formatter.py` | ScheduleFormatter, ScheduleFormatterConfig | Schedule logging and HA sensor formatting |
| `soc_deviation.py` | SocDeviationDetector, SocDeviationConfig | Detects unexpected SOC changes for revalidation |
| `load_prediction_tracker.py` | LoadPredictionTracker | Predicted vs actual load accuracy tracking |
| `slot_outcome_tracker.py` | SlotOutcomeTracker | Per-slot outcome and mode compliance tracking |
| `timezone_utils.py` | normalize_tz_pair, align_to_slot, lookup_by_time, dt_ge | Timezone-aware datetime comparison and alignment |
| `ha_helpers.py` | SensorReader | HA state reading with validation |
| `charge_rate_utils.py` | compute_charge_rates_per_slot | Temperature-aware charge rate computation |

### Data Models
- `BatteryMode` enum: HOLD (0), CHARGE (1), DISCHARGE (2)
- `PricePoint` dataclass: Time (datetime) + price
- `ScheduleEntry` dataclass: Time (datetime) + mode + reason + direct-control fields (export_rate, ac_charge_mode, power_percent, SOC cutoffs)
- `LoadProfileStats` dataclass: Min/max/sum/count for load observations
- `LearningStats` dataclass: Charge rate learning data per SOC range

### Slot Resolution
- Default `slot_minutes=15` (96 slots/day) — matches Nord Pool 15-minute pricing periods
- Configurable via `apps.yaml` (`slot_minutes: 15`)
- Price service requests 15-min resolution from Nord Pool (`resolution` parameter)
- `_normalize_prices()` handles expansion if source data is coarser (e.g., hourly → 4x15min)
- Load profile supports migration from coarser buckets (30-min → 15-min) on first load
- A local day is not always 96 slots: Europe/Riga spring/autumn transitions produce 23/25-hour days. Internally, aware timestamps are keyed, sorted, and compared as UTC instants so the two autumn `03:00` intervals remain distinct; local time is for prediction and presentation.

## Core Algorithm

### Scheduling Logic (`DPOptimizer.optimize`)
Uses **dynamic programming** with SOC state tracking:
1. Discretize the configured SOC range using `soc_step_percent`
2. For each time slot, evaluate HOLD/CHARGE/DISCHARGE transitions
3. Track the best cumulative economic value for each reachable energy level
4. Backtrack to extract optimal action sequence

**Value calculations:**
- Marginal import price: `(spot_price + grid_fee) * import_price_multiplier`
- Net load: `max(0, predicted_load - predicted_pv)`; PV serves load before battery or grid energy
- Grid CHARGE cost uses AC imported energy. Stored battery energy includes the configured storage efficiency, while grid AC-to-DC conversion also applies `inverter_efficiency`; simultaneous PV surplus reduces the grid contribution.
- DISCHARGE serves `min(net_load, discharge_rate)` on the AC side. The SOC transition consumes additional DC energy for inverter loss, and battery wear is charged per discharged DC kWh.
- PV surplus can charge the battery in HOLD/CHARGE and remaining surplus earns `max(0, spot * export_rate_multiplier - grid_export_fee)`.

`efficiency` is the charge-retention factor, not a complete round-trip figure. `inverter_efficiency` applies on grid AC-to-DC charging and battery DC-to-AC discharge, so the modeled grid-charge round trip is approximately `efficiency * inverter_efficiency^2`.

`min_charge_slots_required` is reporting-only: it estimates the aggregate energy deficit but does not constrain the DP. Feasibility comes from SOC-state transitions and power limits.

At the price-horizon boundary, `terminal_energy_value_eur_kwh` values usable stored DC energy. `auto` derives a conservative value from the median forecast import price, discharge conversion, and wear; `0` restores legacy end-of-horizon depletion behavior. It is a salvage value, not a hard terminal-SOC target.

### Inverter Control (DirectControl)
The schedule is executed by sending mode commands to the Growatt WIT inverter
via the `growatt_modbus/set_wit_mode` HA service (no raw register writes):
- Modes: `grid_charge`, `discharge_to_load`, `discharge_to_grid`, `max_export`, `hold`, `passthrough`
- Each command carries power_percent, duration, export_rate, ac_charge_mode, and SOC cutoffs
- AC charge mode auto-selects `pv_priority` vs `ac_priority` based on current PV power
- Duplicate commands within half a slot are skipped; `release_control()` reverts to `passthrough`
- Dry-run mode: `device_id: ""` in apps.yaml logs commands without sending them

### Battery Cost Tracking
- **Units**: `battery_avg_cost` is landed EUR per stored DC kWh, not raw spot price
- **Weighted average**: `(old_stored_energy * old_landed_cost + added_stored_energy * added_landed_cost) / total_stored_energy`
- **Grid charging**: landed cost includes `(spot + grid_fee) * import_price_multiplier` and AC-to-stored-DC conversion losses
- **PV charging**: landed cost is the foregone net export revenue per stored DC kWh; PV is not booked at the grid purchase price
- **SOC-based tracking**: Measures actual SOC changes, not theoretical charging
- **Discharging**: Reduces stored energy without changing its per-kWh average
- **Persistence**: Stored in `input_number.battery_avg_cost` (survives restarts)

The tracker is an operational estimate because aggregate inverter counters may not identify every mixed PV/grid contribution. Projected tracking must use the same load and PV predictors as the schedule. The DP itself optimizes forecast cash flows and does not use `battery_avg_cost` as a charge-count constraint or primary objective.

### Dynamic Configuration
These values read from HA entities at runtime (adjustable without restart):
- `input_number.battery_min_soc` -> min_soc
- `input_number.battery_max_soc` -> max_soc
- `input_number.battery_pv_threshold` -> pv_threshold
- `input_number.battery_avg_cost` -> landed battery-cost persistence
- `input_number.battery_cost_basis_version` -> one-time legacy raw-cost migration marker

### Scheduled Tasks
- **13:15 daily**: Full optimization (after Nord Pool prices publish)
- **Startup**: Initial optimization
- **Every 15 min**: Adaptive re-evaluation + schedule change logging
- **Every 5 min**: Safety checks
- **Hourly**: Mode execution + battery cost update

## Development

This is a Python AppDaemon project. Use `uv` for running Python scripts and syntax checks. No formatter or linter is enforced.

**Shell note**: Even though the platform is Windows, the shell is bash. Don't use Windows-specific syntax like `cd /d`. The working directory is already set, so run commands directly without `cd`.

```bash
# Check syntax
uv run python -m py_compile appdaemon/apps/battery_optimizer.py

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_algorithm.py -v

# Run a single test function
uv run pytest tests/test_algorithm.py::TestFindOptimalSchedule::test_basic_charge_discharge_pattern -v

# Run tests with coverage
uv run pytest tests/ --cov=appdaemon/apps --cov-report=term-missing
```

### Testing Architecture
- `conftest.py` mocks the entire `appdaemon.plugins.hass.hassapi` module before any imports, so library modules can be tested without AppDaemon installed.
- Library modules in `battery_optimizer_lib/` are tested directly. The main `battery_optimizer.py` (AppDaemon orchestrator) is not unit-tested — it's validated via dry-run mode (`device_id: ""` in apps.yaml).
- `apps.yaml` contains a long-lived HA access token — do not commit changes to it carelessly.

## Key Dependencies
- AppDaemon 4, Home Assistant, Nord Pool integration, Growatt Modbus integration
