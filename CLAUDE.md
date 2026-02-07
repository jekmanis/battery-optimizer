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
│   ├── price_service.py           # Nord Pool price fetching
│   ├── tou_sync.py                # TOU schedule sync via Modbus
│   ├── cost_tracker.py            # Battery cost tracking
│   ├── schedule_formatter.py      # Schedule logging/formatting
│   ├── soc_deviation.py           # SOC deviation detection
│   ├── timezone_utils.py          # Timezone-aware datetime helpers
│   ├── ha_helpers.py              # HA state reading helpers
│   └── charge_rate_utils.py       # Temperature-aware rate computation
├── apps.yaml                      # AppDaemon configuration (contains secrets!)
homeassistant/packages/
└── battery_optimizer.yaml         # HA entities, automations, sensors
tests/
├── conftest.py                    # Pytest fixtures + mock AppDaemon setup
├── fixtures/mock_hass.py          # Mock Home Assistant
└── test_*.py                      # Test modules
```

### Package Modules (battery_optimizer_lib/)

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `config.py` | BatteryOptimizerConfig | Typed config dataclass with `from_args()` loader |
| `models.py` | BatteryMode, PricePoint, ScheduleEntry, TouPeriod | Pure data structures and enums |
| `dp_optimizer.py` | DPOptimizer, DPOptimizerConfig, DPOptimizerResult | Dynamic programming SOC-aware scheduling |
| `learning_engine.py` | BatteryLearningEngine | Self-learning charge rate and efficiency tracking |
| `load_profile.py` | LoadProfile | Statistical load forecasting by time-of-day |
| `price_service.py` | NordPoolPriceService | Nord Pool price fetching (built-in HA + HACS) |
| `tou_sync.py` | TouSyncManager | TOU schedule sync and Modbus device control |
| `cost_tracker.py` | BatteryCostTracker, BatteryCostConfig | Battery cost tracking with weighted averages |
| `schedule_formatter.py` | ScheduleFormatter, ScheduleFormatterConfig | Schedule logging and HA sensor formatting |
| `soc_deviation.py` | SocDeviationDetector, SocDeviationConfig | Detects unexpected SOC changes for revalidation |
| `timezone_utils.py` | normalize_tz_pair, align_to_slot, lookup_by_time, dt_ge | Timezone-aware datetime comparison and alignment |
| `ha_helpers.py` | SensorReader | HA state reading with validation |
| `charge_rate_utils.py` | compute_charge_rates_per_slot | Temperature-aware charge rate computation |

### Data Models
- `BatteryMode` enum: HOLD (0), CHARGE (1), DISCHARGE (2)
- `PricePoint` dataclass: Time (datetime) + price
- `ScheduleEntry` dataclass: Time (datetime) + mode + reason
- `TouPeriod` dataclass: Start/end minutes + power percentage
- `LoadProfileStats` dataclass: Min/max/sum/count for load observations
- `LearningStats` dataclass: Charge rate learning data per SOC range

### Slot Resolution
- Default `slot_minutes=15` (96 slots/day) — matches Nord Pool 15-minute pricing periods
- Configurable via `apps.yaml` (`slot_minutes: 15`)
- Price service requests 15-min resolution from Nord Pool (`resolution` parameter)
- `_normalize_prices()` handles expansion if source data is coarser (e.g., hourly → 4x15min)
- Load profile supports migration from coarser buckets (30-min → 15-min) on first load

## Core Algorithm

### Scheduling Logic (`DPOptimizer.find_optimal_schedule`)
Uses **dynamic programming** with SOC state tracking:
1. Discretize SOC into energy levels (0.1 kWh steps)
2. For each time slot, evaluate HOLD/CHARGE/DISCHARGE transitions
3. Track best value for each (charge_count, energy_level) state
4. Backtrack to extract optimal action sequence

**Value calculations:**
- CHARGE cost: `(price + grid_fee) * energy / efficiency`
- DISCHARGE value: `(price + grid_fee) * energy` (avoided import cost)
- Discharge is modeled as self-consumption: `min(predicted_load, discharge_rate)`

### TOU Sync to Inverter
Syncs schedule to inverter's TOU registers for autonomous operation:
- Consolidates contiguous same-mode slots into periods
- Includes today AND tomorrow's schedule (time-of-day based)
- Today's entries take precedence for conflicting time slots
- Maximum 20 periods supported by inverter

### Growatt Modbus Quirks (VPP Protocol)
**Critical**: Register write order for TOU sync:
1. Set `30100=1` (VPP control authority)
2. Set `30410=1` (AC charging enable - required for charge periods!)
3. Set `30407=0` (Disable remote control - so TOU takes precedence!)
4. Set `30411=0` (Clear existing schedule)
5. Zero out all period registers (30412+ for periods to write) - **stale data causes overlap validation failures!**
6. Write periods SEQUENTIALLY:
   - Write period 1 data (30412-30414), set `30411=1`
   - Write period 2 data (30415-30417), set `30411=2`
   - ... repeat for each period
7. Verify final num_periods

**Key insights**:
- If `30407=1` (remote control enabled), it overrides TOU schedule!
- Period 1 MUST start at 00:00 (register 30412 must be 0)
- Zeroed registers [0,0,0] are treated as "empty" and allow writes
- Non-zero stale data causes firmware overlap validation to reject writes

**Key Registers:**
- 30100: VPP Control Authority (1=enable)
- 30407: Remote Power Control (0=disable for TOU, 1=enable for manual control)
- 30409: Remote Power Percent (-100 to +100) - only used when 30407=1
- 30410: AC Charging Enable (1=PV first) - REQUIRED for charge periods
- 30411: Number of TOU periods (0-20) - must be set BEFORE writing period data
- 30412+: TOU period data (3 registers each: start, end, power)

### Battery Cost Tracking
- **Weighted average**: `(old_energy * old_cost + new_energy * charge_price) / total_energy`
- **SOC-based tracking**: Measures actual SOC changes, not theoretical charging
- **Persistence**: Stored in `input_number.battery_avg_cost` (survives restarts)

### Dynamic Configuration
These values read from HA entities at runtime (adjustable without restart):
- `input_number.battery_min_soc` -> min_soc
- `input_number.battery_max_soc` -> max_soc
- `input_number.battery_pv_threshold` -> pv_threshold
- `input_number.battery_avg_cost` -> battery cost persistence

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
