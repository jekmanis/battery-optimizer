# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Battery Optimizer for Growatt WIT Inverter - a Home Assistant AppDaemon application that uses Nord Pool electricity price forecasts to optimize battery charging/discharging schedules.

## Architecture

### File Structure
```
appdaemon/apps/
├── battery_optimizer.py           # Main AppDaemon app (~2100 lines)
├── battery_optimizer_lib/         # Python package for helper modules
│   ├── __init__.py                # Re-exports for convenience
│   ├── models.py                  # Data classes and enums (~100 lines)
│   ├── learning_engine.py         # BatteryLearningEngine (~790 lines)
│   ├── load_profile.py            # LoadProfile (~110 lines)
│   ├── price_service.py           # NordPoolPriceService (~510 lines)
│   ├── tou_sync.py                # TouSyncManager (~790 lines)
│   ├── dp_optimizer.py            # DPOptimizer (~810 lines)
│   ├── cost_tracker.py            # BatteryCostTracker (~820 lines)
│   ├── schedule_formatter.py      # ScheduleFormatter (~610 lines)
│   ├── soc_deviation.py           # SocDeviationDetector (~420 lines)
│   ├── timezone_utils.py          # Timezone helpers (~300 lines)
│   ├── ha_helpers.py              # HA state reading (~210 lines)
│   └── charge_rate_utils.py       # Temperature-aware rates (~50 lines)
├── apps.yaml                      # AppDaemon configuration
homeassistant/packages/
└── battery_optimizer.yaml         # HA entities, automations, sensors
tests/
├── conftest.py                    # Pytest fixtures
├── fixtures/mock_hass.py          # Mock Home Assistant
└── test_*.py                      # Test modules (~336 tests)
```

### Package Modules (battery_optimizer_lib/)

| Module | Classes/Functions | Purpose |
|--------|-------------------|---------|
| `models.py` | BatteryMode, PricePoint, ScheduleEntry, TouPeriod, LearningStats, LoadProfileStats | Pure data structures and enums |
| `dp_optimizer.py` | DPOptimizer, DPOptimizerConfig, DPOptimizerResult | Dynamic programming SOC-aware scheduling algorithm |
| `learning_engine.py` | BatteryLearningEngine | Self-learning charge rate and efficiency tracking |
| `load_profile.py` | LoadProfile, _quantile | Statistical load forecasting by time-of-day |
| `price_service.py` | NordPoolPriceService | Nord Pool price fetching via HA integration |
| `tou_sync.py` | TouSyncManager | TOU schedule sync and device control via Modbus |
| `cost_tracker.py` | BatteryCostTracker, BatteryCostConfig | Battery cost tracking with weighted average calculations |
| `schedule_formatter.py` | ScheduleFormatter, ScheduleFormatterConfig | Schedule logging and formatting for HA sensors |
| `soc_deviation.py` | SocDeviationDetector, SocDeviationConfig, DeviationCheckResult | Detects unexpected SOC changes for schedule revalidation |
| `timezone_utils.py` | normalize_tz_pair, align_to_slot, dt_ge/dt_gt/dt_lt, etc. | Timezone-aware datetime comparison and slot alignment |
| `ha_helpers.py` | SensorReader, get_float_state, get_bool_state, is_state_valid | Home Assistant state reading helpers |
| `charge_rate_utils.py` | compute_charge_rates_per_slot | Temperature-aware charge rate computation |

### Key Code Sections in battery_optimizer.py

| Section | Purpose |
|---------|---------|
| Initialization | AppDaemon setup, config loading, scheduled tasks |
| Price Fetching | Nord Pool API/sensor data retrieval with caching |
| Price Analysis | Statistics and charge/discharge hour calculations |
| Optimization Algorithm | Dynamic programming SOC-aware scheduling |
| Schedule Execution | Full/adaptive optimization, recalculation |
| Mode Execution | Hourly mode application, safety checks |
| Device Control | Growatt Modbus register writes (VPP protocol) |
| TOU Sync | Schedule sync to inverter TOU registers |
| Manual Override | User intervention handling |
| Battery Cost Tracking | Weighted average cost with persistence |
| Helper Methods | SOC reading, timezone handling, slot alignment |
| Properties & Sensor | Dynamic config, schedule sensor updates |

### Data Models
- `BatteryMode` enum: HOLD (0), CHARGE (1), DISCHARGE (2)
- `PricePoint` dataclass: Hour + price
- `ScheduleEntry` dataclass: Hour + mode + reason
- `TouPeriod` dataclass: Start/end minutes + power percentage
- `LoadProfileStats` dataclass: Min/max/sum/count for load observations
- `LearningStats` dataclass: Charge rate learning data per SOC range

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
- **Every 30 min**: Adaptive re-evaluation + schedule change logging
- **Every 5 min**: Safety checks
- **Hourly**: Mode execution + battery cost update

## Development

This is a Python AppDaemon project. Use `uv` for running Python scripts and syntax checks.

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

### Deployment
```bash
# Copy app to AppDaemon
cp appdaemon/apps/battery_optimizer.py /config/appdaemon/apps/

# Copy library package
cp -r appdaemon/apps/battery_optimizer_lib /config/appdaemon/apps/

# Copy configuration
cp appdaemon/apps/apps.yaml /config/appdaemon/apps/

# Copy HA package
cp homeassistant/packages/battery_optimizer.yaml /config/packages/
```

### Testing
Manual verification via Home Assistant UI and AppDaemon logs. Set `device_id: ""` in apps.yaml for dry-run mode (simulates without controlling inverter).

### Key Sensor Attributes
`sensor.battery_optimizer` exposes:
- `current_mode`, `schedule`, `next_charge`, `next_discharge`
- `battery_avg_cost` - weighted average cost of energy in battery
- `discharge_threshold` - current price threshold for discharge decisions

## Key Dependencies
- AppDaemon 4
- Home Assistant
- Nord Pool integration (built-in or HACS)
- Growatt Modbus integration (for inverter control)
