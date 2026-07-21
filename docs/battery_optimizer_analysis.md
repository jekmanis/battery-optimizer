# Battery Optimizer Main Module Analysis

**File:** `appdaemon/apps/battery_optimizer.py`
**Lines:** ~2057 (after cleanup)
**Last analyzed:** 2026-02-02

## Overview

The `BatteryOptimizer` class is an AppDaemon application that orchestrates battery charge/discharge scheduling based on Nord Pool electricity prices. While the core algorithms have been extracted to the `battery_optimizer_lib` package, this file remains the integration layer responsible for:

- AppDaemon lifecycle management
- Home Assistant entity interactions
- Configuration loading
- Scheduled task orchestration
- Event-driven state change handling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BatteryOptimizer (hass.Hass)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Initialization & Config (~300 lines)                               │
│  ├── initialize() - AppDaemon entry point                           │
│  └── _load_config() - Configuration from apps.yaml                  │
├─────────────────────────────────────────────────────────────────────┤
│  Optimization Algorithm (~320 lines)                                │
│  ├── get_prices() - Delegates to NordPoolPriceService               │
│  ├── calculate_min_charge_slots_for_horizon() - Min slots needed    │
│  ├── find_optimal_schedule() - Creates DPOptimizer, runs DP algo    │
│  ├── calculate_expected_soc_schedule() - SOC trajectory projection  │
│  └── Helper methods for slot/rate computations                      │
├─────────────────────────────────────────────────────────────────────┤
│  Schedule Execution (~530 lines)                                    │
│  ├── full_optimize() - Daily 13:15 + startup optimization           │
│  ├── adaptive_optimize() - Periodic re-evaluation                   │
│  ├── execute_scheduled_mode() - Hourly mode application             │
│  ├── _recalculate_remaining_schedule() - SOC deviation handler      │
│  ├── _check_soc_boundaries() - Safety limits enforcement            │
│  └── _check_soc_deviation() - Deviation detection                   │
├─────────────────────────────────────────────────────────────────────┤
│  Inverter Control                                                   │
│  └── DirectControl.apply_mode() - set_wit_mode service commands     │
├─────────────────────────────────────────────────────────────────────┤
│  Manual Override (~50 lines)                                        │
│  ├── on_override_change() - Override toggle handler                 │
│  ├── on_manual_mode_change() - Mode selection handler               │
│  └── _apply_manual_mode() - Manual mode execution                   │
├─────────────────────────────────────────────────────────────────────┤
│  Event Handlers (~35 lines)                                         │
│  └── _on_soc_change() - SOC state change listener                   │
├─────────────────────────────────────────────────────────────────────┤
│  Battery Cost & Learning (~330 lines)                               │
│  ├── _init_battery_cost() - Cost tracker initialization             │
│  ├── _init_learning_engine() - Learning engine setup                │
│  ├── _init_load_profile() - Load profile initialization             │
│  ├── _save_learning_data() - Persist learning data                  │
│  └── _process_soc_change_event() - Cost tracking on SOC change      │
├─────────────────────────────────────────────────────────────────────┤
│  Helper Methods (~190 lines)                                        │
│  ├── _get_current_soc(), _get_pv_power(), etc. - Sensor reading     │
│  ├── _align_to_slot(), _next_slot_time() - Timing utilities         │
│  ├── _is_enabled(), _is_override_active() - State checks            │
│  └── Properties: min_soc, max_soc, pv_threshold, battery_avg_cost   │
├─────────────────────────────────────────────────────────────────────┤
│  Sensor Updates (~70 lines)                                         │
│  └── _update_schedule_sensor() - Main sensor.battery_optimizer      │
└─────────────────────────────────────────────────────────────────────┘
```

## Delegated Components (from battery_optimizer_lib)

| Component | Class | Purpose |
|-----------|-------|---------|
| **Scheduling** | `DPOptimizer` | Dynamic programming SOC-aware optimization |
| **Prices** | `NordPoolPriceService` | Nord Pool API/sensor price fetching |
| **Inverter Control** | `DirectControl` | Mode commands via `growatt_modbus/set_wit_mode` |
| **Cost Tracking** | `BatteryCostTracker` | Weighted average cost calculations |
| **Learning** | `BatteryLearningEngine` | Charge rate and efficiency learning |
| **Load Profile** | `LoadProfile` | Statistical load forecasting |
| **SOC Deviation** | `SocDeviationDetector` | Schedule deviation detection |
| **Formatting** | `ScheduleFormatter` | Log and sensor output formatting |
| **Sensor Reading** | `SensorReader` | HA state reading helpers |

## Scheduled Tasks

| Trigger | Method | Interval | Purpose |
|---------|--------|----------|---------|
| Daily 14:15 | `full_optimize()` | Once | Full schedule recalculation |
| Startup | `full_optimize()` | Once | Initial optimization |
| Every 30 min | `adaptive_optimize()` | 30 min | PV override, schedule change logging |
| Every slot | `execute_scheduled_mode()` | 30/60 min | Apply scheduled mode |
| Configurable | `record_load_observation()` | 30 min | Load profile data collection |

## Event Listeners

| Entity | Handler | Purpose |
|--------|---------|---------|
| `soc_sensor` | `_on_soc_change()` | Safety checks, cost tracking, deviation detection |
| `battery_charge_sensor` | `_on_energy_sensor_change()` | Energy-based cost tracking |
| `battery_discharge_sensor` | `_on_energy_sensor_change()` | Energy-based cost tracking |
| `override_entity` | `on_override_change()` | Manual override toggle |
| `manual_mode_entity` | `on_manual_mode_change()` | Manual mode selection |

## Key Data Flows

### 1. Full Optimization Flow
```
full_optimize()
  → get_prices() [via NordPoolPriceService]
  → calculate_min_charge_slots_for_horizon()
  → find_optimal_schedule()
      → DPOptimizer.optimize()
  → calculate_expected_soc_schedule()
  → execute_scheduled_mode()
  → _update_schedule_sensor()
```

### 2. SOC Change Event Flow
```
_on_soc_change()
  → _check_soc_boundaries() [safety enforcement]
  → _process_soc_change_event() [cost tracking]
  → _check_soc_deviation()
      → SocDeviationDetector.check_deviation()
      → _recalculate_remaining_schedule() [if needed]
```

### 3. Mode Execution Flow
```
execute_scheduled_mode()
  → _handle_mode_transition() [learning engine tracking]
  → DirectControl.apply_mode()
```

## Configuration Properties

### Static (apps.yaml)
- `battery_capacity_kwh`, `charge_rate_kw`, `discharge_rate_kw`, `efficiency`
- `slot_minutes`, `adaptive_recalc_minutes`, `load_observation_minutes`
- `grid_fee_eur_kwh`, `battery_wear_cost_eur_kwh`
- `nordpool_config_entry`, `nordpool_area`, `nordpool_sensor`
- `device_id` (empty = dry-run)

### Dynamic (HA entities, runtime)
- `min_soc` ← `input_number.battery_min_soc`
- `max_soc` ← `input_number.battery_max_soc`
- `pv_threshold` ← `input_number.battery_pv_threshold`
- `battery_avg_cost` ← `input_number.battery_avg_cost` (persisted)

## Internal State

| Attribute | Type | Purpose |
|-----------|------|---------|
| `schedule` | `Dict[datetime, ScheduleEntry]` | Current schedule |
| `current_mode` | `BatteryMode` | Active mode |
| `expected_soc_schedule` | `Dict[datetime, float]` | Projected SOC trajectory |
| `expected_temp_schedule` | `Dict[datetime, float]` | Projected temp trajectory |
| `_last_recalc_trigger` | `str` | Last recalculation reason |
| `_last_soc_deviation` | `float` | Last deviation that triggered recalc |
| `_last_projected_costs` | `Dict` | Battery cost evolution projection |

---

## Dead Code Analysis

### Summary

| Category | Count | Status |
|----------|-------|--------|
| Unused imports | 1 | **REMOVED** |
| Unused methods | 2 | **REMOVED** |
| Redundant functionality | 1 | **REMOVED** |
| **Total removed** | - | **85 lines** |

**Cleanup performed:** 2026-02-02
**Lines before:** 2142 | **Lines after:** 2057

### 1. ~~Unused Imports~~ REMOVED

The `requests` library and `REQUESTS_AVAILABLE` flag were never used. All HTTP functionality is in `NordPoolPriceService`.

### 2. ~~Unused Methods~~ REMOVED

- **`calculate_charge_hours()`** - Superseded by `calculate_min_charge_slots_for_horizon()`
- **`calculate_discharge_hours()`** - Discharge calculations handled by DP optimizer

### 3. ~~Redundant Functionality~~ REMOVED

**`learning_data_entity`** - HA entity-based persistence for learning data was redundant since file-based persistence (`learning_data_file`) is preferred and more reliable (no 255 char limit).

### 3. Intentionally Unused Parameters

The following `kwargs` parameters are unused but required by AppDaemon's callback interface:

- `full_optimize(self, kwargs=None)` - Line 852
- `adaptive_optimize(self, kwargs=None)` - Line 939
- `execute_scheduled_mode(self, kwargs, force=False)` - Line 1088
- `record_load_observation(self, kwargs=None)` - Line 1806

**Recommendation:** Keep these as-is; they're part of the AppDaemon contract.

### 4. Methods That Appear Unused But Are Wrappers

These private methods wrap library functions and provide convenient access:

- `_align_to_slot()` - Wraps `timezone_utils.align_to_slot()`
- `_next_slot_time()` - Wraps `timezone_utils.next_slot_time()`
- `_next_interval_time()` - Wraps `timezone_utils.next_interval_time()`

**Recommendation:** Keep these; they provide cleaner internal access with self-contained timezone handling.

---

## Cleanup Completed

The following dead/redundant code was removed on 2026-02-02:

| Item | Description |
|------|-------------|
| `requests` import | Unused HTTP library (moved to NordPoolPriceService) |
| `calculate_charge_hours()` | Superseded by `calculate_min_charge_slots_for_horizon()` |
| `calculate_discharge_hours()` | Unused legacy method |
| `learning_data_entity` | Redundant HA entity persistence (file-based preferred) |

**Result:** 2142 → 2057 lines (85 lines removed), all 336 tests pass.
