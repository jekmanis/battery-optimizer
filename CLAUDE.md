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
│   ├── pv_bias_tracker.py         # Sliding PV forecast bias + slot-energy sampling
│   ├── price_service.py           # Nord Pool price fetching
│   ├── direct_control.py          # Direct inverter control via set_wit_mode
│   ├── cost_tracker.py            # Battery cost tracking
│   ├── schedule_formatter.py      # Schedule logging/formatting
│   ├── thermal_model.py           # Shared battery temperature model (k1/k2)
│   ├── ambient_service.py         # Time-varying ambient temperature T_ambient(t)
│   ├── soc_projection.py          # Shared slot SOC transition model
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
| `pv_bias_tracker.py` | PvBiasTracker, PvBiasConfig, ClosedSlot | Slot-energy PV sampling and sliding actual/forecast bias |
| `price_service.py` | NordPoolPriceService | Nord Pool price fetching (built-in HA + HACS) |
| `direct_control.py` | DirectControl | Direct inverter control via `growatt_modbus/set_wit_mode` |
| `cost_tracker.py` | BatteryCostTracker, BatteryCostConfig | Battery cost tracking with weighted averages |
| `schedule_formatter.py` | ScheduleFormatter, ScheduleFormatterConfig | Schedule logging and HA sensor formatting |
| `thermal_model.py` | TemperatureProjector, step_temperature, battery_power_for_entry | The ONE battery temperature model. Warming depends on `\|P_bat\|`, never on the scheduled mode |
| `ambient_service.py` | AmbientTemperatureService, AmbientServiceConfig | `T_ambient(t)` across the horizon: weather forecast → outdoor sensor → diurnal profile |
| `soc_projection.py` | project_slot_soc, SocProjectionParams | Single slot-SOC transition model shared by the expected-SOC trajectory and the deviation detector (the DP keeps its own inlined transition; `tests/test_soc_projection.py` guards that they agree) |
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

**One slot-SOC model.** The expected-SOC trajectory, the SOC deviation detector
and the schedule log's fallback trajectory
(`ScheduleFormatter._format_expected_trajectory`) must all go through
`soc_projection.project_slot_soc` — never re-implement the transition locally.
The formatter was the third offender: its HOLD line ignored PV surplus charging
(`end_soc = start_soc`) and its DISCHARGE line drained `min(load, rate)` from
raw load without subtracting PV or dividing by `inverter_efficiency`, so a sunny
slot printed 50.0 %→48.6 % where the shared model gives 55.3 % — a contradiction
inside the very log used to diagnose SOC deviations. Its invariants (partial
first slot,
`pv >= load` during DISCHARGE/HOLD is PV *charging*, export slots use the export
discharge rate, DC-side energy moves the SOC, mid-slot anchoring) are documented
in `docs/scheduling-algorithm.md` § SOC transitions and discretization.
Divergence here caused a production recalculation loop, not a threshold problem.

**One thermal model.** Battery temperature is projected only by
`thermal_model.TemperatureProjector`, shared by the DP trajectory, the
expected-SOC trajectory (`soc_projection`), the schedule formatter and
`charge_rate_utils`. Two invariants must not be broken:

1. Warming is a function of `|P_bat|`, not of the mode — discharging heats the
   pack. Never reintroduce a `mode == CHARGE` branch in a temperature path;
   derive power with `thermal_model.battery_power_for_entry`.
2. Ambient is `T_ambient(t)` from `ambient_service`, never one scalar for the
   whole horizon. In the no-external-source fallback, the learning engine's
   rolling battery minimum anchors the diurnal profile's daily **maximum**, not
   its minimum: the pack is self-heated, so `T_bat >= T_ambient` always and
   `min(T_bat)` is a *ceiling* on ambient. Anchoring it as the trough and adding
   the amplitude put the profile peak at `min + 2A` — an "ambient" hotter than
   the battery, which made `TemperatureProjector` warm an idle pack (33.0 →
   34.6 C over 3 h at 0 kW) and made `record_cooling` reject every summer
   cooling sample as `temp_end < ambient`.

The temperature trajectory in `DPOptimizerResult` is reporting-only; temperature
changes DP decisions solely through `compute_charge_rates_per_slot`. Details,
the `k1`/`k2` units and the calibration/bootstrap rules are in
`docs/scheduling-algorithm.md` § Thermal model.

`min_charge_slots_required` is reporting-only: it estimates the aggregate energy deficit but does not constrain the DP. Feasibility comes from SOC-state transitions and power limits.

At the price-horizon boundary, `terminal_energy_value_eur_kwh` values usable stored DC energy. `auto` derives a conservative value from the median forecast import price, discharge conversion, and wear. It is a salvage value, not a hard terminal-SOC target.

`0` is **no-salvage mode**: it declares stored energy worthless at the horizon, so every plan ends by spending it (`EXPORT (until depleted) -> min_soc`). This is harmless while the daily re-optimization extends the horizon before those slots execute. Which mode is active is STATED, not warned about — once at config load (`config.TERMINAL_VALUE_ZERO_NOTICE`, also emitted from `log_summary`) and rate-limited from the DP (`DPOptimizer(warn_degenerate_terminal=...)`, gated by `_should_warn_degenerate_terminal()` to once per 6 h), all at INFO. Both settings have a real failure mode, so the code must not prescribe one: `0` risks spending the battery at the horizon edge, `"auto"` risks stranding charge there and skipping evening slots priced below the median. On the reference installation `"auto"` was tried and reverted: it stranded ~77% SOC at the horizon edge and skipped evening slots priced below the median, which cost more than the end-of-horizon spend it prevented. Do not re-add the old INFO line "net-load slots worth less than this are HELD" for the zero case — nothing is worth less than zero, so it described a rule that could never fire.

**The schedule log's value column is `ScheduleEntry.marginal_value_eur_kwh`, not the cost basis.** The DP fills it (with `value_basis` ∈ `avoided-import` / `export` / `landed-charge` / `kept`) from the same `_buy_price`/`_sell_price` arithmetic it scores slots with, normalized to one battery DC kWh. It is REPORTING ONLY — the DP objective never reads it, and `tests/test_schedule_value_column.py::test_marginal_value_does_not_change_the_schedule` guards that. Any new tariff formula must go through `_buy_price`/`_sell_price` so the report cannot drift from the objective.

### Inverter Control (DirectControl)
The schedule is executed by sending mode commands to the Growatt WIT inverter
via the `growatt_modbus/set_wit_mode` HA service (no raw register writes):
- Modes: `grid_charge`, `discharge_to_load`, `discharge_to_grid`, `max_export`, `hold`, `passthrough`
- Each command carries power_percent, duration, export_rate, ac_charge_mode, and SOC cutoffs
- AC charge mode auto-selects `pv_priority` vs `ac_priority` based on current PV power
- Duplicate commands within half a slot are skipped; `release_control()` reverts to `passthrough`
- Reliability: each call passes `hass_timeout=config.set_wit_mode_timeout_seconds` (default **15**) and inspects the service response. A raised/`success=False` result is a confirmed failure (ERROR, returns False, last-sent NOT recorded so it retries next slot). A `None` result is an unconfirmed client-side timeout (WARNING, last-sent recorded to avoid schedule spam). The timeout is short *because* the None path is safe: verify-after-set catches a genuinely lost command, whereas a long timeout blocks every other callback of this app.
- Verify-after-set is a **bounded two-step ladder**, max 2 checks and 2 sends per `apply_mode` — never a resend loop:
  1. after `verify_delay_seconds` (default 90) the Inverter Mode sensor (`sensor.growatt_inverter_mode` default, or `inverter_mode_sensor`) is read; a genuine mismatch → WARNING + resend once (bypassing duplicate suppression) + schedule check 2;
  2. after `verify_recheck_seconds` (default 60): match → INFO "recovered after resend"; mismatch → **ERROR** and stop (the next slot retries).

  The re-check exists to separate a lagging HA modbus sensor from an inverter that genuinely drops the override; without it the 30 logged mismatches carried no evidence either way. `DirectControl.get_diagnostics()` exposes the counters (`mismatch_count`, `resend_count`, `resend_recovered_count`, `resend_failed_count`, `persistent_mismatch_count`, `unverifiable_count`) on `sensor.battery_inverter_control_health` and as an attribute of `sensor.battery_optimizer`. A new `apply_mode` supersedes any pending verification timer.
- Dry-run mode: `device_id: ""` in apps.yaml logs commands without sending them

### Battery Cost Tracking
- **Units**: `battery_avg_cost` is landed EUR per stored DC kWh, not raw spot price
- **Weighted average**: `(old_stored_energy * old_landed_cost + added_stored_energy * added_landed_cost) / total_stored_energy`
- **Grid charging**: landed cost includes `(spot + grid_fee) * import_price_multiplier` and AC-to-stored-DC conversion losses
- **PV charging**: landed cost is the foregone net export revenue per stored DC kWh; PV is not booked at the grid purchase price
- **SOC-based tracking**: Measures actual SOC changes, not theoretical charging
- **Discharging**: Reduces stored energy without changing its per-kWh average
- **Persistence**: Stored in `input_number.battery_avg_cost` (survives restarts)
- **Accumulator resync**: `_stored_energy_kwh` is a running total of measured inverter deltas, and it drifts (deltas < 0.05 kWh dropped as noise, midnight counter resets, unmodelled losses). `_resync_stored_energy()` re-anchors it to the SOC-derived value when the battery was empty *before* the event, or as a coarse safety net past a wide drift tolerance (`max(2 kWh, 25% of capacity)`). The depletion case is the one that matters: without it, a charge following a real depletion was averaged against phantom stored energy, so a degenerate 0.0000 basis survived a full depletion (logged 2026-07-28). The drift tolerance is deliberately several charge slots wide — the measured accumulator is the better weighting signal and must not be yanked around by the 1%-granular SOC sensor.
- **A PV cost basis of 0.0000 is correct, not a bug**: PV is booked at foregone net export revenue, and around midday `spot * export_multiplier - export_fee` is genuinely ≤ 0. That is why the schedule log shows the DP's marginal slot value as the primary number and the stored basis as a secondary one.

The tracker is an operational estimate because aggregate inverter counters may not identify every mixed PV/grid contribution. Projected tracking must use the same load and PV predictors as the schedule. The DP itself optimizes forecast cash flows and does not use `battery_avg_cost` as a charge-count constraint or primary objective.

Exact cost formulas, the mode-based source attribution table, slot pricing of energy deltas, and how to interpret the charge log are documented in `docs/scheduling-algorithm.md` § Battery cost tracking.

### Dynamic Configuration
These values read from HA entities at runtime (adjustable without restart):
- `input_number.battery_min_soc` -> min_soc
- `input_number.battery_max_soc` -> max_soc
- `input_number.battery_pv_threshold` -> pv_threshold
- `input_number.battery_avg_cost` -> landed battery-cost persistence
- `input_number.battery_cost_basis_version` -> one-time legacy raw-cost migration marker

### PV forecast bias

**PV shortfall is measured on completed slots, never on a boundary reading.**
`pv_bias_tracker.PvBiasTracker` samples PV power every `pv_sample_seconds` (60s
default) and closes each slot with its *mean* power (= slot energy / slot
hours).  The reactive recalculation requires `pv_reactive_consecutive_slots`
(default 2) consecutive closed slots below `pv_reactive_threshold`, each backed
by at least `pv_reactive_min_samples` readings.  A single instantaneous read
taken at a slot boundary is physically incapable of representing the slot
average (ramps, cloud shadow, sensor latency) and caused 43 recalculations in
33 hours of production logs.

**The streak counts shortfalls since the last recalculation.**
`_check_pv_shortfall` calls `PvBiasTracker.reset_shortfall_streak()` after it
triggers.  `_register_closed` only ever clears the streak on a *good* slot, so
without the reset persistent cloud cover made it grow 2, 3, 4, 5 … and the
`streak < pv_reactive_consecutive_slots` guard could never hold again — every
following slot paid for a full `_recalculate_remaining_schedule` on the same
AppDaemon thread the `_timed_callback` instrumentation warns about. The reset
bounds the cadence at one recalculation per `pv_reactive_consecutive_slots`
slots.

**The bias correction covers the whole remaining horizon, attenuated across day
boundaries.** `get_factor()` returns a clamped
(`pv_bias_min_factor`..`pv_bias_max_factor`) median of `measured / forecast`
over `pv_bias_window_minutes`, relaxing back to 1.0 over `pv_bias_decay_slots`
when observations go stale.  `_predict_pv_kw` multiplies every current/future
slot by it — `_predict_pv_kw_raw` is the un-corrected provider value and MUST be
the one fed back into the tracker, otherwise the bias would feed on itself.
Slots on a *later local day* go through `PvBiasTracker.factor_for_slot`, which
keeps only `pv_bias_next_day_weight ** days` of the deviation from 1.0 and
floors the result at `pv_bias_next_day_min_factor`.  Today's cloud cover is
weather, not a calibration error of tomorrow's forecast: the daily 13:15 run
plans ~33 h ahead, and undamped it scaled all of tomorrow to the 0.2 clamp,
systematically shifting the DP onto paid grid charging.  The `decay_slots`
relaxation cannot substitute for this — it only starts once observations go
stale, which does not happen within a day.  The forecast snapshot per slot is
written once (`ensure_slot_forecast`, first write wins) because
`PvForecastService.refresh_for_shortfall` caps the cached current slot at the
observed production, which would otherwise make the ratio read ~1.0.

### Runtime constraints

**This app needs more than one AppDaemon thread.** `set_wit_mode` is a
synchronous, blocking service call made from a callback, so on the default
single thread one slow inverter write stalls schedule execution, the SOC
listener and PV sampling alike (production: 70 × "Excessive time spent in
callback (limit=10.0s)" at 10–34 s, all on `thread-0`). Set
`appdaemon: total_threads: 4` in `appdaemon.yaml` (or pin the app). The app
instruments its own callbacks via the `_timed_callback` decorator and
`_record_callback_duration()`, warning above `callback_warn_seconds` and
repeating the `total_threads` advice once after three overruns. The decorator
must keep `functools.wraps` + `*args/**kwargs`: AppDaemon calls these
positionally (`execute_scheduled_mode(kwargs, force=True)`) and the orchestrator
is not unit-tested.

Asynchronous I/O is explicitly out of scope — the mitigation is a shorter
timeout plus more threads, not a rewrite.

### Scheduled Tasks
- **13:15 daily**: Full optimization (after Nord Pool prices publish)
- **Startup**: Initial optimization
- **Every `pv_sample_seconds` (60s)**: PV power sampling + slot close + bias refresh
- **Every 15 min**: Adaptive re-evaluation + schedule change logging
- **Every 5 min**: Safety checks
- **Hourly**: Mode execution + battery cost update

## Deployment to the HA machine

The running app lives on the Home Assistant share, not in this repo:

```
//192.168.33.167/addon_configs/a0d7b954_appdaemon/apps/
├── apps.yaml              # LIVE config, contains the HA token — never overwrite from here
├── battery_optimizer.py
└── battery_optimizer_lib/
```

**STOP the AppDaemon add-on before copying more than one file.** AppDaemon
hot-reloads on every `.py` modification, so a multi-file copy is imported
*while it is still in progress*: it will load a new module against its old
peers. That is not hypothetical — on 2026-07-28 it produced

```
ModuleNotFoundError: No module named 'battery_optimizer_lib.soc_projection'
TypeError: record_discharging() got an unexpected keyword argument 'battery_temp_start'
```

from a tree whose files were all individually correct. Copying a *single*
file while running is safe (one reload, no window).

Procedure:

1. Back up `battery_optimizer.py` + `battery_optimizer_lib/` on the share.
2. Stop the add-on.
3. Copy both, then delete `__pycache__` in each directory.
4. Verify every deployed file matches the repo, then start the add-on.

Before deploying, smoke-test the new code against the LIVE `apps.yaml`
(`BatteryOptimizerConfig.from_args`) and import every module — the unit suite
does not cover the orchestrator, so a config or wiring break only shows up here.

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
