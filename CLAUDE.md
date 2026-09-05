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
│   ├── __init__.py                # Re-exports the public classes
│   │                              #   (except direct_control and price_horizon,
│   │                              #    imported by module path)
│   ├── config.py                  # BatteryOptimizerConfig dataclass
│   ├── models.py                  # Data classes and enums
│   ├── dp_optimizer.py            # Dynamic programming optimizer
│   ├── learning_engine.py         # Self-learning charge rate tracker
│   ├── load_profile.py            # Statistical load forecasting
│   ├── pv_profile.py              # Statistical PV production profile
│   ├── pv_forecast_service.py     # PV forecast fetching (Solcast / Forecast.Solar)
│   ├── pv_bias_tracker.py         # Sliding PV forecast bias + slot-energy sampling
│   ├── price_service.py           # Nord Pool price fetching
│   ├── price_horizon.py           # Price coverage health + bounded recovery backoff
│   ├── direct_control.py          # Direct inverter control via set_wit_mode
│   ├── cost_tracker.py            # Battery cost tracking
│   ├── schedule_formatter.py      # Schedule logging/formatting
│   ├── thermal_model.py           # Shared battery temperature model (k1/k2)
│   ├── ambient_service.py         # Time-varying ambient temperature T_ambient(t)
│   ├── slot_energy.py             # Shared pure slot energy-flow transition
│   ├── plan_validation.py         # Continuous replay of the final plan
│   ├── soc_projection.py          # Shared slot SOC transition model
│   ├── soc_deviation.py           # SOC deviation detection
│   ├── load_prediction_tracker.py # Predicted vs actual load accuracy
│   ├── slot_outcome_tracker.py    # Per-slot outcome/compliance tracking
│   ├── timezone_utils.py          # Timezone-aware datetime helpers
│   └── ha_helpers.py              # HA state reading helpers
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
| `price_horizon.py` | PriceHorizonMonitor, PriceHorizonConfig, HorizonHealth | Usable-coverage verdict, retained intervals, recovery backoff state |
| `direct_control.py` | DirectControl | Direct inverter control via `growatt_modbus/set_wit_mode` |
| `cost_tracker.py` | BatteryCostTracker, BatteryCostConfig | Battery cost tracking with weighted averages |
| `schedule_formatter.py` | ScheduleFormatter, ScheduleFormatterConfig | Schedule logging and HA sensor formatting |
| `thermal_model.py` | TemperatureProjector, step_temperature, battery_power_for_entry | The ONE battery temperature model. Warming depends on `\|P_bat\|`, never on the scheduled mode |
| `ambient_service.py` | AmbientTemperatureService, AmbientServiceConfig | `T_ambient(t)` across the horizon: weather forecast → outdoor sensor → diurnal profile |
| `slot_energy.py` | simulate_slot, SlotEnergyParams, SlotEnergyResult | The ONE pure slot transition: every energy flow with its measurement boundary in the name |
| `plan_validation.py` | replay_plan, PlanReplay | Continuous replay of the FINAL plan; prefix energy conservation before clamping |
| `soc_projection.py` | project_slot_soc, SocProjectionParams | Single slot-SOC transition model shared by the expected-SOC trajectory, the projected-cost column and the deviation detector; delegates the physics to `slot_energy.simulate_slot`. The DP keeps its own inlined transition, fused with the value recursion — `test_dp_energy_conservation.py::TestPrefixConservationAcrossConditions::test_no_prefix_creates_energy` replays 210 selected plans and requires the DP's own trajectory to match |
| `soc_deviation.py` | SocDeviationDetector, SocDeviationConfig | Detects unexpected SOC changes for revalidation |
| `load_prediction_tracker.py` | LoadPredictionTracker | Predicted vs actual load accuracy tracking |
| `slot_outcome_tracker.py` | SlotOutcomeTracker | Per-slot outcome and mode compliance tracking |
| `timezone_utils.py` | normalize_tz_pair, align_to_slot, lookup_by_time, dt_ge | Timezone-aware datetime comparison and alignment |
| `ha_helpers.py` | SensorReader | HA state reading with validation |

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
4. Backtrack to extract the best-valued action sequence the merge kept

**Value calculations:**
- Marginal import price: `(spot_price + grid_fee) * import_price_multiplier`
- Net load: `max(0, predicted_load - predicted_pv)`; PV serves load before battery or grid energy
- Grid CHARGE cost uses AC imported energy. Stored battery energy includes the configured storage efficiency, while grid AC-to-DC conversion also applies `inverter_efficiency`; simultaneous PV surplus reduces the grid contribution.
- DISCHARGE serves `min(net_load, discharge_rate)` on the AC side. The SOC transition consumes additional DC energy for inverter loss, and battery wear is charged per discharged DC kWh.
- PV surplus can charge the battery in HOLD/CHARGE and remaining surplus earns `max(0, spot * export_rate_multiplier - grid_export_fee)`.

`efficiency` is the charge-retention factor, not a complete round-trip figure. `inverter_efficiency` applies on grid AC-to-DC charging and battery DC-to-AC discharge, so the modeled grid-charge round trip is approximately `efficiency * inverter_efficiency^2`.

**One charge-rate unit.** A "charge rate" is always `charge_input_dc_kw` — DC power at the battery terminal, BEFORE retention — everywhere a planner touches it: `charge_rate_kw` in apps.yaml, `SocProjectionParams.charge_rate`, the DP's per-slot rate, `|P_bat|` for the thermal model, and everything `BatteryLearningEngine.get_charge_rate_for_soc` returns. Stored energy is `rate * efficiency * duration`; grid AC is `grid_dc / inverter_efficiency`. Learning *observations* are the other quantity — `stored_charge_kw`, a SOC delta or the inverter's energy counter over an interval — and are recorded and persisted in those units unchanged. The single conversion happens at the API boundary in `get_charge_rate_for_soc`. Applying storage retention to a rate that already described stored-energy growth made a learned 40 %→50 % observation replay as 48.5 %; `tests/test_charge_rate_units.py` is that replay. Do not "simplify" by removing the `* efficiency` at a consumer — that fixes learned rates and breaks the nominal fallback, grid costs and PV limits, which is why the contract is named rather than inferred.

`learned_efficiency` is NOT a measurement. It can only be learned from an independent AC meter reading for the charge interval, and there is none; the synthetic `stored / configured_efficiency` input that used to feed it is a tautology and is rejected.

**One slot-energy model.** `slot_energy.simulate_slot` is the pure transition that names every flow (stored in/out, PV vs grid share of a charge, AC served, import, export, `unmet_battery_ac_kwh`). `soc_projection.project_slot_soc` delegates to it. Its `dc_energy_in_kwh`/`dc_energy_out_kwh` are what the pack ACTUALLY moved; the uncapped request lives in `requested_dc_energy_*`. Never report a request as delivered energy.

**Conservative quantization.** A DP state carries a floored bucket *label* and the *exact* energy of the best path reaching it; every transition is computed from the exact energy. Nearest rounding on the grid was not zero-mean for a constant load on a constant slot length — it credited 2.8 kWh of service from a 2.0 kWh battery and published 15 % SOC for a pack its own model had emptied. Pure floor-to-grid is safe but throws away 30 % of the usable energy at 1 % steps, so it is not the answer either. Physics is now exact; what `soc_step_percent` controls is only how aggressively distinct paths are MERGED. **That merge is an approximation, not an exact state reduction** — keeping the highest-valued path per bucket is not a dominance rule, because a lower-valued path holding more energy can be worth more later. Tie-breaking toward more energy IS a dominance rule; the value-only merge is not. Do not write "exact for its discretized model" anywhere: `tests/test_merge_approximation.py` pins the counterexample (10 kWh, 10.9 % initial, two slots at 0.10 then 1.00 EUR/kWh — solver 0.010 EUR, enumeration 0.005 EUR) and fails if the phrase reappears in the code, the docs, CLAUDE.md or the README. Bounds: energy loss < one step per merge, value loss <= step x marginal value per merge, horizon bound `n_slots * step * marginal_value` and nothing tighter is proven. What IS exact: the physics of every transition and replay parity. Energy-limited discharges deliver what the pack has and the grid pays for the rest; no threshold decides whether a slot "counts". `plan_validation.replay_plan`, called last in `find_optimal_schedule`, re-walks the FINAL action sequence and checks, before any clamping, that every slot the continuous model cannot serve in full is one the plan DECLARED energy-limited (`ScheduleEntry.energy_limited`). Accumulating the replay's own clamped flows and comparing them with the bound those flows are built to satisfy is an identity and catches nothing.

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

**One within-slot charge model.** A CHARGE slot runs at a *constant*
`charge_input_dc_kw` looked up at the temperature the slot **starts** at — in
the DP's candidate transition, in `simulate_slot`, in
`plan_validation.replay_plan`, in `project_slot_soc` (expected SOC and the
deviation detector) and in `cost_tracker.project_costs`. The rate never changes
inside a slot; temperature changes only *between* slots, through
`TemperatureProjector`. `learning_engine.predict_charge_input_dc_energy` splits
a slot into a cold and a warm phase using a second thermal model; it is
**diagnostic only** and must not be called from a planning or projection path.
It was, from `project_slot_soc`, and on one 15-minute slot crossing 1 kW → 4 kW
the DP said 12.5 % while the published trajectory said 16.25 %. The bound on the
constant-rate approximation, and why it errs conservative, is in
`docs/scheduling-algorithm.md` § Within-slot charge model.

**One thermal model.** Battery temperature is projected only by
`thermal_model.TemperatureProjector`, shared by the DP's rate refinement
(`_idle_temp_profile`, `_replay_plan_temps`), the expected-SOC trajectory
(`soc_projection`) and the schedule formatter. Two invariants must not be
broken:

1. Warming is a function of `|P_bat|`, not of the mode — discharging heats the
   pack. Never reintroduce a `mode == CHARGE` branch in a temperature path. And
   `|P_bat|` is the power that ACTUALLY flowed: `simulate_slot`'s
   `battery_power_kw`, not `thermal_model.battery_power_for_entry`, which models
   the REQUESTED flow and happily warms a full pack ordered to charge.
2. Ambient is `T_ambient(t)` from `ambient_service`, never one scalar for the
   whole horizon. In the no-external-source fallback, the learning engine's
   rolling battery minimum anchors the diurnal profile's daily **maximum**, not
   its minimum: the pack is self-heated, so `T_bat >= T_ambient` always and
   `min(T_bat)` is a *ceiling* on ambient. Anchoring it as the trough and adding
   the amplitude put the profile peak at `min + 2A` — an "ambient" hotter than
   the battery, which made `TemperatureProjector` warm an idle pack (33.0 →
   34.6 C over 3 h at 0 kW) and made `record_cooling` reject every summer
   cooling sample as `temp_end < ambient`.

3. Charge rates match the SOC and temperature the plan actually reaches. SOC
   dependence is evaluated **per candidate transition**, never from a
   time-indexed array built by pretending charging runs continuously from now —
   that array both warmed a cold pack with imaginary charging and pushed
   low-SOC paths into an imaginary taper. Temperature is handled by a **bounded
   solve/replay/refine**: pass 0 uses the *idle* profile (no heat from an
   action the plan has not committed to), each pass replays the selected plan
   through `TemperatureProjector` with warming from **actual** battery flow.
   Forecasts are fixed for the whole solve so a moving input cannot look like
   non-convergence.

   **The criterion is FEASIBILITY at the REACHED temperature, not a fixed
   point.** After each pass the selected plan is walked forward and every
   charging transition — CHARGE and the PV absorption a HOLD or self-consumption
   DISCHARGE performs — is checked against `rate(soc_start, replayed_temp)`. A
   fixed point alone let a 0.75 kWh shortfall through with `converged=True`.

   **Never sample a predictor to decide whether to check.** The loop used to be
   gated by a probe over three SOCs and a temperature ladder; a learned bucket
   that varied only between the probes skipped refinement entirely. Same
   reasoning as the removed SOC-independence hoist: `charge_rate_predictor` is
   an arbitrary callable. With a temperature reading, refinement always runs.

   On oscillation or budget exhaustion it solves once more on the **minimum
   rate over every profile seen this call**, and replays again. That is a bound
   over those profiles only; if it is still short the branch **degrades** —
   credited charge energy is cut to what the replayed temperature allows, the
   trajectory is rebuilt from that walk, and it is logged at WARNING with the
   shortfall. Economic optimality is lost there; say so rather than implying
   the fallback is always safe. `DPOptimizerResult.rate_refinement_branch`
   names the path taken.

4. **One trajectory, and it is the physical outcome.** Whatever branch chose
   the actions, the published SOC/temperature trajectory is the forward walk of
   those actions at the temperatures they reach. That is what
   `plan_validation.replay_plan` and `project_schedule_trajectory` compute, so
   **no consumer pins a charge-rate lookup to a planning temperature**.
   `planning_temp_by_slot` is a diagnostic; pinning it made validation check
   the planner's arithmetic against the planner's own assumption, which is the
   other half of how that 0.75 kWh went unreported. On a post-hedge shortfall
   the orchestrator reverts the cloud-safe conversions on the affected slots,
   re-validates, and only then degrades (ERROR). Never publish a plan that
   credits charge energy unavailable at the replayed temperature.

The temperature trajectory in `DPOptimizerResult` is reporting-only in the sense
that it is built by the same replay the refinement uses. Details, the `k1`/`k2`
units, the refinement's limits and its measured runtime are in
`docs/scheduling-algorithm.md` § Thermal model and § Charge rates that match the
SOC and temperature the plan reaches.

`min_charge_slots_required` is reporting-only: it estimates the aggregate energy deficit but does not constrain the DP. Feasibility comes from SOC-state transitions and power limits.

At the price-horizon boundary, `terminal_energy_value_eur_kwh` values usable stored DC energy. `auto` derives a conservative value from the median forecast import price, discharge conversion, and wear. It is a salvage value, not a hard terminal-SOC target.

`0` is **no-salvage mode**: it declares stored energy worthless at the horizon, so every plan ends by spending it (`EXPORT (until depleted) -> min_soc`). This is harmless while the daily re-optimization extends the horizon before those slots execute. Which mode is active is STATED, not warned about — once at config load (`config.TERMINAL_VALUE_ZERO_NOTICE`, also emitted from `log_summary`) and rate-limited from the DP (`DPOptimizer(warn_degenerate_terminal=...)`, gated by `_should_warn_degenerate_terminal()` to once per 6 h), all at INFO. Both settings have a real failure mode, so the code must not prescribe one: `0` risks spending the battery at the horizon edge, `"auto"` risks stranding charge there and skipping evening slots priced below the median. On the reference installation `"auto"` was tried and reverted: it stranded ~77% SOC at the horizon edge and skipped evening slots priced below the median, which cost more than the end-of-horizon spend it prevented. Do not re-add the old INFO line "net-load slots worth less than this are HELD" for the zero case — nothing is worth less than zero, so it described a rule that could never fire.

**The schedule log's value column is `ScheduleEntry.marginal_value_eur_kwh`, not the cost basis.** The DP fills it (with `value_basis` ∈ `avoided-import` / `export` / `landed-charge` / `kept`, plus `kept (cloud-safe)` for a hedged slot) from the same `_buy_price`/`_sell_price` arithmetic it scores slots with, normalized to one battery DC kWh. It is REPORTING ONLY — the DP objective never reads it, and `tests/test_schedule_value_column.py::test_marginal_value_does_not_change_the_schedule` guards that. Any new tariff formula must go through `_buy_price`/`_sell_price` so the report cannot drift from the objective.

**The orchestrator may only rewrite a DP action where the DP cannot tell the difference.** The one post-optimization rewrite is the cloud-safe hedge (`battery_optimizer._cloud_safe_hedge`): HOLD -> `discharge_to_load`, so a cloud rather than the grid covers the load. The DP has already chosen the rest of the horizon assuming the HOLD kept its energy, so the hedge is restricted to slots where forecast `pv >= load` (identical `soc_projection` transition), where nothing the plan priced as export would be curtailed by the 0 % export limit `discharge_to_load` writes, and where the avoided import beats both wear and what the plan says the kWh is worth kept — `max(terminal rate, best marginal_value_eur_kwh among LATER DISCHARGE slots)`. **The terminal rate alone is not an opportunity cost**: at the reference installation's `terminal_energy_value_eur_kwh: 0` it is zero while the plan is holding that kWh for the evening peak. The unrestricted version — any PV, import price above wear — emptied the pack in a 0.10 EUR/kWh slot and left the 1.00 EUR/kWh slot to the grid with exact forecasts. The later-slot bound is deliberately conservative (it ignores a recharge before that slot, so it under-hedges); forecast equivalence is not equivalence under a real cloud, and the reactive PV-shortfall replan is what bounds the rest. Full policy and the regression matrix: `docs/scheduling-algorithm.md` § The cloud-safe hedge, `tests/test_cloud_safe_conversion.py`.

### Inverter Control (DirectControl)
The schedule is executed by sending mode commands to the Growatt WIT inverter
via the `growatt_modbus/set_wit_mode` HA service (no raw register writes):
- Modes: `grid_charge`, `discharge_to_load`, `discharge_to_grid`, `max_export`, `hold`, `passthrough`
- Each command carries power_percent, duration, export_rate, ac_charge_mode, and SOC cutoffs
- AC charge mode auto-selects `pv_priority` vs `ac_priority` based on current PV power
- Duplicate commands within half a slot are skipped; `release_control()` reverts to `passthrough`
- Reliability: each call passes `hass_timeout=config.set_wit_mode_timeout_seconds` (default **15**) and inspects the service response. A raised/`success=False` result is a confirmed failure (ERROR, returns False, last-sent NOT recorded so it retries next slot). A `None` result is an unconfirmed client-side timeout (WARNING, last-sent recorded to avoid schedule spam). The timeout is short *because* the None path is safe: verify-after-set catches a genuinely lost command, whereas a long timeout blocks every other callback of this app.
- Health accounting reads `apply_mode_with_outcome`'s `ApplyOutcome`, never the boolean: `apply_mode` returns True for three outcomes the inverter never acknowledged (`DRY_RUN`, `SKIPPED_DUPLICATE`, `UNCONFIRMED_TIMEOUT`), so only `SENT` resets `_consecutive_apply_failures`, an unconfirmed timeout escalates to the same ERROR after 3 in a row (the hung-modbus case), and a duplicate skip or dry run is neutral.
- **Verification is opt-in and pluggable — a `Verifier` strategy, not one hard-wired sensor.** `verify_enabled` (default true) is the master switch; `verify_source` picks the strategy: `registers` / `mode_sensor` / `none` / `auto` (the default: registers whenever `device_id` is set, otherwise none — it deliberately never falls back to the mode sensor).
  - `RegisterVerifier` (recommended) reads holding 30407-30410 and 30200-30201 back through `growatt_modbus/get_register_data` (the schema field is **`start_address`**, not `address`). Expectations are derived from the params that were actually sent, so the check is against the command, not against a label: 30410 accepts `{2, 1}` (the handler writes either for an enabled AC charge), 30408 (duration, which does not count down) is informational only, and **any unclean read — exception, `None`, `ad_status` TIMEOUT/TERMINATING at either envelope depth, `success: False`, a short `values` list — is UNVERIFIABLE, never a MISMATCH.**
  - `ModeSensorVerifier` compares `inverter_mode_sensor` and is only correct while the integration's never-cleared `_failed_optional_holding_addrs` blacklist has not frozen that entity. On 2026-09-01T03:46:34Z one transient read failure froze it at "Passthrough" indefinitely; the 2026-09-02 log then carried 73/73 false mismatches, each paying for a blocking resend on the single AppDaemon thread. Empty `inverter_mode_sensor` disables it — no entity is ever guessed.
  - A `passthrough` match is recorded **non-probative** (`VerificationOutcome.probative=False`): a sensor that ignores overrides entirely reads "Passthrough" too, so counting it as verified would manufacture evidence.
  - `hass_timeout` expiry does not raise and does not return None: AppDaemon 4.5.13 stamps `ad_status: TIMEOUT` on the response, which is classified `UNCONFIRMED_TIMEOUT`, not `FAILED`. `_ad_status_of()` is the one helper both the `set_wit_mode` call and the register read use, because the stamp lands at the top level on some AD versions and under `result` on others.
  - **Physical plausibility is NOT a verification strategy.** Do not add one: on 2026-09-02 a discharge command at -100 % measured -39.7 W because SOC was 12 % against a 10 % cutoff. Correct behaviour, and indistinguishable from a dropped override.
- The ladder itself is a **bounded two-step ladder**, max 2 checks and 2 sends per `apply_mode` — never a resend loop:
  1. after `verify_delay_seconds` (default 90) the configured source is consulted; a genuine mismatch → WARNING (naming the source and the raw value) + resend once (bypassing duplicate suppression) + schedule check 2;
  2. after `verify_recheck_seconds` (default 60): match → INFO "recovered after resend"; mismatch → **ERROR** and stop (the next slot retries).

  The re-check exists to separate a lagging source from an inverter that genuinely drops the override; without it the 30 logged mismatches carried no evidence either way. `DirectControl.get_diagnostics()` exposes the counters (`mismatch_count`, `resend_count`, `resend_recovered_count`, `resend_failed_count`, `persistent_mismatch_count`, `unverifiable_count`) on `sensor.battery_inverter_control_health` and as an attribute of `sensor.battery_optimizer`. A new `apply_mode` supersedes any pending verification timer.
- `inverter_mode_sensor` has a **second, independent consumer**: `_get_inverter_mode` feeds `SlotOutcomeTracker.record_slot_end(actual_mode=...)`, the only per-slot record of mode compliance. Leaving it empty loses that history even when `verify_source: registers` is verifying perfectly, so keep it set (`sensor.growatt_inverter_mode`) for monitoring and let the registers verify.
- Lock order is **app lock → `DirectControl._io_lock` → `DirectControl._state_lock`**,
  never the reverse: nothing may acquire the app lock while holding a
  DirectControl lock. That is why the app drops its lock around
  `apply_mode_with_outcome`/`release_control`, and why `_verify_mode` reports
  its duration to `record_external_callback_duration` (which takes the app
  lock) only after both DirectControl locks are released.
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

**This app needs more than one AppDaemon thread, and `total_threads` alone
is not enough.** `set_wit_mode` is a synchronous, blocking service call made
from a callback, so on the default single thread one slow inverter write stalls
schedule execution, the SOC listener and PV sampling alike (production: 70 ×
"Excessive time spent in callback (limit=10.0s)" at 10–34 s, all on
`thread-0`). Two settings are required (AppDaemon 4.5.13):

```yaml
# appdaemon.yaml
appdaemon:
  total_threads: 4
  thread_duration_warning_threshold: 25   # optional; a set_wit_mode write is legitimately ~15 s
```
```yaml
# apps.yaml (LIVE file — hand-edit; it holds the HA token)
battery_optimizer:
  pin_app: false
```

`total_threads` clears the GLOBAL `pin_apps` flag only
(`models/config/appdaemon.py` `model_post_init`), while `app_should_be_pinned`
reads `cfg.pin_app or self.pin_apps` and `models/config/app.py` defaults
`pin_app: True`. A pinned app with `pin_thread = None` hits `select_q`'s
`"Invalid thread ID for pinned thread in app: ... - assigning to thread 0"`
WARNING **on every dispatch** and still runs everything on thread-0 — so
`total_threads` without `pin_app: false` is strictly worse than the default.
With both set, dispatch is round-robin and this app's callbacks genuinely run
**concurrently**. Do not set `pin_threads` (forced to 0). Rollback lever
without touching `appdaemon.yaml`: `pin_app: true` + `pin_thread: 2` (must be
`< total_threads`). Startup check: expect `Starting apps with 4 worker threads,
with None reserved for pinned apps`, and NO "Invalid thread ID for pinned
thread" line for `battery_optimizer`.

**Concurrency is handled by one app-wide lock, not by an async rewrite.**
`initialize`'s first statement is `self._lock = CallbackLock(log_func=self.log)`
(`battery_optimizer_lib/callback_lock.py`), and the rest of `initialize` runs
under it. Every registered callback carries `@_timed_callback`, which is the
complete chokepoint: its body runs under that lock, so schedule rebuilds, slot
execution, SOC/energy listeners, PV sampling and the trackers keep the
single-threaded semantics the app was written against. The one deliberate
escape hatch is the blocking inverter write: `_apply_mode_tracked` wraps
*only* `apply_mode_with_outcome(entry)` in `with self._lock.unlocked():` (same
for `release_control()` in `_on_enabled_change`). A nested call
(`full_optimize` / `_recalculate_remaining_schedule` → `execute_scheduled_mode`,
depth 2) intentionally does NOT release — the outer frame is mid-rebuild of
`self.schedule` — and must not be deferred via `run_in` (that passes a kwargs
dict, which flips the `kwargs is not None` branch of the execute dedupe).
`record_external_callback_duration` takes the lock itself because DirectControl
calls it from another worker thread. The four JSON save sites rely on the app
lock for their `open(..., "w")` truncation window.

The app instruments its own callbacks via `_timed_callback` and
`_record_callback_duration()` — `time.monotonic()` is sampled outside the
acquire so lock wait counts as thread occupancy — warning above
`callback_warn_seconds` and repeating the `total_threads` + `pin_app` advice
once after three overruns. The decorator must keep `functools.wraps` +
`*args/**kwargs`: AppDaemon calls these positionally
(`execute_scheduled_mode(kwargs, force=True)`) and the orchestrator is not
unit-tested (`tests/test_callback_instrumentation.py` guards the wiring with
source-scanning tests instead).

Asynchronous I/O is explicitly out of scope — the mitigation is a shorter
timeout, more threads plus the app lock, not a rewrite.

The 2026-09-02 production log predates this: it was produced by commit 86a2ffb
imported from a backup directory the deploy script had placed *inside* `apps/`
(AppDaemon rglobs `app_dir` and `sys.path.insert(0)`s every subdirectory
containing `.py` files), not by the files in `apps/` — which is why its log
wording does not match master.

### Scheduled Tasks
- **13:15 daily**: Full optimization (after Nord Pool prices publish)
- **Startup**: Initial optimization
- **Every `pv_sample_seconds` (60s)**: PV power sampling + slot close + bias refresh
- **Every 15 min**: Adaptive re-evaluation + schedule change logging + price-horizon health check
- **Every 5 min**: Safety checks
- **Hourly**: Mode execution + battery cost update
- **On demand, bounded backoff (30s / 2min / 5min / 15min)**: price recovery —
  `_price_recovery_retry`, armed by whichever path noticed the unusable horizon

**Price coverage has one owner.** `battery_optimizer_lib/price_horizon.py`
answers "is the horizon usable" (current interval present, contiguous, reaching
the end of the current publication window — measured between UTC instants, never
as a count of slots, because a Riga DST day is 92 or 100 quarter-hours). The
orchestrator owns only the timer: **at most one pending retry per app
instance**, carrying a generation token so a timer queued before a disable,
a `terminate()` or a successful recovery is inert. Every path that can notice the
same gap in the same minute — `full_optimize`, `_recalculate_remaining_schedule`,
the `no_price`/`no_schedule` HOLD in `execute_scheduled_mode`,
`adaptive_optimize` — shares that one retry and none of them advances the
backoff while it is armed.

The periodic adaptive pass evaluates the **last known** snapshot and never
fetches; the retry is the only new path that performs a blocking price fetch,
and it does so exactly where `full_optimize` already did. Recovery rebuilds
through `_recalculate_remaining_schedule` -> `execute_scheduled_mode`, so
enabled/override gating and command tracking are unchanged: during a manual
override the plan is refreshed and nothing is sent.

**No price is ever synthesized for the current interval.** The old
`_ensure_current_slot_price` substituted yesterday's same-clock price, else the
last past price, else the next price, on the premise that "Nord Pool may exclude
the current hour as past". Both fetch paths request whole days
(`get_price_indices_for_date` per date; `raw_today` / `raw_tomorrow`), so that
premise is false and an absent current interval means the data is missing — and
the substitution planned, logged and EXECUTED the live slot on a number nobody
published (yesterday at 0.01 against a real 1.00 next slot produced a grid-charge
command). Planning now starts at the next validated interval; the current slot
either keeps an existing entry that carries `ScheduleEntry.price_source ==
"market"` provenance or applies `HOLD/no_price`. `execute_scheduled_mode`
refuses to send any non-HOLD current-slot entry without that provenance. Asking a
source for the missing interval is a **fetch** and belongs in the price service
and the retry, never a substitution at planning time. See
`docs/scheduling-algorithm.md`, "The current interval when nobody published a
price for it".

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

**Backups must never live under `apps/`.** AppDaemon discovers apps with
`app_dir.rglob("*.py")` and does `sys.path.insert(0, <dir>)` for every
directory below `apps/` that holds `.py` files and has no `__init__.py`, so
such a directory sits at the *front* of `sys.path` and wins every
`import battery_optimizer` / `import battery_optimizer_lib`. On 2026-09-02
`deploy.ps1` wrote `apps/backup-20260902-015911/` and the add-on then ran the
*previous* commit while SHA256 verification of `apps/` passed — the files in
`apps/` were correct, the wrong ones were imported from the sibling directory.
The only symptoms were an old log wording and a health sensor missing the
attributes the new commit added. Backups therefore go to
`<share-root>/backups/battery_optimizer/`, and nothing but
`battery_optimizer.py`, `battery_optimizer_lib/` and `hello.py` may hold `.py`
under `apps/`.

**SHA256 proves the bytes on the share; it does not prove AppDaemon imported
them.** Every deploy must end with a positive check that the *running* code is
the new one: `initialize` logs `Battery Optimizer version <APP_VERSION>:
orchestrator=<path> lib=<path>`, and `sensor.battery_optimizer` carries the
same as `app_version` / `code_paths`. Both paths must point into `apps/`
itself. Bump `APP_VERSION` in `battery_optimizer.py` with behaviour changes.

Procedure:

1. Back up `battery_optimizer.py` + `battery_optimizer_lib/` on the share.
2. Stop the add-on.
3. Copy both, then delete `__pycache__` in each directory.
4. Verify every deployed file matches the repo, then start the add-on.

Before deploying, smoke-test the new code against the LIVE `apps.yaml`
(`BatteryOptimizerConfig.from_args`) and import every module — the unit suite
does not cover the orchestrator, so a config or wiring break only shows up here.

`scripts/deploy.ps1` automates exactly that procedure (Windows PowerShell 5.1):
git-clean check plus the commit/branch it will deploy, `pytest`, `py_compile`,
`scripts/smoke_config.py` against a temp copy of the LIVE `apps.yaml` (deleted
immediately — the share's `apps.yaml` is only ever read), a pre-flight scan
that aborts on anything under `apps/` AppDaemon would import besides our own
files (`-MoveStrayBackups` relocates legacy `backup-*` directories instead of
aborting), a timestamped `<share-root>/backups/battery_optimizer/backup-<ts>/`
keeping the 5 newest, stop → copy (pruning `.py` files the repo no longer has)
→ `__pycache__` cleanup → SHA256 verification → mtime stamping (`Copy-Item`
preserves the source time, which would make the share's timestamps describe
the git checkout) → start → a best-effort post-deploy read of the add-on log
for "Initializing Battery Optimizer", import errors and the worker-thread
count. Start with `-DryRun`, which runs every check and prints the planned copy
list while writing nothing; `-Restore <backup-dir>` rolls a deploy back through
the same stop/copy/start dance. Add-on stop/start goes through HA's Supervisor
proxy when `-HaToken` is given, otherwise the script pauses for you to do it in
the UI. See `scripts/README.md`.

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
