# Battery Optimizer Scheduling Algorithm

This document describes the dynamic-programming (DP) scheduler in
`appdaemon/apps/battery_optimizer_lib/dp_optimizer.py` and its orchestration in
`appdaemon/apps/battery_optimizer.py`.

## Objective and inputs

For each price interval, the optimizer chooses one of three modes:

- `HOLD`: serve load from the grid; PV may charge the battery or be exported.
- `CHARGE`: serve load and charge the battery from the grid.
- `DISCHARGE`: serve house load from the battery and, when profitable and
  configured, export additional battery energy.

The DP maximizes total economic value over the available price horizon. It uses
Nord Pool spot prices, import/export fees, predicted house load and PV,
battery/inverter losses, battery wear cost, power limits, and the configured SOC
range. SOC is discretized by `soc_step_percent`.

`battery_avg_cost` is an operational accounting value exposed for monitoring.
It is not a charge gate or discharge threshold in the DP. Likewise,
`min_charge_slots_required` is an informational estimate of the aggregate
energy deficit; the SOC-state DP itself enforces feasibility and may select a
different number of charge slots.

## Time and price handling

Price timestamps are normalized to timezone-aware instants. Instant comparisons
and ordering use UTC; schedule keys retain a concrete UTC offset so the repeated
autumn-DST intervals remain distinct. Local time is used for load/PV prediction,
logs, and Home Assistant presentation.

The configured price resolution is normalized to `slot_minutes`: coarser input
is expanded and finer input is averaged. Each `PricePoint.price` and every fee
must use the same currency and per-kWh basis.

The current, partly elapsed interval is scaled by its remaining fraction. Future
intervals use their full duration.

## Energy and cost model

The principal quantities for one interval are:

```text
buy_price  = (spot_price + grid_fee) * import_price_multiplier
sell_price = max(0, spot_price * export_rate_multiplier - grid_export_fee)

stored_charge_kWh = charge_power_kW * slot_hours * slot_fraction * efficiency
AC_grid_charge_kWh = stored_charge_kWh / (efficiency * inverter_efficiency)
AC_from_battery_kWh = discharged_DC_kWh * inverter_efficiency
```

`efficiency` is the battery/storage charge-retention factor used when AC or PV
energy is stored. Despite its historical name, it is not a complete round-trip
efficiency. `inverter_efficiency` is applied to AC-to-DC grid charging and to
DC-to-AC discharge. For the grid-charge-to-AC-discharge path, the implied
round-trip factor is approximately:

```text
efficiency * inverter_efficiency * inverter_efficiency
```

Use values consistent with that model. For example, `0.95` and `0.97` imply an
AC-to-AC round trip of about 89.4%, not 95%.

Import cost includes both grid charging and any remaining house load. Discharge
value is avoided import or net export revenue, less wear cost per discharged DC
kWh. Fixed monthly connection/capacity charges are excluded because a schedule
cannot change them.

## SOC transitions and discretization

The state is `(time, discretized stored energy)`. Each reachable state retains
the best cumulative value and a predecessor for backtracking. Transitions obey
`min_soc`, `max_soc`, charge/discharge power limits, PV availability, and the
remaining fraction of the current interval.

Continuous energy is mapped conservatively to the discrete grid: a transition
must never credit energy that is not physically present. A smaller
`soc_step_percent` reduces this conservative quantization error at the cost of
more states and CPU time.

Temperature-aware charge rates are predicted before the DP from the learned
SOC/temperature model. These are planning estimates; actual SOC deviation can
trigger a re-optimization.

### One bound on the learned charge rate

`BatteryLearningEngine.get_charge_rate_for_soc` is the ONE gate every consumer of
a learned battery power passes through: the DP (via
`charge_rate_utils.compute_charge_rates_per_slot`), the expected-SOC trajectory
(via `soc_projection._effective_charge_rate`), the deviation detector (via
`_project_charge_completion` and `_calculate_extra_charge_slots`) and the
orchestrator's status sensors. The plausibility bound therefore lives there and
nowhere else — three ad-hoc clamps in three consumers is exactly the drift the
"one model" rule exists to prevent.

Two lines of defence, both derived from the same constants in
`learning_engine.py`:

1. **At ingest.** `record_charging` / `record_discharging` reject an observation
   that cannot resolve a rate (`observation_is_resolvable`, below) or whose
   implied rate exceeds `max_plausible_rate_kw = max(charge_rate,
   discharge_rate, effective_export_discharge_rate) * max_rate_factor` (default
   `DEFAULT_MAX_RATE_FACTOR = 2.0`). Rejections are logged
   (`Learning: rejected implausible …`) and counted in
   `get_learning_summary()["rejected_observations"]`.
2. **At read.** `_plausible_rates` filters every median window *before* the
   `[-10:]` slice, and `_bounded_median` clamps the result. `load_from_json`
   runs the same filter (`sanitize_stats`) so a file written before the ingest
   guards existed is neutralised in memory on load and written back clean on the
   next save.

**Why 2x nominal, over all three powers.** The reference installation is
configured at 4.5 kW and its observation history has a hard physical ceiling at
~6.8 kW (the inverter's warm-battery rate, 1.5x nominal). A 1.5x bound would
clip that genuine cluster; 2x keeps it and still rejects everything a 15-minute
slot cannot deliver. The maximum is taken over the charge rate, the load
discharge rate **and** `effective_export_discharge_rate`: an export slot runs at
the export rate, routinely the largest of the three, so leaving it out would
reject the very samples the DP plans an export around.

**Why the floor is quantization-aware, not wall-time.**
`observation_is_resolvable(energy, duration)` accepts a sample when EITHER the
measured energy spans at least two counter ticks (`2 *
counter_resolution_kwh`, default `2 * 0.1 = 0.2` kWh) OR the interval is at
least `min_observation_minutes` (`MIN_OBSERVATION_MINUTES = 0.25`, i.e. 15 s, an
absolute floor). Anything else is one tick of granularity divided by a number,
not a rate.

The earlier flat 1-minute floor was wrong in the same direction the 2x bound was
tuned against. `cost_tracker` re-stamps `_last_sig_soc_time` after **every**
accepted event, so a genuine interval lasts only as long as the counter needs to
advance one 0.1 kWh tick — `0.1 / P` hours, under a minute for any `P` above
6 kW. The 1-minute floor therefore rejected exactly the 6.77-6.82 kW warm-pack
cluster it was meant to protect. The production defect it was introduced for
(0.1 kWh over 44 ms, ~9000 kW) fails the quantization gate anyway, and
`is_plausible_rate` remains the guard that catches a multi-tick delta over a
millisecond interval.

**What this cost in production (2026-09-02).** `cost_tracker` derived an
observation's duration from `_last_sig_soc_time`, which the SOC listener
re-stamps milliseconds before the energy-sensor callback records the charge. A
genuine 0.1 kWh delta over a 10-40 ms "duration" produced 34535 kW and 44653 kW
observations. The live file's 0-25 %/>20 C bucket held
`[2.806, 34535.687, 44653.932, 14308.71, 5.959]`, so the median served to every
consumer was **14308.71 kW**. Three consumers then produced three different
numbers for the same 05:00 CHARGE slot, all from that one file:

| consumer | path | rate | slot effect |
|---|---|---|---|
| DP | `charge_rate_utils` at projected SOC/temp | 10.95 kW | +18.2 %/slot planned |
| expected-SOC trajectory | `project_slot_soc` at 10 %, 16 C | 3.01 kW | +5.0 %/slot |
| deviation detector | `_project_charge_completion` at 10 %, 21.9 C | 14308.71 kW | "projected to reach 21894.1 %" |

The DP's 10.95 kW is the tell: `compute_charge_rates_per_slot` walks the SOC
forward assuming continuous charging, so the first slot's 14308.71 kW saturated
the projection at 100 % and *every remaining slot* was priced from the
`90-100`/`>20` bucket (median 10.95 kW) instead of the bucket it would really be
in. Reality delivered 6.86 kW (SOC 9 %→21 % in 15 min).

`scripts/clean_learning_data.py` inspects a learning file and writes a cleaned
copy using the same `sanitize_stats` rule; it never modifies its input.

### Shared slot transition (`soc_projection.py`)

Five components must agree on what one slot does to the SOC: the DP (which
chooses the plan), the expected-SOC trajectory
(`BatteryOptimizer.calculate_expected_soc_schedule`), the deviation detector
(`SocDeviationDetector`), the schedule log's fallback trajectory
(`ScheduleFormatter._format_expected_trajectory`, used whenever
`dp_soc_trajectory` does not cover a slot) and the projected-cost column
(`BatteryCostTracker.project_costs`). The latter four delegate to
`battery_optimizer_lib/soc_projection.py::project_slot_soc`; the DP keeps its own
inlined transition because it is fused with the value recursion and the discrete
energy grid. `tests/test_soc_projection.py` pins them together.

`project_costs` was the fourth private copy: it capped charging with its own
headroom arithmetic and clamped a DISCHARGE at `min_soc` *before* adding PV
surplus, where the shared model adds PV, clamps at `max_soc`, then subtracts. It
now derives both the SOC and the DC energies it prices from
`project_slot_soc`'s `SocTransition`; only the landed-cost and weighted-average
arithmetic stays local, so the projected-cost column cannot drift from the
SOC/deviation columns of the same log.

The formatter was the last holdout and had to carry `inverter_efficiency` and a
`predict_pv_kw` callback to join: its HOLD branch printed `end_soc = start_soc`
(no PV surplus charging) and its DISCHARGE branch removed
`min(load_kw, discharge_rate) * slot_hours` from raw load — no PV subtraction, no
DC conversion. On a sunny slot (PV 4.0 kW, load 0.8 kW, SOC 50 %) it printed
HOLD 50.0 %→50.0 % and DISCHARGE 50.0 %→48.6 % against the shared model's
55.3 %, i.e. the diagnostic surface contradicted the trajectory the deviation
detector compares against.

Invariants, all of which were violated at some point and produced recalculation
loops in production:

1. **Partial first slot.** The current slot is projected with the same
   `first_fraction = (slot_minutes - minutes_into_slot) / slot_minutes` formula
   everywhere (`_compute_slot_fractions`, `DPOptimizer.optimize`,
   `calculate_expected_soc_schedule`). Projecting a full slot when only minutes
   remain guarantees a false "SOC behind plan" at the next slot boundary.
2. **DISCHARGE with `pv >= load` is a charge, not a discharge.** The battery
   serves `max(0, load - pv)` on the AC side and stores `max(0, pv - load)`
   (capped by the charge rate). The same holds for HOLD, which is what the
   cloud-safe HOLD→DISCHARGE conversion relies on.
3. **Export slots drain at `effective_export_discharge_rate`**, not at the load
   rate.
4. **DC energy moves the SOC.** AC load served is divided by
   `inverter_efficiency`; stored energy is multiplied by `efficiency`.
5. **Anchoring.** When the trajectory is (re)built mid-slot, its first entry
   describes the *recalculation instant*, not the slot boundary. That instant is
   stored in `BatteryOptimizer._expected_soc_anchor` and passed to the deviation
   detector, which otherwise would count the already-elapsed part of the slot
   twice.
6. **DP agreement tolerance.** The shared projection may differ from the DP by at
   most one `soc_step_percent` grid step *per slot*, compared from the DP's own
   slot-start SOC (never cumulatively).

## Thermal model (`thermal_model.py`)

One physics model owns battery temperature projection. `thermal_model.step_temperature`
implements

```
T(t+dt) = Ta(t) + (T(t) - Ta(t)) * exp(-k1*dt) + k2 * |P_bat| * dt/60
```

- `k1` — Newtonian relaxation rate, **per minute**. The exponential form (not the
  Euler `T + k1*(Ta-T)*dt`) is deliberate: the learned `temp_cooling_rates` are
  already decay-per-minute values computed as `-ln(ratio)/duration`, so historical
  learning data stays valid.
- `k2` — self-heating, **Celsius per kWh moved through the battery**. It depends on
  `|P_bat|` only, so charging and discharging of equal magnitude heat identically.

### Invariants

1. **Warming is a function of power, not of mode.** A 5.9 kW discharge heats the
   pack. Never reintroduce a `mode == CHARGE` branch in a temperature path.
   `thermal_model.battery_power_for_entry` is the single place that derives
   `|P_bat|` from a scheduled slot, and its energy split mirrors
   `soc_projection.project_slot_soc` exactly — including the case where a
   `DISCHARGE` slot has `pv >= load`. There the shared SOC model *charges* the
   pack from `min(pv - load, charge_rate)`, so reporting 0 kW made a pack whose
   SOC was rising read as thermally idle. The orchestrator's cloud-safe
   HOLD -> `discharge_to_load` conversion turns midday HOLD slots into DISCHARGE,
   so that was the routine midday case, and it was a `mode`-keyed special case of
   exactly the kind this invariant forbids. `DISCHARGE` with `pv >= load` and
   `HOLD` with the same forecast must return the same `|P_bat|`.
2. **Ambient is `T_ambient(t)`, never one scalar for the horizon.**
   `ambient_service.AmbientTemperatureService` resolves it per slot with the chain
   *HA weather forecast -> outdoor temperature sensor -> diurnal profile around the
   learned battery minimum*. The learning engine's rolling minimum
   (`get_estimated_ambient_min_temp`) anchors the **daily maximum** of that
   profile, not a constant and not its trough: the pack is self-heated, so
   `T_bat(t) >= T_ambient(t)` always and `min(T_bat)` is an *upper bound* on
   ambient. Anchoring it as the minimum and adding the amplitude put the peak at
   `min + 2A` (default +8 C), i.e. an "ambient" above the battery's own
   temperature — the projector then warmed an idle pack (33.0 -> 34.6 C over 3 h
   at 0 kW) and `record_cooling` discarded every summer sample via
   `temp_end < ambient_temp`, so `k1` never got calibrated. The fallback profile
   therefore spans `[min - 2A, min]`.
3. **One projector, four consumers.** `DPOptimizer._build_temp_trajectory`,
   `soc_projection.project_slot_soc` (used by the expected-SOC trajectory),
   `ScheduleFormatter` and `charge_rate_utils.compute_charge_rates_per_slot` all go
   through the same `TemperatureProjector`. Two different models on two code paths
   is the bug this replaced.
4. **The trajectory is reporting-only in the DP output.**
   `_build_temp_trajectory` runs *after* `_build_schedule`. Temperature influences
   decisions solely through `compute_charge_rates_per_slot`, because that feeds
   `get_charge_rate_for_soc(soc, temp)`.
5. **Projections are bounded.** `TemperatureProjector.project` clamps to
   `MAX_BATTERY_TEMP_C` and cannot undershoot `min(start, ambient) - 2 C`.
   The unbounded linear projection it replaced reached ~230 C after 132 slots.

### Calibration

`k1`/`k2` are fitted over `LearningStats.thermal_samples`
(`[T_start, T_end, dt, |P_bat|, T_ambient]`, last 300) **to the exponential model
above, not to its Euler linearisation**. The fit minimises the residual of
`step_temperature` itself,

```
r = (T_start - Ta) * exp(-k1*dt) + k2 * |P_bat| * dt/60 - (T_end - Ta)
```

by damped Gauss-Newton in pure Python, starting from the Euler normal-equation
solution. Fitting the Euler form `(T_end-T_start)/dt = -k1*(T_start-Ta) + k2'*|P|`
directly — as the calibration originally did — recovers a `k1` low by roughly
`k1*dt/2`: **2.9 % at dt=5 min, 16.0 % at 30 min and 28.7 % at 60 min** for
`k1 = 0.012/min`. Thermal samples span whole charge/discharge sessions, so 20-40 min
intervals are the norm and the bias was systematic against the very projector the
coefficients feed. `k2` is fitted directly in C per kWh and is unaffected by the
linearisation. At least 20 samples are required and the regressors must not be
collinear (all-equal power cannot separate relaxation from heating). Results are
clamped to `k1 ∈ [MIN_COOLING_RATE_PER_MIN, MAX_COOLING_RATE_PER_MIN] = [0.001, 0.1]`
per minute and `k2 ∈ [0, MAX_HEATING_C_PER_KWH] = [0, 3]` C/kWh.

`|P_bat|` is the `k2` regressor, so `record_thermal_observation` rejects any
sample whose power exceeds the same `max_plausible_rate_kw` bound the charge-rate
consumers use — a corrupted power does not merely add noise here, it drags the
whole pooled fit. `load_from_json` discards a persisted `thermal_coeffs` that
fails `thermal_coeffs_are_sane` (outside those ranges, or non-finite) and
`reset_thermal_calibration()` re-bootstraps from scratch on demand.

The `k2` ceiling was raised from 2 to 3 C/kWh because 2 was **binding on real
data**: the reference pack measured 21.9 C → 25.8 C while storing 1.716 kWh in
one 15-minute slot = 2.27 C/kWh, and the warming-rate bootstrap wanted 2.105.
A high `k2` on this installation is a genuine property of a small thermal mass,
not evidence of corruption — the corruption showed up in the *rate*, not in the
temperature.

**One ambient source for both recorders.** `record_charging` and
`record_discharging` feed a single pooled regression whose relaxation regressor is
`-(T_start - Ta)`, so both must take `ambient_temp` from the ambient service.
`record_charging` used to have no such parameter and fell back to the rolling
battery-temperature minimum: in summer that sits ~10 C above the real ambient, so
two thermally identical samples entered the fit as `x1 = -3` (charge) and
`x1 = -13` (discharge) and `k1` absorbed the charge/discharge mode instead of the
relaxation.

Until then `get_heating_coefficient()` **bootstraps** from the already-collected
charge warming rates: `median(C/min) / nominal_charge_rate * 60`. This matters
because the pre-existing learning data contains no usable power information —
`temp_warming_rates` is aggregated per starting-temperature bucket without `|P_bat|`,
and `record_discharging` historically took no temperatures at all. Genuine `k2`
calibration only becomes available after several days of operation.

### Configuration

`ambient_weather_entity` / `outdoor_temp_sensor` select the ambient source; prefer a
sensor in the room the battery lives in, since a weather entity reports *outdoor*
air. `ambient_diurnal_amplitude_c` and `ambient_diurnal_peak_hour` shape the fallback
profile. `thermal_default_cooling_rate_per_min` / `thermal_default_heating_c_per_kwh`
are the pre-calibration defaults.

Note that with the default `temp_ranges` of `[5, 10, 15, 20]` every summer
temperature falls in the single `>20` bucket, so a more accurate summer forecast may
not change any DP decision. Finer `temp_ranges` would expose the benefit, at the cost
of re-splitting existing observations across new buckets.

## End-of-horizon value

Without an end condition, a finite-horizon optimizer treats stored energy after
the last known price as worthless and tends to empty the battery. The terminal
energy value corrects that horizon effect:

- A numeric `terminal_energy_value_eur_kwh` values each usable DC kWh remaining
  at the horizon by that amount.
- `terminal_energy_value_eur_kwh: auto` derives a conservative value from the
  median horizon import price, battery-to-AC conversion, and wear cost.
- `0` disables terminal value: stored energy is worth nothing at the horizon edge.

The default application configuration uses `auto`. This is a value, not a hard
terminal-SOC constraint: sufficiently valuable load or export may still justify
ending near `min_soc`.

### `0` is no-salvage mode, and the app says which mode is active

Pinning the terminal value to `0` is not "no adjustment" — it is an explicit
claim that stored energy is worthless at the horizon, which makes spending it
there optimal by construction. The symptom in the schedule log is the last
slots always reading:

```text
07-30 00:30  DISCHARGE  ... (until depleted) [EXPORT]  -> 11.2%
```

This is usually harmless: the daily re-optimization extends the horizon before
those slots execute. And `"auto"` is not a free upgrade — it has its own failure
mode. On the reference installation `"auto"` was tried and reverted: it stranded ~77% SOC at the horizon edge and skipped evening slots priced below the median, which cost more than the end-of-horizon spend it prevented.

Two things must remain true here:

1. The active mode is surfaced at **INFO**, never as a warning and never with a
   recommendation: once at config load (`config.TERMINAL_VALUE_ZERO_NOTICE`,
   also emitted by `log_summary`) and rate-limited from the DP
   (`DPOptimizer(warn_degenerate_terminal=...)`, gated to once per 6 h by
   `BatteryOptimizer._should_warn_degenerate_terminal`).
2. The old INFO line "net-load slots worth less than this are HELD" must not be
   printed for the zero case. No slot is worth less than zero, so it described a
   rule that could never fire and read like a normal, working configuration.

The warning does not change the schedule. The deployed `apps.yaml` has to be
changed to `"auto"`.

## PV forecast and live control

Forecast PV participates directly in the DP. PV first serves predicted load;
surplus can charge the battery within its charge/headroom limits, and remaining
surplus can earn net export revenue.

### The cloud-safe hedge

After optimization, some HOLD slots are converted to DISCHARGE(to load), tagged
`[cloud-safe]` in the schedule log. On the Growatt WIT, `discharge_to_load`
charges from PV surplus exactly like `hold` while the sun covers the load, but
the battery — not the grid — picks up the load the moment clouds cut PV,
without waiting for the next re-optimization.

This is a **hedge, not an economic improvement on the DP.** The DP has already
chosen the whole horizon on the assumption that a HOLD slot preserves its
energy, so a slot may only be rewritten where the DP's own model cannot tell
the two actions apart. `BatteryOptimizer._cloud_safe_hedge` requires all four:

1. **Forecast PV covers forecast load** (`pv >= load`). Then the net load is
   zero and `soc_projection.project_slot_soc` gives DISCHARGE and HOLD the same
   transition: no DC out, and `min(pv - load, charge_rate)` stored. With
   `0 < pv < load` the actions differ — DISCHARGE drains a pack the DP reserved
   for a later slot — and no import price makes them equivalent.
2. **Nothing the plan was going to sell gets curtailed.** `discharge_to_load`
   pins the export limiter to 0 % (see `direct_control.expected_registers`),
   while `hold` leaves it open. So either the sell price is zero, or the HOLD
   plan stored the whole surplus in the pack. The check reads the shared
   model's continuous trajectory of the HOLD plan, not the DP's quantized one:
   a single SOC step of rounding is enough to hide a slot's worth of
   exportable PV. A full pack with a **sellable** surplus therefore never
   converts — which is the planning side of the execution-time
   `DISCHARGE -> HOLD at max SOC with PV > load` override. (With export
   remuneration at zero there is nothing to curtail, so this condition does not
   apply and a full pack may convert; the execution-time override still stands
   behind it.)
3. **The avoided import beats battery wear**, per discharged DC kWh.
4. **The avoided import beats what the plan says the kWh is worth kept** —
   `max(terminal rate, best marginal_value_eur_kwh among LATER DISCHARGE slots
   of this plan)`. The horizon-end terminal rate alone is not an opportunity
   cost: with the common `terminal_energy_value_eur_kwh: 0` it is zero while
   the plan is reserving that kWh for a 1.00 EUR/kWh evening slot, and the
   hedge would spend it to avoid 0.10 of import the moment a cloud arrived.
   The per-slot `marginal_value_eur_kwh` the DP already fills is exactly the
   EUR the plan expects from that kWh in that slot; a missing value counts as
   zero. This is deliberately **conservative**: it ignores whether the pack
   would have been recharged (from PV or a cheap slot) before the expensive
   slot, so the hedge is refused on some slots where spending the kWh would in
   fact have cost nothing. Under-hedging is the accepted direction of error —
   the insurance is optional, the energy the plan is counting on is not.

Converted entries keep the DP's marginal value, because their modeled flow is
unchanged; their `value_basis` becomes `kept (cloud-safe)` so a DISCHARGE row
never reports a bare HOLD label. The published SOC and temperature trajectories
are rebuilt from the final schedule through the shared model, and the mode
census is taken after the conversion.

**Forecast equivalence is not equivalence under every cloud event.** If PV does
collapse, the pack drains where the plan expected it to idle. Conditions 3 and
4 price that per kWh against the plan's own numbers, but they cannot model a
*sequence* of cloudy slots, a load above forecast, or a peak that arrives after
a charge the hedge made unaffordable. The expected SOC trajectory
assumes PV covers the slot, so a cloud-induced drain surfaces as an SOC
deviation, and the reactive PV-shortfall path forces a forecast refresh and a
replan — that, not the hedge itself, is what limits the exposure.

During execution, live PV above `pv_threshold_w` can pause a scheduled grid
charge so solar can charge instead. This real-time safety/operational override
does not rewrite future schedule entries.

When measured PV falls materially below the current-slot forecast, the app
forces a rate-limited forecast refresh, caps that slot at the observed output,
and replans. This prevents the normal forecast cache from repeatedly selecting
HOLD from a stale optimistic value.

### The ramp gate: `pv_reactive_min_forecast_w`

One threshold gates *both* the reactive shortfall check and the sliding bias
window (`PvBiasConfig.min_forecast_kw` is derived from it), so a slot below it
contributes neither a recalculation trigger nor a ratio observation.

It must sit above the sunrise/sunset ramp. On 2026-09-02 the 07:00 and 07:15
slots were forecast at 292 W and measured 0 W; with the old 200 W gate those two
ratio-0.0 observations were the entire evidence base, and the median dropped the
whole-horizon bias onto the 0.20 clamp at 07:30 — a 5x under-forecast of a sunny
day derived from two pre-production slots. A few minutes of ramp-timing error is
a ~100% *relative* error on a ramp slot while being economically meaningless:
below the site's baseline load the DP's net load `max(0, load - pv)` barely
moves either way.

The default is therefore 600 W — above the ramp, roughly one baseline house
load, and still only ~12% of a 5 kW array's peak, so genuine daytime cloud
events are unaffected. `pv_bias_min_slots` (default 2) is the second guard: the
factor stays at 1.0 until that many qualifying observations are in the window.

## Battery cost tracking

The weighted average stored-energy cost is persisted across restarts and exposed
as `battery_avg_cost`. Grid charging should be recorded on a landed stored-kWh
basis, including configured variable import charges and conversion losses. PV
charging should use its opportunity cost (foregone net export revenue), not the
current grid purchase price. Discharging reduces stored energy without changing
the per-kWh average.

`input_number.battery_cost_basis_version` distinguishes the landed-cost basis
(version 2) from legacy raw-spot values. Version 1 is migrated once using a
conservative grid-charge attribution and then persisted as version 2.

The tracker is an estimate because inverter aggregate charge counters may not
identify the source of every charged kWh. It is useful for reporting, but the DP
optimizes the forecast cash flows directly.

### Stored-energy cost formulas

All costs are per stored DC kWh (`BatteryCostTracker` in `cost_tracker.py`):

```text
grid_landed_cost    = (spot + grid_fee) * import_price_multiplier
                      / (efficiency * inverter_efficiency)

pv_opportunity_cost = max(0, spot * export_rate_multiplier - grid_export_fee)
                      / efficiency
```

The division by `efficiency` converts an acquisition price into a
per-stored-kWh figure: storing 1 kWh retains only `efficiency` of the input
energy, so each stored kWh consumed `1/efficiency` kWh of exportable PV (grid
charging additionally pays the AC-to-DC `inverter_efficiency` loss). The booked
PV cost per stored kWh is therefore *higher* than the net export price. Example
with default fees: spot 0.108 gives a net export price of 0.088 EUR/kWh but a
stored-energy cost of `(0.108 - 0.02) / 0.85 = 0.1036` EUR/kWh.

### Source attribution

Inverter charge counters do not label the source of each kWh, so measured
charging is attributed by `_observed_charge_cost`. The commanded mode is the
first input, but **it is not sufficient on its own**: a source is only accepted
when it could physically have supplied the energy. The rules are evaluated in
order.

| # | Condition | Source | Cost applied |
|---|---|---|---|
| 1 | commanded mode is CHARGE | `grid` | `grid_landed_cost` (conservative if PV also contributed) |
| 2 | a `grid_charge` command is still in force at the inverter **and** measured PV is below `cost_pv_attribution_min_w` (or unreadable) | `grid-command` | `grid_landed_cost` |
| 3 | HOLD / DISCHARGE and measured PV >= `cost_pv_attribution_min_w` | `pv` | `pv_opportunity_cost` (discharge-to-load still accepts surplus PV into the battery) |
| 4 | HOLD / DISCHARGE and measured PV below the floor | `no-pv-grid` | `grid_landed_cost` |
| 5 | HOLD / DISCHARGE and no PV reading available at all | `pv` | `pv_opportunity_cost` (legacy behaviour; no PV provider injected) |
| 6 | unknown (before first mode callback) | `unknown-grid` | `grid_landed_cost` (conservative) |
| — | slot price unavailable | — | current average preserved unchanged |

Rules 2 and 4 exist because the mode the *app* believes it is in is not the mode
the *inverter* is executing. A `set_wit_mode` override runs for `slot_minutes +
direct_control_buffer_minutes`, so a grid charge commanded at 05:00 is still
running when the app moves to HOLD at 05:15. On 2026-09-02 the +0.1 kWh measured
five seconds after that transition — an hour before sunrise — was booked
`[inverter, pv]` at 0.0253 EUR/kWh and pulled the basis from 0.1261 to 0.1199.
Rule 2 keeps grid attribution for the remainder of that command plus
`cost_grid_charge_grace_seconds` after another mode supersedes it (the energy
counters lag the command); rule 4 is the backstop for any other night-time
charge. Both guards fail *toward* the grid cost, which is the conservative
direction: it never books unpaid-for energy at a lower cost than it had.

**Measured PV outranks the command window.** The window is a *time* bound on a
command that has already been superseded, not evidence about the kWh being
measured now, so rule 2 is conditional on the PV floor rather than sitting
unconditionally ahead of rule 3. Without that condition a midday CHARGE -> HOLD
transition booked genuine 4 kW PV charging at the grid price for the whole grace
period — the same error as the pre-dawn case, in the opposite direction.

`cost_pv_attribution_min_w` (default 100 W) is the PV floor. The PV reading and
the grid-charge window are injected into `BatteryCostTracker`
(`get_pv_power_w_func`, `grid_charge_active_func`) rather than read off the app,
so the rule is unit-testable; without a PV provider the tracker keeps rule 5.

### Pricing of energy deltas

Each measured charge delta is priced at the slot that was active when the
energy accrued (`_last_price_slot`, recorded at the previous event), not the
slot containing the log timestamp. Consecutive deltas inside one 15-minute
price slot therefore log identical stored-energy costs, and a delta logged
just after a slot boundary still uses the previous slot's price.

### Reading the charge log

A charge event logs the delta, its attributed source, the stored-energy cost
of the delta, and the resulting weighted average:

```text
Battery charged: +0.100 kWh [inverter, pv] at stored-energy cost 0.1036 EUR/kWh,
new avg cost: 0.1128 EUR/kWh
```

The average is weighted by `_stored_energy_kwh`, an internal accumulator of
usable energy above `min_soc`. It is synced from SOC at startup and on energy
sensor recovery, then maintained by adding/subtracting measured deltas. Two
consequences when reading the log:

- Near `min_soc` the accumulator is close to zero, so each small charge is a
  large fraction of the total and the average moves quickly toward the cost of
  the fresh energy. With several kWh stored, the same 0.1 kWh delta barely
  moves it. Fast swings at low SOC are expected, not a tracking fault.
- The first charges after a deep discharge can show a `new avg cost` above the
  logged charge cost: a small expensive remnant still dominates the weighted
  average until fresh energy washes it out.

### Accumulator resync

The accumulator drifts away from the true stored energy: deltas below 0.05 kWh
are discarded as noise, midnight counter resets skip a delta, and conversion
losses are unmodelled. `_resync_stored_energy(current_soc, energy_in_transit)`
re-anchors it to the SOC-derived value in two cases:

- **Depleted.** The SOC *before* the event was at or within 1% of `min_soc`.
  This is the case that corrupts the cost basis: a charge following a genuine
  depletion must take the new energy's landed cost outright, and cannot do so
  while phantom stored energy still carries the old average. Production
  (2026-07-28 11:12) showed `Safety: HOLD (battery depleted at 10.0%)` with the
  basis stuck at 0.0009 EUR/kWh afterwards.
- **Gross drift.** The accumulator is more than `max(2 kWh, 25% of capacity)`
  from the SOC-derived value. This is a coarse safety net only. The tolerance is
  intentionally several charge slots wide: the accumulator tracks measured DC
  energy, which is the better weighting signal, and must not be pulled around
  slot by slot by a 1%-granular SOC sensor.

`current_soc` already includes the delta being processed, hence the signed
`energy_in_transit_kwh` (+charge, −discharge) that reconstructs the pre-event
state. Every resync is logged. `_compute_weighted_avg_cost` itself is correct
and is not modified — with `old_energy = 0` it already returns `added_price`.

The log line spells the in-transit term out, because consecutive resyncs at the
same SOC otherwise read as a contradiction:

```text
Resyncing stored-energy accumulator 0.100 -> 0.143 kWh (SOC 11.0% = 0.143 kWh) (depleted)
Resyncing stored-energy accumulator 0.143 -> 0.043 kWh (SOC 11.0% = 0.143 kWh less the 0.100 kWh charged in this event) (depleted)
```

The first is a plain SOC observation (`process_soc_change`, no delta), which
anchors to the SOC *now*. The second comes from the energy-delta path
(`_process_energy_change`), which anchors to the state *before* the delta. Both
are correct; only the message hid the 0.100 kWh that separates them.

### A PV basis of 0.0000 is correct

PV energy is booked at the foregone net export revenue,
`max(0, spot * export_rate_multiplier - grid_export_fee)`. Around midday, spot
frequently sits at or below the export fee, so the true opportunity cost of
storing that kWh is zero and the tracked basis correctly decays toward 0.0000.

This is why the schedule log does not use the basis as its primary number. The
first column is `ScheduleEntry.marginal_value_eur_kwh` — the slot's own
economics per battery DC kWh, computed by `DPOptimizer._marginal_slot_value`
from the same `_buy_price`/`_sell_price` helpers the DP objective uses:

| `value_basis` | Value per DC kWh |
|---|---|
| `avoided-import` | `buy * inverter_efficiency - wear` |
| `export` | `sell * inverter_efficiency - wear` |
| `landed-charge` | `-buy / (efficiency * inverter_efficiency)` (negative) |
| `kept` | the terminal rate |

The stored basis stays visible as a secondary figure (`stored 0.0000`, annotated
`[stored basis ~0: PV booked at export floor]`). These are two different
quantities and the log must keep showing both. The marginal value is REPORTING
ONLY: the DP objective never reads it, per the invariant that the DP does not
use `battery_avg_cost` as a constraint.

## Tariff and tax assumptions

Spot price and `grid_fee_eur_kwh` must have the same VAT basis. The
`import_price_multiplier` is applied to their sum: leave it at `1.0` when both
already include the desired taxes, or use (for example) `1.21` only when both
are VAT-exclusive and 21% VAT applies. Export revenue is configured separately
through `grid_export_fee_eur_kwh` and `export_rate_multiplier`. Set all three
from the actual electricity contract; the example values are assumptions, not
universal Latvian tariffs.

## Re-optimization and execution

The app performs a full optimization after tomorrow's prices are expected and
re-evaluates periodically. Material SOC deviation, refreshed prices, or changed
forecasts cause the remaining horizon to be optimized again. The selected mode
is applied through `growatt_modbus/set_wit_mode`; no raw inverter-register writes
are performed by the optimizer.

Use dry-run mode (`device_id: ""`) first and compare the schedule, SOC trajectory,
and actual inverter behavior before enabling hardware control.
