# Task: correct optimizer energy accounting, planning, and recovery

Date: 2026-09-05

Status: implementation requested by this brief; none of the fixes below have
been implemented as part of writing it.

## Objective

Fix the five correctness defects identified in the architecture review. Deliver
a schedule whose economic decisions survive orchestration, whose credited
energy is physically available under its stated model, and whose price horizon
recovers automatically after temporary data failures.

This is an implementation brief for an AI coding agent. Read the repository's
`AGENTS.md` and applicable local instructions before changing code. Preserve
unrelated working-tree changes. Implement and test the fixes; do not stop after
producing another analysis or plan.

Do not claim global optimality for the real battery system. Discretization,
forecast uncertainty, limited action choices, and any thermal approximation
remain relevant. State precisely which model the solver optimizes and which
properties the implementation guarantees.

## Scope and boundaries

In scope:

- The five defects below, their reproductions, and regression coverage.
- Refactoring required to make the relevant energy transitions consistent.
- Necessary corrections to existing tests and architecture/configuration docs.
- Local verification, simulated inverter responses, and dry-run validation
  where an appropriate environment is available.

Out of scope:

- Live deployment, changing a real inverter's settings, or issuing live mode
  commands as part of this task without separate authorization.
- Replacing the optimizer wholesale or introducing an unrelated forecasting
  system.
- Changing the user's electricity contract, tariff formulas, SOC limits, or
  terminal-value preference to hide a defect.
- Removing safety guards, increasing deviation thresholds to conceal model
  errors, or disabling verification to make tests pass.
- Automatically resetting saved learning data or unrelated user changes.

Continue to send inverter commands through `growatt_modbus/set_wit_mode`.
Preserve command acknowledgement/timeout distinctions, verification generation
guards, bounded override duration, and SOC cutoffs.

## Architecture and source map

All paths below are relative to the repository root. Symbol names are more
reliable than line numbers after implementation begins.

| Component | Main files/symbols | Responsibility |
| --- | --- | --- |
| Orchestrator | `appdaemon/apps/battery_optimizer.py`: `find_optimal_schedule`, `full_optimize`, `adaptive_optimize`, `execute_scheduled_mode` | Gather inputs, generate and modify plans, execute modes, trigger recovery |
| Planner | `appdaemon/apps/battery_optimizer_lib/dp_optimizer.py`: `DPOptimizer`, `_run_dp`, `_build_schedule`, `_discharge_index` | Economic objective, energy state transitions, partial-slot lookahead |
| Charge-rate learning | `battery_optimizer_lib/learning_engine.py`: `record_charging`, `get_charge_rate_for_soc`, `predict_charge_energy_with_warming` | Learn and predict charging capability |
| Rate precomputation | `battery_optimizer_lib/charge_rate_utils.py`: `compute_charge_rates_per_slot` | Currently predicts rates on a continuous-charging path |
| Continuous projection | `battery_optimizer_lib/soc_projection.py`: `project_slot_soc` | Expected SOC, deviation detection, and other projection consumers |
| Thermal projection | `battery_optimizer_lib/thermal_model.py`, `ambient_service.py` | Battery temperature from battery power and time-varying ambient |
| Prices | `battery_optimizer_lib/price_service.py`: `NordPoolPriceService` | Fetch, normalize, and cache price intervals |
| Execution | `battery_optimizer_lib/direct_control.py`: `DirectControl` | Apply commands, classify outcomes, verify inverter state |
| Cost accounting | `battery_optimizer_lib/cost_tracker.py` | Learn from observations and project stored-energy costs |

Abbreviated `battery_optimizer_lib/` paths in the table live under
`appdaemon/apps/`.

The main architectural problem is duplicated or inconsistent assumptions:
the DP implements its own transitions, partial-slot lookahead repeats them,
the shared continuous projector has another implementation, and the
orchestrator changes DP actions after optimization. Fix the relevant contracts
between these components, not just the visible output columns.

## Existing verification baseline

The review ran on Windows with Python 3.13.11.

```powershell
uv run pytest tests/ -q
uv run pytest tests/ -q --continue-on-collection-errors
```

Observed baseline:

- The ordinary run stopped on four collection errors.
- Continuing past collection errors yielded 991 passed tests, one failed test,
  and four collection errors.
- Collection failed in `test_algorithm.py`, `test_partial_slot_regression.py`,
  `test_soc_deviation.py`, and `test_temperature_aware_soc.py`. A fixture class
  defines a method named `datetime`, then evaluates annotations such as
  `datetime.datetime` in the class body. The method shadows the module.
- `test_deploy_script.py::test_post_deploy_check_reads_the_addon_log` failed
  against the existing locally modified deployment script. Investigate the
  current baseline before attributing this failure to optimizer changes.
- `CLAUDE.md` and `scripts/deploy.ps1` already had user changes. Preserve them.

Unblock the four collection errors with a narrow annotation/import correction,
for example postponed annotations or an unambiguous datetime module alias.
Keep compatibility with the project's declared Python versions. Do not bypass
these tests or change interpreter versions merely to hide the problem.

The five reproductions below were exercised with local Python objects and
mocked AppDaemon surfaces. They do not represent measurements of live inverter
behavior. Convert them into maintained regression tests.

## Task 1: preserve the DP's economic decisions during PV postprocessing

Priority: high.

### Defect

In `BatteryOptimizer.find_optimal_schedule`, the cloud-safe conversion changes
HOLD into discharge-to-load whenever:

```text
predicted_pv_kw > 0
and buy_price > battery_wear_cost
```

This is not sufficient to establish that the actions are equivalent or that
discharging now is economical. PV may be below household load, and stored
energy may have greater value in a later slot. Wear is only one marginal cost;
using the energy now also gives up its future use.

Rebuilding SOC and temperature trajectories after conversion makes reporting
describe the modified plan, but does not repair its economics. The rest of the
schedule was selected assuming the original HOLD action preserved energy.

### Reproduction

Use two consecutive 60-minute slots and these inputs:

| Parameter | Value |
| --- | --- |
| Capacity | 10 kWh |
| Minimum / maximum / initial SOC | 10% / 100% / 20% |
| Initially usable stored energy | 1 kWh |
| Charge / discharge power | 4 kW / 1 kW |
| Storage and inverter efficiency | 1.0 / 1.0 |
| Import fee, wear, terminal energy value | 0 |
| Export remuneration | Disabled, e.g. `export_rate_multiplier=0` |
| First slot | Spot/import price 0.10 EUR/kWh; load 2 kW; PV 1 kW |
| Second slot | Spot/import price 1.00 EUR/kWh; load 1 kW; PV 0 kW |

The raw DP chooses HOLD, DISCHARGE. It imports 1 kWh in the first slot for
0.10 EUR and uses the battery in the second slot.

The orchestrator converts this to DISCHARGE, DISCHARGE. The battery reaches
minimum SOC after the first slot, leaving the second slot's 1 kWh to the grid
at 1.00 EUR. Its projected SOC becomes 20% -> 10% -> 10%.

No cloud or forecast error is necessary. All rates and forecasts can be exact.

### Implementation direction

1. Remove the unconditional economic override. The orchestrator must not
   replace a resource-preserving action using only the current price and wear.
2. If retaining the cloud-safe behavior, restrict conversion at minimum to
   forecast-equivalent actions: PV covers the forecast load, and discharge to
   load has the same modeled energy flow and export behavior as HOLD for that
   slot. Check the actual modeled and execution semantics, not just `pv > 0`.
3. Treat protection against forecast shortfall as an explicit policy. If it
   changes expected energy use, include it in the planner's inputs/objective
   or reoptimize with the changed transition. Do not silently spend energy
   assigned to later slots.
4. Preserve minimum/maximum SOC execution guards and the existing reactive PV
   shortfall mechanism. Those do not justify arbitrary post-plan changes.
5. Update mode counts, reasons, `value_basis`, `marginal_value_eur_kwh`, and
   trajectories whenever the final action changes. A discharge entry must not
   retain an unexplained HOLD/terminal-value label.

The narrow fix does not require a new stochastic optimizer. Keep any retained
hedge explicit and explain that forecast equivalence is not equivalence under
every possible cloud event.

### Acceptance criteria

- An orchestrator-level regression reproduces the table above and preserves
  the economically correct reservation of energy.
- Tests cover `0 < PV < load`, `PV == load`, and `PV > load` separately.
- Include a later expensive slot and a positive terminal-value case so the
  tests exercise energy opportunity cost, not just wear.
- Any retained conversion is shown to preserve the modeled transition and
  reported objective for the conditions where it is allowed.
- Execution-level safety overrides remain effective.

## Task 2: establish consistent units for learned charge rates

Priority: high.

### Defect

`BatteryLearningEngine.record_charging` calculates:

```text
energy_added = measured stored energy
           or capacity * (soc_end - soc_start) / 100
learned_rate = energy_added / duration_hours
```

This learned rate describes stored-energy growth. The DP subsequently uses
`rate * efficiency * duration`, and `project_slot_soc` applies the same storage
factor to the learned rate or learned warming-aware energy. Storage loss is
therefore counted again for SOC-derived learning observations.

The fallback nominal rate and learned rate currently share one API without an
adequately explicit energy-boundary contract. Merely removing a multiplication
everywhere may fix learned rates while breaking nominal rates, grid costs, or
PV limits.

### Reproduction

Create a learning engine with a 10 kWh capacity, 4 kW nominal charge rate, and
storage efficiency 0.85. Record three identical observations:

```text
SOC: 40% -> 50%
Duration: 15 minutes
No measured battery-energy override; infer stored energy from SOC.
```

Each observation adds 1 kWh in 0.25 hours, so the learned rate is 4 kW of
stored-energy growth. Project the same charge interval from 40% using the
learned rate. The reviewed implementation returns 48.5%, although the observation
it just learned ends at 50%.

### Implementation direction

Choose and document one contract before editing consumers. A recommended
approach is to distinguish input-side charge capability from stored-energy
growth explicitly, with conversion at well-defined boundaries.

Use unambiguous names such as:

```text
charge_input_dc_kw     # before storage retention
stored_charge_kw       # rate of energy accumulated in the battery
grid_charge_ac_kwh     # purchased energy before inverter conversion
```

For the current configured efficiency interpretation:

```text
stored_charge_kw = charge_input_dc_kw * storage_efficiency
stored_energy_kwh = stored_charge_kw * duration_hours
grid_ac_kwh = grid_input_dc_kwh / inverter_efficiency
```

For charging supplied only from the grid:

```text
grid_ac_kwh = stored_energy_kwh /
              (storage_efficiency * inverter_efficiency)
```

Do not apply the grid inverter loss to DC-coupled PV by accident. Confirm the
existing PV predictor's units and conversion convention before changing the
source split. A generic name such as `energy_ac` must not conceal a stored-DC
quantity.

Required work:

1. Audit nominal rates, SOC-derived observations, measured energy observations,
   warming-aware predictions, SOC/rate plausibility bounds, and persisted
   learning records.
2. Keep the SOC-derived reproduction authoritative: its stored-energy boundary
   is unambiguous. Verify and document the meaning of inverter counters
   separately; their names alone do not prove the measurement boundary.
3. Make `get_charge_rate_for_soc` return one consistent quantity for nominal
   fallback and learned observations. Alternatively expose explicit distinct
   methods and update every caller.
4. Adapt DP transitions, partial-slot transitions, PV charging, continuous SOC
   projection, thermal power inputs, charge-slot estimates, and cost projection.
5. Preserve current tariff and efficiency semantics unless an explicit migration
   is necessary. Explain any migration and its effect on existing config.
6. Preserve historical learning observations. If their stored representation
   changes, version it and provide deterministic backward-compatible loading.
   Do not divide old values repeatedly on subsequent loads or silently discard
   the user's history. Keeping stored-energy observations in their original
   units and converting at the API boundary can avoid destructive migration.
7. Check whether efficiency learning uses genuinely independent measured grid
   energy. A synthetic `stored_energy / configured_efficiency` input is not an
   independent efficiency measurement; do not present it as one.

### Acceptance criteria

- Replaying the learned 40% -> 50% example predicts 50%, within numeric tolerance.
- Nominal fallback and an equivalent learned observation predict the same
  physical rate after accounting for their documented source units.
- Test efficiency 1.0 and a non-unit value, with and without temperature data.
- Grid-only charging, PV-only charging, and mixed-source charging conserve
  energy and bill only their corresponding AC imports.
- A grid-only example storing 1 kWh at storage efficiency 0.85 and inverter
  efficiency 0.97 imports approximately 1.21286 kWh under that model.
- Battery wear and DC-to-AC discharge accounting retain their intended units.
- Existing persisted data can load, save, and reload without double conversion.
- Update comments/docstrings and `docs/scheduling-algorithm.md` with the selected
  rate contract. Update `README.md` and `apps.yaml` if configuration changes.

## Task 3: prevent cumulative creation of available energy by SOC rounding

Priority: high.

### Defect

`_discharge_index` normally rounds the resulting energy to the nearest state.
It only falls back to conservative rounding if the index would stay unchanged
or increase. This prevents a particular sub-step free-discharge case, but not
systematic undercounting of larger discharges.

The comment that nearest rounding makes errors zero-mean is not a guarantee:
repeated similar load and slot duration produce repeated errors of the same
sign. Passing a per-slot tolerance test is insufficient.

### Reproduction

```text
Capacity: 10 kWh
Minimum / maximum / initial SOC: 10% / 100% / 30%
SOC step: 1% = 0.10 kWh
Slot duration: 15 minutes
Slots: 20
Load: 0.56 kW in every slot
PV and charging capability: zero for this isolated discharge test
Discharge cap: 4 kW
Efficiency: 1.0
Import price: 1.00 EUR/kWh throughout
Fees, wear, export remuneration, terminal value: zero
```

Each full slot consumes 0.14 kWh. The DP repeatedly deducts only 0.10 kWh.
It selects discharge across all 20 slots, although initial usable energy is
2 kWh and total load energy is 2.8 kWh. Some terminal depletion can be partial;
that does not repair the preceding accumulated error. After 15 slots its
trajectory still shows 15% SOC; actual continuous energy has already reached
the 10% cutoff.

The inverter cutoff protects the physical minimum. It does not make the
planner's credited battery supply available; the grid must cover the deficit.

### Implementation direction

Use a conservative energy representation as the initial repair:

1. A state's represented energy must never exceed physically available energy
   for the path it represents. Audit initial-state rounding as well as charge,
   HOLD/PV-charge, discharge, export, and partial-first-slot transitions.
2. Floor post-transition available energy to the grid, allowing only a small
   floating-point tolerance at exact grid boundaries. Remove nearest rounding
   where it can credit nonexistent residual energy.
3. Preserve exact observed starting energy for any explicitly continuous first
   transition, but conservatively map its result before continuing the DP.
   Apply the same rule in the partial-slot lookahead path.
4. If conservative rounding is too pessimistic for normal 15-minute loads,
   improve the state representation or supported resolution. Do not restore
   unsafe rounding to recover apparent profit. Alternatives such as residual
   energy tracking require preserving distinct future-relevant states; keeping
   a single arbitrary residual per bucket is not automatically correct.
5. Keep economic accounting consistent with energy actually delivered by each
   candidate. If a candidate is limited by remaining energy, charge the grid
   for the unmet load and distinguish that from full battery supply.
6. Replay the selected schedule continuously with the shared physical model.
   Compare credited battery service and imports, not just clamped final SOC.

Conservative quantization may discard modeled energy and underestimate value.
That is a documented accuracy/performance tradeoff, not permission to create
energy. A continuous replay may therefore have more remaining energy than a
conservative DP path. Compare like-for-like assumptions when asserting bounds,
especially after SOC-dependent rates are introduced in Task 4.

### Acceptance criteria

- The 20-slot reproduction no longer credits battery service in excess of
  available usable energy. Planned grid imports include any unmet demand.
- Check a cumulative conservation equation at every prefix of a test horizon:

  ```text
  cumulative stored-DC discharge <=
      initially usable stored-DC energy + cumulative actual stored-DC charge
  ```

- Check AC energy served after inverter loss, not just raw DC totals.
- Include starting SOC between grid points, exact grid boundaries, small loads,
  repeated fractional-step loads, export, PV charging, and partial first slots.
- Include values immediately on either side of a rounding boundary.
- Assert physical flow limits before clamping SOC. Clamping an impossible
  trajectory to `min_soc` must not make a conservation test pass.
- Review projection fields such as `dc_energy_in_kwh` and
  `dc_energy_out_kwh`: distinguish requested energy from actual capped energy
  if consumers need both. Do not treat an uncapped request as delivered energy.
- Benchmark the normal planning horizon at supported resolutions and report
  the cost of any accuracy change.

## Task 4: use charge rates compatible with candidate SOC and temperature

Priority: medium; required for a coherent physical planning model.

### Defect

`compute_charge_rates_per_slot` advances SOC and temperature as though charging
occurs continuously from now onward. `_run_dp` then uses that single rate for
every reachable state at the corresponding time.

Two opposite errors can result:

- Imaginary charging warms a cold battery, making a later planned charge appear
  faster than the selected path can achieve.
- Imaginary charging raises SOC into a taper region, making later rates too
  low even for candidate paths that remained at low SOC or discharged.

The temperature trajectory built after action selection does not feed these
corrections back into the DP. A more accurate log is not a corrected schedule.

### Reproduction

Use a controlled predictor to isolate this architectural defect:

```text
Capacity: 10 kWh; min/max/initial SOC: 10% / 100% / 10%
Three 15-minute slots
Prices: 0.60, 0.01, 1.00 EUR/kWh
Load: 0, 0, 4 kW
PV: zero
Storage/inverter efficiency: 1.0
Discharge cap: 4 kW
Fees, wear, export remuneration, terminal value: zero
Initial temperature: 0 degrees C
Rate predictor: 1 kW below 10 degrees C; 4 kW otherwise
Test charge temperature predictor: temp + duration_minutes
Test idle temperature predictor: unchanged temperature
```

The reviewed rate precomputation returns `[1, 4, 4]`. The selected schedule is
HOLD, CHARGE, DISCHARGE. Its own chosen-path temperature before charging is
still 0 degrees C, for which the rate lookup returns 1 kW, not 4 kW.

These intentionally synthetic callbacks expose the disagreement; they are not
a claim about the real pack's warming rate. Add realistic shared-thermal-model
tests as well.

### Implementation direction

Make SOC dependence part of candidate transition evaluation. Do not use a
time-only array produced by an unrelated SOC trajectory. Cache lookups by the
inputs that actually determine the result, where useful for performance.

Temperature needs an explicit design decision because it depends on history.
Choose and document one of these approaches, with its limits:

- Include discretized temperature in the DP state. This is the clearest
  formulation when thermal history materially changes feasible charge power,
  but increases state count and needs a convergence/performance assessment.
- Use a bounded solve/replay/refine process: solve, project the selected path
  with the shared model, revise rate assumptions, and solve again. Detect
  oscillation and stop after a fixed iteration budget. A fixed point is not
  proof of global optimality; require final feasibility checks and a documented
  conservative fallback when convergence fails.
- Use justified conservative thermal/rate bounds instead of path temperature.
  They must be valid over reachable conditions, including SOC tapering and
  non-monotonic rate behavior. An arbitrary fixed temperature or an assumed
  continuously warmed battery is not such a bound.

Do not keep only one temperature label for each SOC state and assume the
highest current economic value dominates all alternatives: a lower-value but
warmer path can have better future charging opportunities. State merging needs
a justified dominance rule or must be described as an approximation.

Implementation requirements common to all approaches:

1. Use the existing shared thermal model and time-varying ambient service.
   Charge and discharge can both heat the pack through actual battery power.
2. Limit warming energy to actual battery flow, including full/depleted SOC
   boundaries and partial slots. Imaginary power at a full battery must not
   create future charging capability.
3. Align the rate's units with Task 2 before interpreting thermal power.
4. Use the same treatment of within-slot warming/tapering in the candidate
   transition and final replay, or document and bound the approximation.
5. Ensure replay validates the final action sequence, including any allowed
   orchestrator conversion. Resolve a material rate/energy disagreement before
   publishing the plan; changing expected-SOC reporting alone is insufficient.
6. Keep forecasts fixed within a solve/refinement pass so changing external
   inputs cannot masquerade as a failure to converge.

### Acceptance criteria

- The synthetic example no longer relies on heat from an unselected action.
- A low-SOC path retains its appropriate charge capability even if another
  reachable path is near full and tapering.
- Cover cold idle followed by charging, discharge followed by charging, PV
  charging, near-full tapering, changing ambient, and missing temperature data.
- Test a material within-slot temperature threshold crossing.
- Chosen-path replay and planned energy agree within a specified tolerance;
  justify the tolerance independently of the observed error.
- Any iterative approach has bounded runtime, oscillation coverage, and a
  tested fallback that does not credit unavailable charge energy.
- Document whether the final solver is exact for its discretized model or an
  approximation. Report runtime at the actual supported planning horizons.

## Task 5: recover price data and maintain a usable schedule horizon

Priority: medium.

### Defect

`full_optimize` returns when prices or future prices are missing, without
scheduling a price retry. `adaptive_optimize` handles solar override and reactive
PV shortfall; it is not a regular price-refresh or horizon-extension pass.
`execute_scheduled_mode` applies HOLD when the current slot has no entry.

A transient fetch failure or missing tomorrow prices can therefore leave the
optimizer on an old/incomplete plan until an unrelated trigger or the next
daily optimization. Accurate SOC tracking does not guarantee such a trigger.

### Reproduction

Use an AppDaemon test double with enabled automation, valid constant SOC, no
manual override, zero PV, and no PV-shortfall event:

1. Make the initial price fetch return an empty list.
2. Call `full_optimize` and verify that no schedule can be generated.
3. Make a subsequent fetch capable of returning valid current/future prices.
4. Advance time, call `adaptive_optimize`, then execute the current slot.

The reviewed implementation still has only one price-fetch call, no scheduled
retry, and applies `HOLD/no_schedule`. The new price data is never requested by
these paths.

Also reproduce a successful fetch containing today but no tomorrow intervals
after the configured tomorrow-price publication window. That is a horizon
coverage problem even though the fetch returned a nonempty list.

### Implementation direction

Add one owner for price-recovery scheduling and horizon health. Reuse existing
timers where practical, and avoid overlapping retry mechanisms.

1. Define freshness using usable interval coverage and successful validation,
   not merely `_last_recalc_time`, a nonempty cache, or a fetch timestamp.
   Check the current interval, chronological continuity, and the expected
   forward horizon for the current publication window.
2. Treat missing tomorrow prices after that window as incomplete coverage.
   Before publication, do not retry continuously for legitimately unavailable
   data. Base the decision on the existing configured publication time.
3. Retry transient failures with bounded backoff, for example 30 seconds,
   2 minutes, 5 minutes, then a capped interval. These are suggested values;
   choose defaults that fit the existing callback/network costs.
4. Maintain at most one pending retry per app instance. Reset backoff after
   successful recovery. A stale timer must not replace a newer valid plan.
5. Make the periodic adaptive path also check horizon health so an absent or
   exhausted schedule can recover even if no SOC/PV event occurs.
6. Retain still-valid schedule entries and cached prices when a refresh fails
   or returns a shorter partial response. Merge/replace validated intervals
   deliberately; do not discard a known tomorrow horizon solely because a
   later service response contains today only. Respect cache age and source
   validity when retaining data.
7. On successful recovery, rebuild using current SOC, current time, and the
   remaining fraction of the active slot. Apply the result through the normal
   execution path so override/enabled checks and command tracking still work.
8. Preserve safe missing-schedule behavior while waiting. Recovery must not
   invent a cheap price and force charging to solve a coverage failure.
9. Cancel or render retries inert when disabled or terminated, and handle
   re-enable/restart correctly. During manual override, do not send automatic
   commands; any background horizon refresh must still respect that boundary.
10. Surface last successful horizon end, incomplete/stale status, and pending
    retry information through existing diagnostics where practical. Keep logs
    useful and rate-limited.

Use canonical instants for coverage and interval stepping. A local DST day can
have 23 or 25 hours; checking for exactly 96 slots is incorrect. Differentiate
an expected publication delay from a gap inside otherwise available data.

### Acceptance criteria

- Temporary empty response followed by success recovers without a SOC change,
  manual action, or another daily optimization.
- A current schedule is built and applied after recovery when automation is
  enabled and no manual override is active.
- Missing tomorrow prices trigger bounded retries after the configured window
  and stop retrying after sufficient coverage arrives.
- A failed or shortened refresh does not destroy still-valid future coverage.
- Repeated callbacks produce one pending retry, not a retry storm.
- Disable, re-enable, override, restart, and stale callback cases are tested.
- Missing current slots, internal gaps, midnight, and both DST transitions are
  covered with a deterministic clock.
- Price service failure does not make the optimizer claim a successful fresh
  horizon. Recovery callback/network latency remains bounded.

## Cross-cutting implementation rules

### Shared transitions and final-plan validation

Prefer a small pure transition API that returns enough information to evaluate
physical feasibility and economic value consistently:

```text
Inputs:
  initial stored energy, mode/action, duration, rate capability,
  load/PV, efficiencies, SOC limits, temperature/ambient as needed

Outputs:
  actual stored energy in/out, final stored energy,
  grid import, export, unmet battery-served demand,
  actual battery power/temperature information needed by consumers
```

This is a suggested contract, not a requirement to introduce a large new object
hierarchy. Reuse `soc_projection` where practical. If the DP keeps an inlined
hot path for performance, document why and prove parity against the pure model
over the relevant parameter space. Do not leave the partial-slot path untested.

Snapshots and results should describe one generation of a plan. Avoid exposing
partially replaced schedules during callbacks. Keep the existing lock ordering
and direct-control verification-generation behavior intact when adding retries
or iterative solves. Do not perform new long blocking calls while holding locks
without understanding their effect on safety and execution callbacks.

### Independent correctness checks

Tests that copy the implementation's formula can reproduce the same mistake.
Use these complementary checks:

- Observation replay: learned measurements reproduce their physical SOC gain.
- Analytical energy balance: simple constant-load/source cases have independently
  calculated imports, exports, and SOC changes.
- Small-horizon enumeration: for a small fixed action space, compare the DP's
  selected value against exhaustive feasible action sequences. Separate
  quantization error from an incorrect recurrence or postprocessing override.
- Continuous selected-plan replay: verify prefix energy conservation and actual
  demand coverage before clamping, including partial depletion.
- Metamorphic checks: under controlled assumptions, reducing available energy
  cannot create extra credited battery service, and an equivalent nominal and
  learned rate cannot change physical predictions just because of its source.

Keep tests deterministic. Use mocked services, clocks, forecasts, and thermal
predictors; no live HA connection is needed for these regressions.

## Suggested implementation sequence

1. Establish the current test baseline and fix the four fixture collection
   errors without touching production behavior.
2. Add the Task 1 reproduction and repair economic postprocessing.
3. Establish the charge-rate unit contract and implement Task 2 across all
   consumers, with persisted-data compatibility tests.
4. Implement Task 3's conservative state accounting and whole-horizon checks.
5. Implement Task 4 on top of the corrected rate/energy contract. Measure its
   performance and document approximation limits.
6. Implement Task 5's bounded recovery and horizon-health checks.
7. Run integrated verification, reconcile documentation with final behavior,
   and provide a concise handoff with remaining limitations.

Task 5 is largely independent of the energy-model work, but there must still be
one clear owner for recovery state and callback scheduling.

Do not silently expand the work to fix unrelated historical issues. Record any
new material issue separately with evidence. Do not call these five tasks
complete while their reproductions still fail.

## Verification and delivery

Run focused tests while implementing, then the full required suite after
integration:

```powershell
uv run python -m py_compile appdaemon/apps/battery_optimizer.py
uv run pytest tests/ -v
```

Relevant existing test areas include `test_dp_optimizer.py`,
`test_dp_business_semantics.py`, `test_soc_projection.py`,
`test_learning_engine.py`, `test_learning_engine_rate_bounds.py`,
`test_temperature_aware_soc.py`, `test_partial_slot_regression.py`,
`test_pv_reactive_shortfall.py`, `test_schedule_mode_counts.py`,
`test_schedule_value_column.py`, `test_price_service.py`,
`test_config_execution_guards.py`, `test_callback_lock.py`, and
`test_direct_control.py`.

Add tests in these modules or new narrowly named modules according to their
responsibility. A large number of existing passing tests is not a substitute
for the five reproductions and whole-plan checks in this brief.

For available dry-run validation, use an isolated configuration with
`device_id: ""` and inspect AppDaemon logs and `sensor.battery_optimizer`.
Do not change the active live configuration merely to run a test. If no suitable
HA/AppDaemon environment is available, state that runtime validation remains
unperformed; local mocks do not establish real hardware correctness.

Deliver:

- Production changes for all five tasks, with readable helpers and explicit units.
- Regression tests demonstrating each previously failing reproduction.
- Updated architecture and scheduling documentation; configuration docs for
  every new key, if any.
- A description of persisted-learning compatibility or migration.
- Test results that distinguish new failures from the existing deployment-script
  baseline, without reverting unrelated user edits.
- Runtime measurements for any solver/state-space/refinement change.
- A handoff explaining what was fixed, why the final energy model is consistent,
  what approximation remains, and which live/dry-run checks were performed.

Completion means the economic policy, selected actions, energy accounting,
published trajectory, and recovery behavior meet the acceptance criteria above.
It does not mean only that the schedule log looks plausible or that individual
SOC values have been clamped into bounds.
