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

### Charge-rate units — the one contract

Two different quantities were both called "the charge rate", and the confusion
cost a factor of `efficiency` on every learned observation.

| Name | Meaning | Where it appears |
| --- | --- | --- |
| `charge_input_dc_kw` | DC power at the battery terminal, **before** retention | `charge_rate_kw` in `apps.yaml`, `SocProjectionParams.charge_rate`, the DP's per-slot rate, `|P_bat|` for the thermal model |
| `stored_charge_kw` | rate at which **stored** energy grows | every learning observation, persisted as-is |
| `grid_charge_ac_kwh` | AC energy purchased to charge | the import cost |

```text
stored_charge_kw    = charge_input_dc_kw * efficiency
grid_charge_ac_kwh  = grid_charge_dc_kwh / inverter_efficiency
                    = stored_kwh / (efficiency * inverter_efficiency)
```

Storing 1 kWh from the grid at `efficiency=0.85` and `inverter_efficiency=0.97`
therefore imports **1.21286 kWh** AC. DC-coupled PV surplus charges the pack
without the grid inverter conversion and is billed at neither the import price
nor that loss.

**`BatteryLearningEngine.get_charge_rate_for_soc` is the boundary.** It returns
`charge_input_dc_kw`, for both the nominal fallback (already a terminal power)
and learned observations (stored-side, divided by the configured
`efficiency` on the way out). Every consumer can then keep doing
`rate * efficiency * duration`, and a nominal rate and an equivalent learned
observation predict identical physics.

Consequences worth stating, because the defect was invisible without them:

- Replaying a learned observation reproduces it. A 40 % → 50 % charge in 15 min
  on a 10 kWh pack projects to 50 %, not 48.5 % as before.
- The conversion uses the **configured** `efficiency`, not `learned_efficiency`,
  because that is the constant every consumer multiplies back in. Dividing out
  with one factor and multiplying back with another would reintroduce the
  mismatch.
- Persisted learning files are **unchanged**: observations always were, and
  remain, `stored_charge_kw`. There is no migration and no repeated division on
  reload. `tests/test_charge_rate_units.py` pins save → load → save → load.
- `learned_efficiency` is **not** currently a measurement. The only input it
  could be learned from is an independent AC meter reading for the charge
  interval, and there is none; `cost_tracker` used to pass
  `stored_energy / configured_efficiency`, whose quotient with the stored energy
  is the configured constant by construction. That input was removed, the
  tautological value is rejected, and `learned_efficiency` stays at the
  configured value until a real measurement exists.

### The shared slot transition (`slot_energy.py`)

`slot_energy.simulate_slot` is the one pure function that answers *where every
kWh of a slot came from and went to*: stored energy in/out, the PV and grid
shares of a charge, AC served, grid import, export, and the AC demand a plan
credited to the battery that the battery could not supply
(`unmet_battery_ac_kwh`). It has no clock, no forecast and no learning engine —
callers pass the rate capability they decided on.

`soc_projection.project_slot_soc` delegates to it, so the SOC view and the
energy view of a slot cannot drift apart. It reports what the pack **actually**
stored or delivered; the uncapped request is kept separately in
`requested_dc_energy_*`. Treating a request as delivered energy is precisely how
credited energy gets created out of nothing.

## SOC transitions and discretization

The state is `(time, discretized stored energy)`. Each reachable state retains
the best cumulative value and a predecessor for backtracking. Transitions obey
`min_soc`, `max_soc`, charge/discharge power limits, PV availability, and the
remaining fraction of the current interval.

### Conservative quantization: bucket label plus exact path energy

A DP state used to *be* the energy of its grid point, and a discharge was
rounded to the **nearest** grid point. Nearest rounding is unbiased only for a
random signal. A constant load on a constant slot length produces the same
error with the same sign in every slot: at 0.14 kWh per slot on a 0.10 kWh grid
the planner deducted 0.10 kWh twenty times and credited 2.8 kWh of battery
service from a battery holding 2.0 kWh, publishing 15 % SOC after 15 slots for
a pack its own model had already emptied. The old no-free-lunch guard caught
only the sub-step case, never the systematic one.

Rounding **down** instead is safe but far too pessimistic on its own. Measured
on that same reproduction, floor-to-grid serves 1.40 kWh of the 2.00 kWh
available and discards 30 % of it, discharging for 10 slots instead of 15. A
15-minute slot on the reference installation routinely moves barely two grid
steps, so that is the normal case. Buying the accuracy back by shrinking
`soc_step_percent` costs proportionally more states and CPU — over a 132-slot
horizon with a partial first slot: 1 % → 125 ms, 0.5 % → 241 ms, 0.25 % →
527 ms.

So the grid stays where it is and each state carries **both**:

- an **index**, floored, so a state's label never claims energy the path does
  not hold. The index is what merges paths — it is the resolution of the
  *optimization*.
- `dp_energy[idx]`, the **exact** continuous energy of the best path reaching
  that bucket. Every transition is computed from it.

The consequences are worth being precise about:

- **Physics is exact.** No transition can create a joule; the initial observed
  energy is kept exactly; replaying the backtracked plan continuously reproduces
  the planner's own trajectory to floating-point precision.
- **The merging is the approximation, and it is not a valid state reduction.**
  Two paths in one bucket differ by up to one step and only the better-valued
  one survives. **A higher-valued path does not dominate a lower-valued path
  holding more energy** — the extra energy can be worth more later than the
  value gap is worth now. The solver is therefore *not* exact for its
  discretized model. That claim was made here and in the code; it was wrong.

  **The counterexample**, at the default 1 % step, pinned as a named regression
  test (`tests/test_merge_approximation.py`): 10 kWh pack, min SOC 10 %, initial
  10.9 %, two 15-minute slots drawing 0.2 kW at 0.10 then 1.00 EUR/kWh, unit
  efficiencies, no fees, wear or terminal value. The solver returns
  `DISCHARGE, DISCHARGE` costing **0.010 EUR**; exhaustive enumeration of the
  same action space finds `HOLD, DISCHARGE` at **0.005 EUR**. Both paths land in
  bucket 0 after slot 1 — `DISCHARGE` holding 1.04 kWh at value 0, `HOLD`
  holding 1.09 kWh at value −0.005 — so `DISCHARGE` wins the merge and the extra
  0.05 kWh, worth 1.00 EUR/kWh in slot 2, is discarded with the losing path. At
  a 0.1 % step the two no longer share a bucket and the solver finds 0.005,
  which is the evidence that what is lost is merge resolution and not
  correctness of the transition.

  **A finer grid is not reliably better.** The merge error is *not monotone* in
  `soc_step_percent`. Measured on a 5-slot case (prices 0.6450 / 0.9446 / 0.6896
  / 0.7114 / 0.0915 EUR/kWh, load 1.35 kW, 1 kW charge capability, initial SOC
  14.75 % on a 10 kWh pack) the gap against exhaustive enumeration is 0.0000 EUR
  at a 2 % step, **0.0092 EUR at 1 %**, and 0.0000 again at 0.5 %, 0.25 % and
  0.1 %. Halving the step moved a path across a bucket boundary and lost it.
  Neither the pinned counterexample above nor
  `docs/dp_optimization_parameters.md`'s 0.25 % recommendation should be read as
  "finer is safer".

  **When the approximation bites.** It needs a physical limit (`min_soc` or
  `max_soc`) truncating a transition inside the horizon, so in practice a nearly
  empty or nearly full pack. A randomised sweep found a gap in 35 of 300 cases
  with an initial SOC of 10-20 % (worst 0.060 EUR over five slots) and in **0 of
  600** cases with an initial SOC of 50-95 %. The pinned example is the *mildest*
  of its family: at an initial SOC of 10.5 % rather than 10.9 % the same two
  slots cost 0.045 EUR of gap, and a 0.9 % step already recovers the pinned case
  — 0.1 % is nowhere near necessary for it.

  **Size of the gap.** Per merge, the energy loss is less than one step (a
  bucket is one step wide) and the value loss is at most
  `step * marginal_value` of a kWh. Over the horizon the only bound established
  here is the sum, `n_slots * step * marginal_value`: a discarded path can be
  discarded again at every later slot. That is loose — the errors are not
  independent and a merged path usually rejoins — but nothing in this
  implementation proves anything tighter, and the honest statement is the loose
  one.

  Ties are broken toward the path holding more energy. *That* is a genuine
  dominance rule (equal value, more energy can only widen later options); it
  makes the merge deterministic and it is not the part that is approximate.

  **What IS exact:** the physics of every transition, at the path's exact
  continuous energy; replay parity, so `plan_validation.replay_plan` walking the
  published action sequence reproduces the planner's own SOC trajectory; the SOC
  dependence of the charge rate; and the value arithmetic of a given action
  sequence under a given temperature profile. Not the search.

  **The exact alternative, and why it is not used.** Keeping a
  Pareto-nondominated set of `(value, energy)` labels per bucket — discarding a
  label only when another in the same bucket has *both* at least as much value
  and at least as much energy — is exact for the discretized model, because that
  is the real dominance relation. It is not adopted because the label count per
  bucket is unbounded without a cap: every distinct value that survives on the
  energy axis is a separate label, and the DP's inner loop is already the
  hot path (a 132-slot horizon at a 1 % step runs the whole DP once per
  partial-first-slot candidate). A capped variant — keep the best *k* labels by
  value, or bucket the value axis too — reintroduces an approximation with a
  second tuning knob and no better bound than the one above. If it is ever
  revisited, `tests/test_merge_approximation.py` is the test that must change,
  deliberately, with the new rule's runtime measured.
- **Energy-limited candidates pay the grid.** A discharge delivers what the pack
  has; whatever it cannot cover is charged to the grid at the import price in the
  same slot. No threshold decides whether a slot "counts" — the energy does. The
  old `> min_energy + dc * 0.5` and `* 0.3` thresholds are gone.
- **The published SOC trajectory is built from the path's energies**, not from
  grid indices.

`soc_step_percent` therefore now controls only how aggressively distinct paths
are merged, not how much energy the plan may invent.

### Final-plan replay

`plan_validation.replay_plan` walks the **final** action sequence — after any
orchestrator postprocessing — through `slot_energy.simulate_slot` in continuous
energy, and `BatteryOptimizer._validate_final_plan` runs it as the last step of
`find_optimal_schedule`, on the plan that will actually execute. It checks, at
every prefix and **before** any SOC clamping:

```text
cumulative stored-DC discharge
    <= initially usable stored-DC energy + cumulative actual stored-DC charge
```

plus the AC energy served after inverter loss, the AC demand the plan assigned
to the battery that the battery could not supply, and agreement with the
trajectory about to be published.

Validation runs **before** the mode census, the projected-cost column and the
decision log are derived, because `_resolve_plan_shortfall` can still change the
plan at that point — it reverts an unserviceable cloud-safe hedge slot back to
`HOLD`. Deriving those three first published a census, a cost column and a
decision log describing a schedule that `find_optimal_schedule` does not return.

The trajectory tolerance is one tenth of a DP grid step. That number is derived
from the representation, not fitted to an observed error: the planner and the
replay evaluate the same closed-form transition on the same fixed forecasts and
the same per-slot rates, so they can differ only by floating-point accumulation
— on the order of `n_slots * 2.2e-16 * capacity`, about 1e-12 SOC % over a
132-slot horizon. A tenth of a step is astronomically above that and far below
anything quantization could explain, so a breach is a model disagreement.

A breach is reported (ERROR for conservation, WARNING for trajectory
disagreement) and never silently rewrites the plan.

### Charge rates that match the SOC and temperature the plan reaches

The rate used to come from a time-indexed array built by advancing SOC and
temperature *as if charging ran continuously from now*, and the DP then applied
that one number to every reachable state at that time. Both halves were wrong,
in opposite directions:

- imaginary charging warmed a cold pack, so a later planned charge looked faster
  than the selected path could achieve;
- imaginary charging pushed SOC into a taper region, so paths that stayed low or
  discharged were denied capability they actually had.

**SOC dependence is exact.** The rate is evaluated per candidate transition,
from that state's own energy, memoized by `(soc, slot temperature)`. There is no
"the rate looks SOC-independent, hoist it out of the state loop" fast path: it
existed, it decided from a fixed probe set, and a curve that was flat at every
probe but tapered at a temperature the *refined* profile reached had its taper
erased — the plan then invented 1.875 kWh in one slot with `converged=True`.
`charge_rate_predictor` is an arbitrary callable; a finite sample of it proves
nothing.

**Temperature depends on history**, which a one-dimensional energy state cannot
carry, so it needs an explicit design decision. The chosen one is a **bounded
solve / replay / refine**:

| Pass | What it plans with |
| --- | --- |
| 0 | the **idle** temperature profile — the pack is only as warm as it would be with no battery activity at all, so no heat can come from an action the plan has not committed to |
| n | the profile produced by replaying the *selected* plan through the shared `TemperatureProjector`, with warming driven by **actual** battery flow |

**The stopping criterion is FEASIBILITY AT THE REACHED TEMPERATURE**, not a
fixed point. After each pass, `DPOptimizer._replay_plan` walks the selected plan
forward, looks the rate up at the temperature the pack has actually reached in
that walk, and reports how much of the charge energy the plan credited the pack
could not have taken — over CHARGE and over the PV absorption a HOLD or
self-consumption DISCHARGE performs, since `simulate_slot` caps all three with
the same `charge_input_dc_kw`. A plan is converged only when that shortfall is
zero **and** the profile is stable.

A fixed point on its own was not enough, and the gap was not theoretical. The
loop used to be gated by a sampled "is this rate curve temperature sensitive?"
probe over three SOCs (min, mid, max) and a six-point temperature ladder. A
learned bucket that varied only at a SOC *between* those probes skipped
refinement entirely: a three-slot CHARGE, CHARGE, DISCHARGE plan came back
`converged=True` after one pass, and replaying it at the temperatures it reaches
left **0.75 kWh of the final slot's load uncovered**. Production validation
reported nothing, because it looked the rate up at
`planning_temp_by_slot` — the planner's own assumption. The probe is gone: with
a temperature reading the refinement always runs; without one there is nothing
to refine and a single solve is the whole answer.

Two profiles count as the same when their temperatures agree within 0.25 C
**or** when they imply the same charge capability at every state of the DP's SOC
grid — temperature only reaches the plan through the rate, and a pack that keeps
warming inside one temperature bucket would otherwise never settle.

On oscillation or exhaustion it falls back to a **conservative solve**: per
(slot, SOC) the *minimum* rate over every temperature profile seen in this call,
followed by one more replay. **Limits of that fallback, stated rather than
implied:**

- "Minimum over the profiles seen" is a lower bound over *those profiles only*.
  If the conservative plan reaches a temperature none of them visited and the
  curve dips there, it can still be short — which is why the replay after it is
  not optional. `tests/test_thermal_feasibility_refinement.py` covers both: a
  set of profiles that covers the reached temperatures (fallback succeeds) and
  one that does not (it degrades).
- If it is still short the branch **degrades**: the credited charge energy is
  reduced to what the replayed temperature allows, the trajectory is rebuilt
  from that walk, and it is logged at WARNING with the shortfall in kWh.
  **Economic optimality is lost in that branch** — the actions were chosen for
  energy the pack will not have. What survives is that nothing published credits
  unavailable energy.
- A fixed point is not a proof of optimality of any kind. See § Conservative
  quantization for what the solver does and does not guarantee.
- A conservative fallback under-charges rather than over-charges *in its
  decisions*. What is published is the physical walk, so the trajectory is not
  pessimistic even when the decisions are.

`DPOptimizerResult` says which path produced the plan:
`rate_refinement_branch` is one of `single_solve`, `converged`,
`conservative_fallback`, `degraded`; `rate_refinement_shortfall_kwh` carries the
kWh that triggered a degrade; `rate_refinement_converged`,
`rate_refinement_fallback` and `rate_refinement_degraded` are the same
information as booleans.

**One trajectory, and it is the physical outcome.** Whatever branch chose the
actions, what gets published is the walk of those actions at the temperatures
they reach. In the converged branch that is identical to the DP's own energies
by construction. In the conservative branch it is the *outcome* rather than the
pessimistic assumption the decisions were made on — the inverter will charge at
whatever the pack can take. This is also exactly what
`plan_validation.replay_plan` and `BatteryOptimizer.project_schedule_trajectory`
compute, which is why **no consumer pins a charge-rate lookup to a planning
temperature any more**. `DPOptimizerResult.planning_temp_by_slot` survives as a
diagnostic only.

**Orchestrator validation.** `BatteryOptimizer._replay_schedule` looks the rate
up at the temperature the replay reaches, resolved exactly as
`soc_projection._effective_charge_rate` does it (learned rate when the engine
has one, configured nominal otherwise). If the final, post-hedge plan still
credits the battery with AC service it does not have,
`_resolve_plan_shortfall` first **reverts the cloud-safe conversions on the
affected slots** — the hedge only converts slots the DP's model cannot tell
apart from HOLD, so a replay that disagrees indicts the conversion — and
re-validates. If it is still short, those slots are declared energy-limited
(which is the truth: they deliver what the pack has and the grid pays for the
rest), the replayed trajectory is published, and it is logged at ERROR. A plan
that credits charge energy unavailable at the replayed temperature is never
published.

Warming follows `simulate_slot`'s actual `battery_power_kw`, so a full pack
ordered to charge — or an empty one ordered to discharge — moves nothing and
warms nothing. Imaginary power must not manufacture future charging capability.
Forecasts are computed once and reused across every refinement pass, so a moving
input cannot masquerade as a failure to converge.

### Within-slot charge model — one of them, in every consumer

**A charge slot runs at a constant `charge_input_dc_kw` for its whole length.**
The rate does not change inside a slot; temperature evolves *between* slots, and
only through `thermal_model.TemperatureProjector`.

**Which constant** is decided by one helper, `slot_energy.charge_rate_for_span`:

```text
r_start   = rate(soc_start, temp_start)
soc_end   = min(max_soc, soc_start + r_start * slot_hours * efficiency)
rate      = min(r_start, rate(soc_end, temp_start))
```

i.e. the rate is evaluated at the SOC the slot starts from **and** at the SOC
that rate would reach, and the slower of the two runs for the whole slot. The
end-of-span probe steps back from `soc_end` by a billionth of the span, because
the pack is at `soc_end` for zero seconds: a slot that exactly fills a learned
bucket (a clean 40 % → 50 % calibration observation) must still replay its own
measurement.

Freezing the rate at the start SOC instead over-credited every slot that crossed
one of the learning engine's SOC-taper buckets (25 / 50 / 75 / 90 %), and no
validation could catch it, because `plan_validation.replay_plan` and
`DPOptimizer._replay_plan` evaluated the same frozen model. Measured on a 10 kWh
pack with a 4 kW → 1 kW taper at 90 % and a 15-minute slot from 88 %: 1-minute
sub-stepped truth 92.0 %, frozen model **98.0 %** — six SOC points of energy
credited to a plan that could not take it. The taper at 25 % from 22 % gave
+5 points the same way.

Every consumer calls that one helper:

| Consumer | Where |
| --- | --- |
| DP candidate evaluation and the partial-slot lookahead | `dp_optimizer._run_dp`, `_build_schedule` |
| the DP's feasibility/temperature replay | `dp_optimizer._replay_plan` |
| the pure physical transition | `slot_energy.simulate_slot` (given the span rate) |
| final-plan replay | `plan_validation.replay_plan` |
| expected-SOC trajectory and the deviation detector | `soc_projection.project_slot_soc` |
| projected-cost column | `cost_tracker.project_costs` |

The DP memoizes it by state index within a slot, as before; it is now two rate
lookups per state per slot instead of one, and both go through the same
`(soc, temperature)` cache.

There used to be a second one. `project_slot_soc` called
`learning_engine.predict_charge_input_dc_energy`, which split a CHARGE slot into
a cold phase and a warm phase using the learning engine's own warming-rate model
(`get_time_to_reach_temp`, `predict_temp_after_duration`) — a second thermal
model, reached only on that one code path. On a 10 kWh pack at 10 % SOC with a
single 15-minute CHARGE crossing 1 kW → 4 kW halfway, the DP answered 12.5 % and
the projector answered 16.25 %. The published trajectory therefore disagreed
with the plan by 3.75 SOC points on one slot, and the deviation detector raised
SOC shortfalls against a battery that was following the planner exactly.

`predict_charge_input_dc_energy` survives in the learning engine, marked
**diagnostic only**; nothing in planning or projection calls it. Its
`temp_threshold` argument was never an `apps.yaml` key — it was a parameter of
`project_slot_soc` with a default of 16 °C — and it has been removed from that
signature rather than left accepted-and-ignored.

**What the model claims, and which claims can fail.** Two different kinds of
statement, kept apart on purpose — the bound that used to be documented here
conflated them and was wrong under either reading.

*An identity.* Against a 1-minute sub-stepped reference through the same rate
curve, the error per slot is at most

```text
(max rate visited - min rate visited) * slot_hours * efficiency        kWh stored
(max rate visited - min rate visited) * slot_hours * efficiency / capacity * 100  SOC points
```

This cannot fail and proves nothing on its own: the slot runs at one rate drawn
from the set the truth visits, and the truth is a duration-weighted average of
that same set. It is stated because it is the only bound that survives a
non-monotonic rate curve, and because it is the number a reader wants when a
direction claim does not apply.

*Falsifiable direction claims*, and the conditions they need:

- **Exact** for a piecewise-constant bucket rate when no bucket boundary falls
  strictly inside `[soc_start, reached_soc)`. Both probes then return the same
  rate and there is nothing to approximate. On the reference pack a 15-minute
  slot moves at most about 8 SOC points against 25-point buckets, so most slots
  clear the buckets entirely — but a slot that *does* cross one is not exact,
  only conservative: 88 % over a 4 kW → 1 kW taper at 90 % gives 90.5 % against
  a sub-stepped truth of 92.0 %, and 22 % over a 1 kW → 4 kW step at 23 % gives
  24.5 % against 29.0 %. "At most one boundary" was the condition for the rule
  being well-behaved, never for it being exact.
- **Conservative** (never over-credits) when the rate is monotone over the SOC
  span the slot covers: the minimum of the two endpoints is then a lower bound
  on every rate the slot visits.
- **Conservative** when the pack *warms* during the slot and the rate is
  non-decreasing in temperature — the physical case while charging, since the
  rate is looked up at the start-of-slot temperature.
- **May over-credit**, and bounded rather than eliminated, on a pack that
  *cools* while charging **and the rate is non-decreasing in temperature over
  the range the slot traverses and monotone over the SOC span the slot
  covers**. Temperature is not spanned (the DP's 1-D energy state cannot carry
  it), so the over-credit is then at most
  `(rate(T_start) - rate(T_end)) * slot_hours * efficiency`: every temperature
  the slot visits is between `T_end` and `T_start`, so every rate it visits is
  between the two endpoint rates. It is the same monotonicity direction the
  warming bullet above needs, and only this direction produces an over-credit
  at all — a rate that *falls* as temperature rises makes a cooling pack faster
  than the rate it was looked up at, so the model under-credits and there is
  nothing to bound. Without the temperature monotonicity that bound compares
  two endpoints of a curve the slot leaves:
  a pack cooling 20 → 5 °C on a curve of 2.0 kW at or above 19 °C, 0.1 kW from
  11 to 19 °C and 1.9 kW below 11 °C has both endpoints fast and the middle
  slow — the model says 25.0 %, the sub-stepped truth is 22.4 %, and the
  endpoint bound allows 0.25, a violation by a factor of ten and a half. Only
  the identity above survives there.

  The SOC condition is the same trap one axis over: `charge_rate_for_span`
  probes two SOCs at *one* temperature, so a rate that dips in SOC between the
  probes and recovers by the reached SOC passes the minimum test unchanged.
  4 kW outside 11-19 %, 0.1 kW inside it, plus a 0.008 kW/°C slope, cooling
  20 → 5 °C from 10 %: the model over-credits by 8.59 SOC points against an
  endpoint bound of 0.30, a factor of 29. And the model does not always
  over-credit a cooling pack — that holds only for SOC-independent curves. A
  plain 4 → 1 kW taper at 26 % with the same slope, from 20 %, *under*-credits
  by 4.15 points, because the reached-SOC probe lands past the taper and the
  slow rate is applied to the whole slot. Both stay inside the identity bound.
- **An approximation otherwise**, with only the identity above. The pinned
  counterexample: a non-monotonic curve of 1.0 kW below 14 °C, 6.0 kW from 14 to
  20 °C and 1.2 kW above 20 °C, on a pack warming 1 °C/min from 10 °C. The
  sub-stepped truth is 17.667 %, the model says 12.5 % — an error of **5.17 SOC
  points** against the 0.5 points the old "rate at the end minus rate at the
  start" bound allowed, a violation by a factor of ten. The identity bound for
  that slot is 12.5 points, and it holds.

`tests/test_within_slot_charge_model.py` measures all of these against the
sub-stepped reference: the direction claims on monotone curves, the identity on
the non-monotonic counterexample, and the taper scenarios above in every
consumer.

A finer model is possible — N sub-steps of the shared projector — but it would
have to be *the same code path* in all five consumers above, and it would
multiply the DP's inner loop by N. The constant rate is the cheap end of that
trade, taken deliberately and bounded here.

Two designs were rejected:

- **Discretized temperature in the DP state.** The clearest formulation, but it
  multiplies the state count by the number of temperature buckets, and the
  partial-first-slot lookahead already runs the whole DP once per candidate. The
  normal 132-slot horizon would go from ~140 ms to well over a second on the
  single AppDaemon thread.
- **A fixed conservative temperature.** "Coldest plausible" is not a valid bound
  over reachable conditions once SOC tapering and non-monotonic rate behaviour
  are in play, and it would refuse to plan the warm-pack charging the
  installation actually does.

**Runtime.** 132-slot horizon (33 h at 15-minute slots), 14.3 kWh pack, median
of five runs, without / with a partial first slot. Absolute numbers are
machine-dependent — a reviewer measured roughly twice these on their hardware —
so read the shape, not the digits:

| Rate curve | 1 % step, before | 1 % step, after |
| --- | --- | --- |
| no temperature reading (single solve) | 48 / 188 ms | 45 / 179 ms |
| SOC + temperature, converges in 1 pass | 72 / 202 ms | 66 / 198 ms |
| a curve that forces refinement | 222 ms (4 passes) / 429 ms (2) | 274 ms (4) / 984 ms (4) |

Earlier measurements on a 14.3 kWh pack, for the shape across step sizes:

| Rate curve | 1 % step (91 states) | 0.5 % (181) | 0.25 % (361) |
| --- | --- | --- | --- |
| flat | 54 / 187 ms | 97 / 369 ms | 194 / 786 ms |
| SOC taper | 53 / 203 ms | 100 / 403 ms | 201 / 1004 ms |

Three things the table is there to say:

- **The worst case is bounded, real, and it got worse.** At most
  `MAX_RATE_REFINEMENT_PASSES + 1 = 4` solves, and the partial-first-slot
  lookahead runs the whole DP once per candidate on top of that. Requiring
  feasibility rather than only a fixed point means a curve that used to settle
  in two passes can now use the full budget: on the measured staircase curve
  with a partial first slot, 429 ms became 984 ms. Converging cases are
  unchanged, and the removed sensitivity probe pays for itself there. Around
  1 s at the reference installation's 1 % step in the worst case.
- **It runs under the app callback lock**, on AppDaemon's single thread, so it
  delays every other callback of this app for that long. That is the argument
  against putting temperature into the DP state, and the reason
  `MAX_RATE_REFINEMENT_PASSES` is 3 rather than "until it settles".
- **Whether the refinement converges depends on `soc_step_percent`.** In the
  bottom row a 1 % grid oscillates and falls back while 0.5 % and 0.25 % reach a
  fixed point in two passes: a finer grid changes which plan each pass selects.
  The step is an accuracy/performance control with a third effect.

These are planning estimates; an actual SOC deviation still triggers a
re-optimization.

### One bound on the learned charge rate

`BatteryLearningEngine.get_charge_rate_for_soc` is the ONE gate every consumer of
a learned battery power passes through: the DP (via `DPOptimizer._rate_for`,
once per candidate state), the expected-SOC trajectory
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
   `[-10:]` slice, and `_bounded_input_dc_rate` converts the median to terminal
   power and clamps the result. `load_from_json`
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
| DP | the time-indexed rate array, at a projected SOC/temp | 10.95 kW | +18.2 %/slot planned |
| expected-SOC trajectory | `project_slot_soc` at 10 %, 16 C | 3.01 kW | +5.0 %/slot |
| deviation detector | `_project_charge_completion` at 10 %, 21.9 C | 14308.71 kW | "projected to reach 21894.1 %" |

The DP's 10.95 kW is the tell. That table is history: the rate array it names
walked the SOC forward assuming continuous charging, so the first slot's
14308.71 kW saturated the projection at 100 % and *every remaining slot* was
priced from the `90-100`/`>20` bucket (median 10.95 kW) instead of the bucket it
would really be in. Reality delivered 6.86 kW (SOC 9 %→21 % in 15 min). The
array is gone — the DP evaluates the rate per candidate state — but the bound is
what stopped the 14308.71 kW, and the bound is what this section is about.

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
3. **One projector, every consumer.** `DPOptimizer._idle_temp_profile` and
   `DPOptimizer._replay_plan_temps`, `soc_projection.project_slot_soc` (used by
   the expected-SOC trajectory, the projected-cost column and the deviation
   detector) and `ScheduleFormatter` all go through the same
   `TemperatureProjector`. Two different models on two code paths is the bug
   this replaced.
4. **Temperature reaches the DP's decisions through the refinement loop.**
   `_idle_temp_profile` supplies pass 0 and `_replay_plan_temps` each later
   pass; the per-slot temperature then feeds `get_charge_rate_for_soc(soc, temp)`
   inside the candidate transition. The temperature trajectory in
   `DPOptimizerResult` is the last pass's replay, so the reported trajectory and
   the one the plan was priced at are the same object — they used to be built by
   two different functions, one before and one after `_build_schedule`.
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
   while `hold` leaves it open. So either the sell price is zero, or the plan
   was not selling anything in that slot. The check reads the **pre-hedge
   replay's own `grid_export_ac_kwh`** (`plan_validation.replay_plan`, built by
   `BatteryOptimizer._replay_schedule` — the same construction
   `_validate_final_plan` uses), whose charge-rate lookup is pinned to
   `DPOptimizerResult.planning_temp_by_slot`, i.e. the temperatures the DP
   actually priced the slot at.

   It used to infer absorption from `project_schedule_trajectory`'s SOC span
   instead. That looks the charge rate up at the *projector's* own evolving
   temperature, which is a different plan whenever the rate refinement falls
   back to its conservative idle profile: on a cold pack the re-projection
   "absorbs" a surplus the DP had booked as export revenue, the slot converts,
   and `discharge_to_load` curtails the sale the schedule was chosen for.

   A full pack with a **sellable** surplus therefore never
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

## Price coverage and recovery

A non-empty fetch is not a usable horizon, and `_last_recalc_time` is not
freshness. `battery_optimizer_lib/price_horizon.py` owns one verdict
(`PriceHorizonMonitor.evaluate`) built from three questions, all answered on
canonical UTC instants:

1. **Is the current interval present?** The interval whose start equals the
   current slot boundary must exist. Without it there is nothing to execute.
2. **Is the data continuous from there?** Intervals must step forward by exactly
   `slot_minutes` of elapsed time. The first break ends the usable horizon.
3. **Does it reach far enough?** Before the configured `tomorrow_prices_hour`,
   the required end is the next local midnight. From that hour on, it is the
   midnight after that. A reply containing today only is therefore *complete*
   at 10:00 and *incomplete* at 15:00.

The required end is a local midnight converted to its instant, never a slot
count: a Europe/Riga spring day needs 92 fifteen-minute intervals and an autumn
day needs 100. The verdict distinguishes a `gap` (data exists past the break -
a hole in otherwise available data) from `tomorrow_missing` (nothing past the
break, and publication was expected).

That midnight is computed in a zone **with DST rules**, taken from AppDaemon's
`get_timezone()`. `_get_local_timezone()` cannot be used for it: it falls back
to `datetime.now().astimezone().tzinfo`, a fixed `datetime.timezone` carrying
today's offset, whenever `self.datetime()` is naive — and
`combine(2024-04-01, 00:00, +02:00)` is an hour later than Riga's real midnight
that day. Left uncorrected, a complete horizon read as `tomorrow_missing` for
the whole spring-transition afternoon, and an incomplete one read as complete
every autumn. When no region zone can be resolved the app falls back to the
offset and says so once at WARNING.

An unusable verdict is *noted*, not acted on: `tomorrow_missing` is the normal
state from `tomorrow_prices_hour` until tomorrow publishes, and that window sits
inside the PV day. The adaptive pass records the failure, arms the retry, and
continues to the reactive PV-shortfall check.

### Recovery

An unusable verdict arms **one** pending retry with a bounded backoff
(`price_retry_delays_seconds`, default 30 s / 2 min / 5 min, then
`price_retry_max_seconds`). Rules:

- At most one pending retry per app instance. Every path that can notice the
  same missing horizon in the same minute - the daily optimization, the slot
  execution's `no_schedule` HOLD, the periodic adaptive pass - shares it, and
  none of them advances the backoff while it is armed.
- The retry carries a generation token. Disabling the optimizer, terminating
  the app, or a successful recovery clears the token, so a timer the scheduler
  has already queued cannot replace a newer valid plan.
- On success the backoff resets and the schedule is rebuilt from the **current**
  SOC and time (the active slot contributes only its remaining fraction)
  through `_recalculate_remaining_schedule`, which finishes with the normal
  `execute_scheduled_mode` path. Enabled and manual-override checks therefore
  still apply: during an override the plan is refreshed and no command is sent.
- While waiting, a slot with no entry stays `HOLD`. Recovery never invents a
  price and never forces charging to paper over missing data.

### The current interval when nobody published a price for it

Coverage question 1 above can fail on its own: the horizon is otherwise fine,
but the interval the app is *living in* is not in the data. The planner used to
manufacture one - yesterday's same-clock price, else the most recent past
price, else the next price - and then plan, log and **execute** the live slot on
it. With yesterday at 0.01 EUR/kWh and the real next interval at 1.00, that is a
grid-charge command issued at a price that does not exist, and an armed retry
does not prevent the command.

The premise was that "Nord Pool may exclude the current hour as past". It does
not hold for either fetch path: `_get_prices_via_service` asks
`nordpool.get_price_indices_for_date` for whole **dates**, and
`_get_prices_via_sensor` reads the whole-day `raw_today` / `raw_tomorrow`
attributes. Both deliver the elapsed part of the day, so a missing current
interval means the data is genuinely missing.

What happens instead:

- **Planning starts at the next validated interval.** The DP is given the
  prices as fetched; it finds no index for the current slot, so it solves the
  remaining slots at full width with no partial first slot. The horizon is not
  lost because one interval is.
- **The current slot is resolved BEFORE the solve, not after it — the whole
  decision.** `_resolve_unpriced_current_slot` picks the entry that will run
  for the rest of this quarter hour (the retained one, or the `HOLD` fallback,
  which still absorbs PV surplus), that entry is walked through
  `soc_projection.project_slot_soc` for the remaining fraction of the slot, the
  DP is handed the SOC and temperature that walk ends at, and the entry joins
  the schedule **before** the cloud-safe hedge, `_validate_final_plan`, the
  mode census, the projected-cost column and the decision log.

  Solving from the SOC measured mid-slot modelled the retained action as doing
  nothing: a retained `CHARGE` running 10:07 → 10:15 at 4.5 kW × 0.85 adds
  about 3.6 SOC points on the 14.3 kWh reference pack that the plan never saw,
  and a retained `DISCHARGE` errs the other way.

  Deciding the entry *afterwards*, in the two planning paths, was the same
  ordering defect one step later: on the 10:07 fixture the mode census reported
  0 charge slots against a schedule holding a retained `CHARGE`, the final-plan
  replay covered 55 of 56 slots — the missing one being the slot actually sent
  to the inverter — and the projected-cost column had no row for it. **Nothing
  writes to the schedule after `_validate_final_plan`**, and no planning path
  adds an entry the planner did not produce.
- **A previously planned entry is retained** if it was itself built from a
  published price. `ScheduleEntry.price_source` records that provenance
  (`"market"`), stamped when the schedule is built against the price keys the
  DP was actually handed. A retained entry is still subject to every execution
  guard - the enable switch, the manual override, the min-SOC and max-SOC
  overrides. Retention keeps a decision made on real data; it does not exempt
  it from anything.
- **Otherwise the slot gets a `HOLD` entry with reason `no_price`**, no price
  provenance, the retry stays armed, and no other command is sent for that
  slot. (`no_schedule` remains the reason for the different failure where
  prices are fine and the plan ran out; a restart that has not fetched yet also
  reports `no_schedule`, because an empty snapshot says nothing about whether
  the interval was published.)

  It is an **entry, not an absence**. The pre-solve step advanced the pack
  across this interval, so a schedule that omits it makes both callers rebuild
  `expected_soc_schedule` from the measured SOC and skip the interval
  altogether: at 10:07 with 40 % SOC and 3 kW of PV the DP starts 10:15 at
  42.3776 % while the published trajectory says 40 %, for the same quarter
  hour. `schedule` is the one source of truth for what runs, and every
  consumer walks it; the alternative — threading an advanced SOC, temperature
  and time anchor through `calculate_expected_soc_schedule`,
  `project_schedule_trajectory`, the cost projection and the deviation
  detector's anchor — is four more places that can disagree.

  A stand-in `HOLD` must not make the missing price look answered, so the
  paths that used to key off the absence test for it instead
  (`_is_no_price_fallback`): `execute_scheduled_mode` still applies
  `HOLD/no_price` and still arms the retry, the diagnostics still report
  `current_slot_entry: fallback`, and the adaptive horizon extension still
  rebuilds over it once the interval is published.
- **`execute_scheduled_mode` will not send a non-HOLD current-slot entry that
  carries no provenance.** After the above there should be none, so this is a
  guard that makes the claim falsifiable rather than a rule the code merely
  intends to follow: such an entry executes as `HOLD/unpriced_slot` and says so
  at WARNING.

This applies to every planning path - the daily optimization, every
`_recalculate_remaining_schedule` trigger (SOC deviation, PV shortfall,
depletion, price recovery) and the adaptive horizon extension - because a rule
applied in one of them is a rule the other four break. Each of those rebuilds
also asks the monitor to judge the snapshot it just fetched, so a rebuild
triggered by something other than a price problem is still able to notice one
and arm the retry.

Asking a source for the missing interval is a **fetch**, not a substitution,
and belongs in the price service and the bounded retry - which already
re-request whole days. `NordPoolPriceService.get_prices_for_date` remains
available for that.

The periodic adaptive pass evaluates the **last known** snapshot rather than
fetching: a price fetch is a blocking REST call on the shared AppDaemon worker
thread, and the retry is what pays that cost. Only when the snapshot is unusable
- or when it is fine but the current slot has no entry, i.e. the plan ran out -
does the adaptive pass act.

### Retained intervals

`get_prices()` merges each reply with the still-valid intervals already known.
A fresh value always wins for the same instant; retained values only fill
instants the reply does not contain, and only in the future. This exists because
the price service replaces its cache wholesale on any non-empty reply, so a
today-only response could shorten a horizon that already held tomorrow.

`price_retain_max_age_hours` measures the time since the last **non-empty**
reply, not the age of an individual interval: it is the backstop for a source
that goes permanently silent. It is not the thing that bounds how long an
interval survives — pruning to the future on every merge is, so an interval can
outlive at most its own instant.

An interval that a later reply **omits while spanning it** is retained. A
partial response and a genuine withdrawal are indistinguishable here, and
discarding a known price because one response came back short is the failure
this merge exists to prevent. A correction therefore only takes effect for
instants the reply actually contains.

`sensor.battery_optimizer` publishes the verdict under `price_horizon`:
coverage end, required end, the failure reason, the last successful horizon
end, pending-retry/attempt information, and - for the slot running right now -
`current_slot_priced` plus `current_slot_entry`
(`planned` / `retained` / `fallback`). The last two are scoped to the interval
they describe, so they report `null` rather than the previous slot's answer.
`current_slot_priced` is also `null` when it is genuinely *unknown* — a restart
that has never fetched prices cannot say whether the interval it woke up in was
published, and it reports `no_schedule`, not `no_price`.

`sensor.battery_optimizer` also publishes `rate_refinement`, the outcome of the
solve/replay/refine loop for the current plan: `branch` (`single_solve`,
`converged`, `conservative_fallback` or `degraded`), `passes`, `converged`,
`fallback`, `degraded` and `shortfall_kwh`. `degraded` is the one that matters
to a reader: the plan's actions were chosen for charge energy the pack will not
have at the temperatures it reaches, so nothing published credits that energy
but the plan is no longer economically optimal. It is logged at WARNING when it
happens; the attribute is how you see it afterwards.

## Re-optimization and execution

The app performs a full optimization after tomorrow's prices are expected and
re-evaluates periodically. Material SOC deviation, refreshed prices, or changed
forecasts cause the remaining horizon to be optimized again. The selected mode
is applied through `growatt_modbus/set_wit_mode`; no raw inverter-register writes
are performed by the optimizer.

Use dry-run mode (`device_id: ""`) first and compare the schedule, SOC trajectory,
and actual inverter behavior before enabling hardware control.

### There is no restart override

AppDaemon can restart in the middle of a CHARGE or DISCHARGE interval, and the
plan it was executing is gone. Nothing reads it back. **The DP's partial-slot
fraction is the continuity mechanism**: the first slot of the solve is the
remaining minutes of the interval the app woke up in, priced at that interval's
real price and started from the SOC that was just measured. Whatever the DP
decides there is what runs.

A forced continuation on top of that answer can only duplicate it or contradict
it, and it contradicted it. Two measured counterexamples, on a 10 kWh pack at
50 % with 59 minutes of the slot left:

- prices 2.00 / 0.05 / 0.05 with a previous CHARGE: the forced CHARGE imports
  3.93 kWh at 2.00 EUR/kWh — about 7.87 EUR — where the DP discharges and
  refills two slots later at 0.05.
- prices -0.50 / 1.00 / 1.00 with a previous DISCHARGE: the forced DISCHARGE
  spends the pack while the grid is *paying* to take energy, where the DP
  charges.

On an interval nobody published a price for it was worse than an economic
error. The continuation carried no provenance, so `execute_scheduled_mode`
refused it and applied `HOLD` — after the plan had already advanced the pack
across the refused action. On the reference fixture that is a 20-point SOC
error in the published trajectory, and every later discharge scheduled on
energy that will not exist.

**No action change may follow the final validation** either — the one
post-optimization rewrite is the cloud-safe hedge, which runs before it. See
`tests/test_restart_continuity.py`.
