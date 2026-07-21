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

## End-of-horizon value

Without an end condition, a finite-horizon optimizer treats stored energy after
the last known price as worthless and tends to empty the battery. The terminal
energy value corrects that horizon effect:

- A numeric `terminal_energy_value_eur_kwh` values each usable DC kWh remaining
  at the horizon by that amount.
- `terminal_energy_value_eur_kwh: auto` derives a conservative value from the
  median horizon import price, battery-to-AC conversion, and wear cost.
- `0` disables terminal value and reproduces the legacy end-of-horizon behavior.

The default application configuration uses `auto`. This is a value, not a hard
terminal-SOC constraint: sufficiently valuable load or export may still justify
ending near `min_soc`.

## PV forecast and live control

Forecast PV participates directly in the DP. PV first serves predicted load;
surplus can charge the battery within its charge/headroom limits, and remaining
surplus can earn net export revenue. The generated schedule is not rewritten
after optimization merely because forecast PV is present; uncertainty should be
handled by refreshing the forecast and re-running the optimizer.

During execution, live PV above `pv_threshold_w` can pause a scheduled grid
charge so solar can charge instead. This real-time safety/operational override
does not rewrite future schedule entries.

When measured PV falls materially below the current-slot forecast, the app
forces a rate-limited forecast refresh, caps that slot at the observed output,
and replans. This prevents the normal forecast cache from repeatedly selecting
HOLD from a stale optimistic value.

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
