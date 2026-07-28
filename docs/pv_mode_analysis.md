# PV-Aware Mode Analysis

## Problem Statement

The battery optimizer had a SELF_CONSUMPTION mode that was the **only** mode modeling PV production in the DP optimizer. This caused two problems:

1. **PV clipping**: The Growatt WIT inverter's `self_consumption` mode does not export excess PV when the battery is full and load is covered. Energy is wasted (MPPT clips).

2. **Model-reality gap**: The DP optimizer assumed only SELF_CONSUMPTION benefits from PV, but the inverter uses PV in **every** mode. HOLD and DISCHARGE looked artificially expensive during PV hours, biasing the optimizer toward SELF_CONSUMPTION.

## Principle

**No PV should ever be clipped.** The inverter should always route PV production through this priority chain:

1. PV → power house load
2. PV surplus → charge battery (when not full)
3. PV surplus → export to grid (when battery full or grid price is high)

## Inverter Mode Behavior (Growatt WIT)

| Mode | Inverter Command | Battery | PV Handling | Grid |
|------|-----------------|---------|-------------|------|
| **HOLD** | `hold` | No grid charge; PV surplus charges battery for free | PV → load → battery → export | Covers load deficit |
| **CHARGE** | `grid_charge` + `pv_priority` | Charges (PV surplus free, grid supplements) | PV → load → battery | Supplements charge |
| **CHARGE** | `grid_charge` + `ac_priority` | Charges (from grid) | PV → load | Charges battery |
| **DISCHARGE** | `discharge_to_load` | Discharges to cover net load | PV → load first (reduces discharge) | Covers remaining deficit |
| **DISCHARGE** | `max_export` / `discharge_to_grid` | Discharges at full rate | PV adds to export | Receives export |
| **SELF_CONSUMPTION** | `self_consumption` | Charges from PV surplus, discharges for load deficit | PV → load → battery → **CLIPS when full** | Covers deficit |

### Key Finding: SELF_CONSUMPTION Clips PV

When the battery reaches max SOC in `self_consumption` mode, the Growatt WIT does **not** export excess PV to the grid. The inverter's "Export Auto" label is misleading — excess PV is clipped by the MPPT tracker, wasting energy.

Observed in logs (March 28, 2026):
```
06:00  SELF_CONSUMPTION  pv~0.07kW   — pre-sunrise twilight, functionally useless PV
16:30  SELF_CONSUMPTION  pv~0.17kW   — battery at 100%, PV surplus clipped
```

## Previous DP Model (Before Fix)

| Mode | DP Cost Formula | PV Considered? |
|------|----------------|----------------|
| HOLD | `buy_price * load_kwh` | No — assumed grid pays ALL load |
| CHARGE | `buy_price * charge_input + buy_price * load_kwh` | No — assumed ALL charge from grid |
| DISCHARGE | `wear * min(load, rate) * hours` | No — assumed battery covers ALL load |
| EXPORT | `sell * (battery - load) - wear * battery` | No — PV not added to export |
| SELF_CONSUMPTION | `buy_price * grid_import + wear * discharge` | **Yes** — only mode with PV |

This made SELF_CONSUMPTION the only attractive option during PV hours, even though other modes also benefit from PV in reality.

## New DP Model (After Fix)

PV is modeled in **all** modes. SELF_CONSUMPTION is removed from the optimizer.

| Mode | DP Cost Formula | PV Effect |
|------|----------------|-----------|
| **HOLD** | `buy_price * max(0, load-pv)`, no charge cost | PV covers load, surplus charges battery for free (up to charge_rate), excess exports |
| **CHARGE** | `buy_price * max(0, charge_input - pv_surplus) + buy_price * max(0, load-pv)` | Only evaluated when PV surplus < charge rate (grid needed to supplement). When PV fully covers charge rate, HOLD handles it instead. |
| **DISCHARGE** | `wear * min(max(0, load-pv), rate)` | PV reduces effective load, less battery drain |
| **EXPORT** | `sell * max(0, battery+pv-load) - wear * battery` | PV adds to exported power |

### What Replaces SELF_CONSUMPTION?

The DP naturally decomposes SELF_CONSUMPTION into existing modes:

| SELF_CONSUMPTION Scenario | Now Handled By |
|--------------------------|----------------|
| PV > load, battery not full | **HOLD** — PV surplus charges battery for free (no grid) |
| Load > PV, battery has charge | **DISCHARGE** — battery covers deficit after PV offset |
| PV > load, battery full | **HOLD** — PV covers load, surplus exports to grid |
| PV > load, high grid price | **EXPORT** — battery + PV → grid for maximum revenue |

## Inverter Command Mapping

| DP Decision | Inverter Command | Parameters |
|------------|-----------------|------------|
| HOLD (night) | `hold` | Battery idle, grid covers load |
| HOLD (PV) | `hold` | PV covers load + charges battery for free |
| CHARGE (PV < charge rate) | `grid_charge` | `ac_charge_mode=pv_priority` — grid supplements PV |
| CHARGE (no PV) | `grid_charge` | `ac_charge_mode=ac_priority` — full grid charge |
| DISCHARGE (self-consumption) | `discharge_to_load` | `export_rate=0` |
| DISCHARGE (export) | `max_export` or `discharge_to_grid` | `export_rate=100` |

## Verified Behavior

### HOLD Mode PV Charging (Confirmed 2026-03-28)

**Verified**: The Growatt WIT in `hold` mode charges the battery from PV surplus. The inverter does NOT pull from grid in HOLD — only PV surplus (after covering load) goes to the battery. This makes HOLD the ideal mode during PV hours: free charging, no grid cost, battery stores PV surplus.

### HOLD Mode PV Export

**TODO**: Verify whether excess PV exports to grid when the battery is full in HOLD mode. If HOLD clips PV when battery is full, an alternative would be needed — possibly `discharge_to_load` at minimal power.

### Discharge-to-Load PV Charging (Confirmed 2026-03-28)

**Verified**: The Growatt WIT in `discharge_to_load` mode charges the battery from PV surplus. The inverter uses PV as the first priority regardless of mode:

1. PV → covers house load
2. PV surplus → charges battery
3. Remaining PV surplus → exports to grid

**Test Results** (via Modbus register 30409=-100, export limit=0%):

| Metric | Value |
|--------|-------|
| Mode | discharge_to_load (30409=-100) |
| Solar Total | 1,423 W |
| House Load | 424 W |
| Battery | 976 W (CHARGING) |
| Grid Export | 447 W |
| Battery SOC | 64% |

This means `discharge_to_load` is functionally identical to `hold` when PV is producing (both charge from PV surplus), but provides **cloud resilience**: when PV drops to 0, battery covers load instead of grid.

## Cloud-Safe Mode Selection

### Problem

During PV hours, the optimizer plans HOLD because PV covers load and surplus charges the battery for free. But if clouds suddenly kill PV production, HOLD means the grid covers all load — which may be more expensive than using battery energy.

### Solution: Cloud-Safe HOLD → DISCHARGE Conversion

Since `discharge_to_load` behaves identically to `hold` when PV is producing (confirmed above), the optimizer post-processes the DP schedule:

**For each HOLD slot where PV > 0:**
- If `buy_price (price + grid_fee) > battery_wear_cost`: convert to DISCHARGE(to load, export_rate=0)
- If `buy_price <= battery_wear_cost`: keep HOLD (grid backup is cheaper)

This gives free cloud insurance — identical behavior when PV is available, but battery covers load instead of grid when clouds hit.

### Reactive PV Re-evaluation

During adaptive re-evaluation (every 15 min), the **just-completed** slot is
compared to the forecast snapshot taken for that slot:

- PV power is sampled every `pv_sample_seconds` (60s default). A slot's measured
  value is the **mean of its samples** — i.e. its energy — not a single
  instantaneous reading taken at the boundary. A boundary read cannot represent
  a slot average during morning/evening ramps or passing clouds; the previous
  implementation produced 43 recalculations in 33 hours of production logs.
- A slot counts as a shortfall when `measured/forecast < pv_reactive_threshold`
  (0.5), the forecast exceeds `pv_reactive_min_forecast_w` (200W) and at least
  `pv_reactive_min_samples` (3) samples were collected.
- A full recalculation requires `pv_reactive_consecutive_slots` (2) consecutive
  shortfall slots. Recalculation starts from actual SOC, so the DP adjusts the
  remaining schedule.
- An unavailable PV sensor yields *no* sample rather than a 0 W reading, so a
  dead sensor can no longer masquerade as "0% of forecast".

### Sliding PV Forecast Bias

The shortfall correction used to be applied to the current slot only
(`PvForecastService.refresh_for_shortfall`), so every rerun re-planned the rest
of the horizon on the same optimistic provider forecast.

`pv_bias_tracker.PvBiasTracker` now maintains a **sliding bias factor**: the
median `measured/forecast` ratio over `pv_bias_window_minutes` (120), clamped to
`[pv_bias_min_factor, pv_bias_max_factor]` (0.2..1.5) and relaxing back to 1.0
over `pv_bias_decay_slots` (8) once observations go stale. `_predict_pv_kw`
multiplies **the current and every remaining slot** by that factor, so the DP,
the expected-SOC trajectory, the projected costs and the min-charge estimate all
see the corrected forecast. `_predict_pv_kw_raw` returns the un-corrected
provider value and is what feeds the ratio, preventing a bias-on-bias loop.

The factor is logged on every material change and exposed on
`sensor.battery_optimizer_schedule` as `pv_bias_factor` / `pv_bias_samples` /
`pv_bias_enabled`. Set `pv_bias_enabled: false` to disable it entirely.

### Cost Comparison

| Scenario | HOLD (grid fallback) | DISCHARGE (battery fallback) |
|----------|---------------------|------------------------------|
| PV available | PV covers load, surplus charges battery | Same — PV covers load, surplus charges battery |
| Cloud (PV=0) | Grid pays: `(price + grid_fee) * load` | Battery pays: `wear_cost * load` |
| Typical day | ~0.06-0.07 EUR/kWh | ~0.017 EUR/kWh |

Battery backup is typically 3-4x cheaper than grid during most hours, making DISCHARGE the better default for PV hours.

## Configuration

The `enable_self_consumption` and `self_consumption_min_pv_kw` config fields have been removed. PV-aware cost modeling is always active — the PV predictor is unconditionally passed to the DP optimizer.

Cloud-safe mode selection is always active. Configurable thresholds:
- `pv_reactive_threshold`: Fraction below which measured PV triggers recalc (default: 0.5 = 50%)
- `pv_reactive_min_forecast_w`: Minimum forecast (W) to check PV shortfall (default: 200W)
- `pv_reactive_consecutive_slots`: Consecutive shortfall slots before a full recalc (default: 2)
- `pv_reactive_min_samples`: Minimum samples in a slot before its mean is trusted (default: 3)
- `pv_sample_seconds`: PV power sampling interval (default: 60, clamped to one slot)
- `pv_bias_enabled`: Master switch for the sliding forecast bias (default: true)
- `pv_bias_window_minutes`: Ratio window used for the median (default: 120)
- `pv_bias_min_slots`: Minimum in-window observations before the factor leaves 1.0 (default: 2)
- `pv_bias_min_factor` / `pv_bias_max_factor`: Clamp for the factor (default: 0.2 / 1.5)
- `pv_bias_decay_slots`: Slots without fresh data to relax back to 1.0 (default: 8)

## Files Changed

- `dp_optimizer.py` — PV modeled in HOLD (free PV charging + SOC change), CHARGE (PV reduces grid cost), DISCHARGE (PV reduces drain), EXPORT (PV adds to revenue); removed SELF_CONSUMPTION blocks
- `battery_optimizer.py` — HOLD SOC projection includes PV charging; DISCHARGE uses PV offset and PV surplus charging; solar override uses HOLD; cloud-safe HOLD→DISCHARGE conversion; reactive PV recalculation; removed `enable_self_consumption` conditionals
- `config.py` — removed `enable_self_consumption` and `self_consumption_min_pv_kw` fields; added `pv_reactive_threshold` and `pv_reactive_min_forecast_w`
- `direct_control.py` — `BatteryMode.SELF_CONSUMPTION` mapping initially retained, later removed
- `models.py` — `BatteryMode.SELF_CONSUMPTION` enum value initially retained, later removed

**Update (2026-07):** the `BatteryMode.SELF_CONSUMPTION` enum member and all
remaining consumer branches (direct_control, formatters, cost projection, SOC
deviation, slot outcome matching) were fully removed in the dead-code cleanup.
Stale `"SELF_CONSUMPTION"` strings in a schedule persisted by a pre-removal
build are safely ignored on restore (name lookup raises `KeyError`, which is
caught and the entry skipped).
