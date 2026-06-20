# DP Optimization Parameters — Deep Analysis

This document lists every parameter that influences the dynamic programming (DP)
battery scheduling optimizer, its default value, your configured value (from
`apps.yaml`), and how it enters the cost/revenue calculations.

---

## 1. Price Inputs

| Parameter | What It Represents | Source |
|---|---|---|
| `price` (per slot) | Nord Pool spot price (EUR/kWh) | Nord Pool service call; API returns EUR/MWh, converted to EUR/kWh by dividing by 1000 |

**Important**: The spot price from Nord Pool is the *raw wholesale price* — no VAT,
no distribution, no transmission fee included. All additional costs are modeled
separately via `grid_fee` and `grid_export_fee`.

---

## 2. Pricing Parameters (Fees & Commissions)

These are the core parameters that model your grid contract economics.

| Parameter | `apps.yaml` key | Default | **Your Value** | Unit | Used In |
|---|---|---|---|---|---|
| Grid buy fee | `grid_fee_eur_kwh` | 0.052 | **0.052** | EUR/kWh | Added to spot price for every kWh purchased from grid |
| Grid export fee | `grid_export_fee_eur_kwh` | 0.02 | **0.02** | EUR/kWh | Subtracted from spot price when selling to grid |
| Export rate multiplier | `export_rate_multiplier` | 1.0 | **1.0** | ratio | Multiplier on spot price before export fee deduction |
| Battery wear cost | `battery_wear_cost_eur_kwh` | 0.0 | **0.017** | EUR/kWh | Per-kWh cost for every discharge cycle (degradation) |

### How prices are computed in the DP

```
buy_price  = nordpool_spot + grid_fee           # cost per kWh to BUY from grid
sell_price = nordpool_spot * export_rate_multiplier - grid_export_fee   # revenue per kWh when SELLING to grid
```

**Grid fee breakdown** (from apps.yaml comment):
- Trading margin: 0.01238 EUR/kWh
- Distribution fee: 0.03962 EUR/kWh
- Total: **0.052 EUR/kWh**

---

## 3. What The DP Does NOT Model (Potential Gaps)

Review these against your actual electricity contract to see if anything is missing:

| Cost Component | Modeled? | Notes |
|---|---|---|
| **Nord Pool spot price** | Yes | Raw wholesale price in EUR/kWh |
| **Trading margin** | Yes | Included in `grid_fee` (0.01238) |
| **Distribution/network fee** | Yes | Included in `grid_fee` (0.03962) |
| **VAT (PVN)** | **NO** | Not applied to spot price, grid_fee, or sell price. If your contract charges VAT on electricity, the DP underestimates buy costs and overestimates sell revenue |
| **Mandatory procurement component (OIK)** | **NO** | If your contract includes OIK (obligātā iepirkuma komponente), it's not modeled |
| **Transmission fee** | **Unclear** | May or may not be included in the 0.03962 distribution portion of grid_fee |
| **Electricity tax (akcīzes nodoklis)** | **NO** | Not modeled separately |
| **Capacity fee / fixed monthly** | N/A | Fixed costs don't affect optimization decisions (correct to exclude) |
| **Net metering balance** | **NO** | If you have a net metering agreement with accumulation, the sell-back economics may differ from `sell_price` |
| **Time-of-use network tariffs** | **NO** | If your distribution tariff varies by time of day (e.g., day/night rates), this is not modeled — `grid_fee` is constant |
| **Feed-in cap / export limit** | **NO** | No limit modeled on how much you can export per slot |
| **Negative price floor** | **NO** | If spot goes negative, `sell_price` can go negative too (you'd pay to export). The DP handles this correctly — it won't choose to export at a loss because `sell_price > 0` is checked |

---

## 4. Battery Physical Parameters

| Parameter | `apps.yaml` key | Default | **Your Value** | Unit | Role in DP |
|---|---|---|---|---|---|
| Battery capacity | `battery_capacity_kwh` | 14.3 | **14.3** | kWh | Defines SOC energy range |
| Charge rate | `charge_rate_kw` | 4.5 | **4.5** | kW | Max grid charge power (per-slot, may be reduced by learning engine for SOC/temp) |
| Discharge rate | `discharge_rate_kw` | 4.5 | **5.9** | kW | Max self-consumption discharge power |
| Export discharge rate | `export_discharge_rate_kw` | 0.0 (→ discharge_rate) | **0.0 → 5.9** | kW | Discharge power during grid export (falls back to discharge_rate) |
| Efficiency | `efficiency` | 0.85 | **0.95** | ratio | Round-trip charge efficiency: 1 kWh from grid → `efficiency` kWh into battery |
| Min SOC | `min_soc` | 10 | **10** | % | Lower bound for DP state space |
| Max SOC | `max_soc` | 100 | **100** | % | Upper bound for DP state space |
| SOC step | `soc_step_percent` | 1.0 | **0.25** | % | DP granularity (0.25% of 14.3 kWh = 0.036 kWh steps) |

---

## 5. Schedule Resolution

| Parameter | `apps.yaml` key | Default | **Your Value** | Unit |
|---|---|---|---|---|
| Slot duration | `slot_minutes` | 15 | **15** | minutes |
| Derived slot hours | (computed) | — | **0.25** | hours |
| Slots per day | (computed) | — | **96** | count |

---

## 6. Load Prediction Parameters

| Parameter | `apps.yaml` key | Default | **Your Value** | Role |
|---|---|---|---|---|
| Load quantile | `load_quantile` | 0.75 | **0.75** | Statistical percentile for load forecast (conservative — assumes higher-than-median load) |
| Base consumption | `base_consumption_w` | 500 | **500** | Fallback W when no load profile available |
| Load zero floor | `load_zero_floor_w` | 450 | **450** | Minimum load reading (clamps sensor zeros) |
| Load sensor | `load_power_sensor` | — | **sensor.growatt_house_consumption** | Source for load observations |

**In the DP**: `load_kw[t]` = predicted household load for each slot. Used to compute:
- `net_load_kw = max(0, load_kw - pv_kw)` — load that battery/grid must cover
- `discharge_kwh = min(net_load_kw, discharge_rate) * slot_hours * fraction` — self-consumption discharge

---

## 7. PV Prediction Parameters

| Parameter | `apps.yaml` key | Default | **Your Value** | Role |
|---|---|---|---|---|
| PV quantile | `pv_quantile` | 0.5 | **0.5** | Percentile for PV profile forecast |
| PV forecast source | `solcast_today_entity` | — | (configured) | Solcast forecast sensor |

**In the DP**: `pv_kw[t]` = predicted PV production for each slot. Used to compute:
- `net_load_kw = max(0, load - pv)` — residual load after PV
- `pv_surplus_kw = max(0, pv - load)` — excess PV that can charge battery or export
- HOLD action: PV surplus charges battery for free (up to charge rate), excess PV beyond charge rate is exported at `sell_price`
- CHARGE action: PV offsets grid charge cost (`pv_free_charge_kwh`)

---

## 8. DP Action Value Formulas

For each time slot `t`, the DP evaluates four possible actions:

### HOLD (battery idle, grid covers net load)
```
cost    = buy_price × net_load_kwh
revenue = sell_price × (excess_pv_exported + pv_that_didnt_fit_in_battery)
value   = -cost + revenue
```
PV surplus charges battery for free (up to charge rate). If battery is full,
unused PV energy is exported.

### CHARGE (grid charges battery)
```
grid_charge_cost = max(0, total_charge_energy - pv_free_charge) × buy_price
load_cost        = buy_price × net_load_kwh
value            = -(grid_charge_cost + load_cost)
```
When PV surplus already covers the full charge rate, CHARGE is suppressed (HOLD handles it).

### DISCHARGE — self-consumption (battery covers household load)
```
discharge_kwh = min(net_load_kw, discharge_rate) × slot_hours × fraction
value         = -(battery_wear_cost × discharge_kwh)
```
The "revenue" is implicit: by discharging, you avoid paying `buy_price × discharge_kwh`
for grid import. The DP captures this because the HOLD alternative would pay that cost.

### DISCHARGE — grid export (sell battery energy)
```
export_kwh = max(0, discharge_energy + pv_energy - load_energy)
value      = sell_price × export_kwh - battery_wear_cost × discharge_kwh
```
Only allowed when `sell_price > 0` and there's exportable surplus.

---

## 9. Tie-Breaking Logic

When two actions produce nearly equal value (within 1e-6 EUR), a secondary
tie-breaker prefers:
- **Charging at lower prices** (bias: `-price × 1e-5`)
- **Charging earlier** (bias: `+slot_index × 1e-7`)

This ensures the DP prefers cheap, early charging when values are otherwise equal.

---

## 10. Temperature & SOC-Aware Charge Rate

The charge rate per slot is not constant — it's projected forward accounting for:
- **Battery temperature**: Higher temp → faster charging (learning engine data)
- **SOC level**: Higher SOC → BMS reduces charge rate (learning engine data)
- **SOC projection**: Assumes continuous charging to conservatively project rate decline

If no learning data is available, the static `charge_rate_kw` (4.5 kW) is used.

---

## 11. Partial First Slot

If the optimizer runs mid-slot (e.g., 7 minutes into a 15-minute slot), the first
slot's energy calculations are scaled by `fraction = (slot_minutes - minutes_into_slot) / slot_minutes`.

---

## 12. Summary: Your Configured Cost Model

```
BUYING from grid:    nordpool_spot + 0.052 = effective buy price (EUR/kWh)
SELLING to grid:     nordpool_spot × 1.0 - 0.02 = effective sell price (EUR/kWh)
CYCLING battery:     0.017 EUR/kWh wear cost per discharge
CHARGING efficiency: 95% (1 kWh from grid → 0.95 kWh stored)
```

### Example at spot price 0.10 EUR/kWh:
- Buy:  0.10 + 0.052 = **0.152 EUR/kWh**
- Sell:  0.10 - 0.02 = **0.080 EUR/kWh**
- Spread: 0.152 - 0.080 = 0.072 EUR/kWh (buy-sell gap)
- Min profitable discharge cycle: buy_spread must exceed `wear_cost / efficiency` = 0.017/0.95 ≈ 0.018 EUR/kWh

### Items to validate against your contract:
1. **Is 0.052 EUR/kWh the complete buy-side fee?** Does it include all of: trading margin, distribution, transmission, OIK, electricity tax?
2. **Is VAT excluded intentionally?** If you pay 21% PVN on electricity, the true buy cost is `(spot + 0.052) × 1.21` and sell revenue is `(spot - 0.02)` (or `(spot - 0.02) × 1.21` if VAT applies to sell-back).
3. **Is 0.02 EUR/kWh the correct export deduction?** Check your net metering / sell-back terms.
4. **Is the distribution fee time-invariant?** Some Latvian DSO tariffs have day/night differentials.

---

## 13. Battery & Inverter Efficiency Analysis (Growatt HOPE 14.3L-A1 + WIT Inverter)

### Inverter Specs (from user manual)

| Parameter | Value |
|---|---|
| Max efficiency | 97.60% |
| European weighted efficiency | 97.00% |
| MPPT efficiency | 99.90% |

### Battery Sensor Measurement Point

The Growatt Modbus integration reads battery energy from **input registers**:
- `battery_charge_today`: registers **1056-1057** (or VPP 31206-31207)
- `battery_discharge_today`: registers **1052-1053** (or VPP 31202-31203)

These are in the hardware register range and measure energy at the **DC battery bus**
(between inverter and battery). They do NOT include inverter AC-DC conversion losses.

### How The Learning Engine Uses These Sensors

The learning engine (`learning_engine.py:134-142`) computes charge rate as:
```python
energy_added = energy_to_battery_kwh   # from DC-side sensor (regs 1056-1057)
charge_rate = energy_added / (duration_minutes / 60)   # → DC-side kW
```

**The learned charge rate is a DC-side measurement** — it represents kW flowing
into the battery at the DC bus, AFTER the inverter has already taken its ~3% cut.

### How The DP Uses Efficiency

In `dp_optimizer.py:428-429`:
```python
charge_energy_kwh = slot_charge_rate × efficiency × slot_hours    # → stored in battery
charge_cost_kwh   = slot_charge_rate × slot_hours                 # → "drawn from grid"
```

The DP treats `slot_charge_rate` as **grid-side (AC) power** and applies
`efficiency` to get battery-stored energy. But the learned rate is actually
**DC-side**, creating a mismatch:

| Quantity | DP assumes | Reality (DC-side rate R) |
|---|---|---|
| Grid energy drawn | R × h | R / 0.97 × h (inverter loss) |
| Energy stored in battery | R × 0.95 × h | R × h (already DC-side) |
| Grid cost | buy_price × R × h | buy_price × R / 0.97 × h |

**Net effect**: The DP **underestimates grid cost by ~3%** (missing inverter
charging loss) and **underestimates stored energy by ~5%** (applying efficiency
to an already-lossy measurement). On discharge, the DP assumes 100% DC→AC
conversion, **overestimating discharge value by ~3%** (missing inverter output loss).

The **cost per kWh stored** remains correct (`buy_price / efficiency`), so the
DP makes correct *relative* decisions between slots. But the absolute energy
amounts per slot are ~5% too conservative, which may cause the DP to schedule
1-2 extra charge slots than necessary.

### What `efficiency: 0.95` Actually Represents

Given that sensors are DC-side:

| Efficiency layer | One-way | Round-trip |
|---|---|---|
| Inverter AC↔DC (European weighted) | 97.0% | 94.1% (0.97²) |
| Battery pack DC (LiFePO4 HOPE) | ~97.5% | ~95.0% |
| **Full system AC→battery→AC** | — | **~89.4%** (0.97 × 0.975 × 0.975 × 0.97) |

The `efficiency: 0.95` in your config matches the **DC-bus round-trip** (charge
sensor / discharge sensor ratio). This is a reasonable choice since the DP
operates in terms of DC-bus energy. The ~3% inverter loss each way is not
separately modeled but has minor impact on optimization quality since it
affects all actions equally.

### Cycle Life: "6000@93%DOD"

- **6,000 cycles**: Battery retains >=80% capacity after 6,000 full cycles
- **93% DOD**: Each cycle uses 93% of 14.3 kWh = **13.3 kWh** (matches "usable energy" spec)
- At 1 cycle/day: ~16.4 years lifetime (beyond 10-year warranty)

### Wear Cost Calculation Discrepancy

```
apps.yaml uses:   1524.03 / (8000 × 14.3 × 0.80 DOD) = 0.017 EUR/kWh
Datasheet says:   1524.03 / (6000 × 14.3 × 0.93 DOD) = 0.019 EUR/kWh
```

The 8,000 cycle figure may come from marketing materials for shallower DOD.
Using the datasheet's conservative 6,000@93%DOD gives **0.019 EUR/kWh** — a
minor but directionally more conservative value.
