# Household Consumption Data: Collection, Storage & Usage in DP Optimizer

> Audit document for the Battery Optimizer system's load profiling subsystem.

---

## 1. Overview

The Battery Optimizer uses a statistical load profile to predict household electricity consumption for each 15-minute slot of the day. These predictions directly influence the dynamic programming (DP) optimizer's decisions about when to charge, hold, or discharge the battery — making load data quality a critical factor in optimization accuracy.

**Key files:**

| File | Purpose |
|------|---------|
| `appdaemon/apps/battery_optimizer.py` | Orchestrator: reads sensor, records observations, invokes optimizer |
| `appdaemon/apps/battery_optimizer_lib/load_profile.py` | Statistical engine: stores samples, computes quantile forecasts |
| `appdaemon/apps/battery_optimizer_lib/models.py` | `LoadProfileStats` dataclass |
| `appdaemon/apps/battery_optimizer_lib/dp_optimizer.py` | DP algorithm: consumes load predictions |
| `appdaemon/apps/battery_optimizer_lib/config.py` | Configuration defaults and loader |

---

## 2. Data Source

### 2.1 Sensor

| Property | Value |
|----------|-------|
| **HA Entity** | `sensor.growatt_house_consumption` (configurable via `load_power_sensor`) |
| **Unit** | Watts (W) |
| **Source** | Growatt inverter's built-in CT clamp measuring power delivered to household |
| **Reading method** | `SensorReader.get_float()` in `ha_helpers.py` |

### 2.2 Sensor Reading Logic (`battery_optimizer.py:1848-1861`)

```python
def _get_load_power(self) -> Optional[float]:
    load_w = self._sensors.get_float(self.config.load_power_sensor)
    if load_w is None:
        return None
    if load_w <= 0:
        # Use last known value or floor when sensor reports zero
        if self._last_nonzero_load_w is not None:
            return max(self._last_nonzero_load_w, self.config.load_zero_floor_w)
        return self.config.load_zero_floor_w  # default: 450 W
    self._last_nonzero_load_w = load_w
    return load_w
```

**Validation & edge cases:**
- `is_state_valid()` rejects `"unknown"`, `"unavailable"`, and `None` states
- Zero/negative readings are replaced with `max(last_known, 450W)` floor to prevent unrealistic zeros during inverter communication gaps
- If sensor is not configured (`load_power_sensor: ""`), load profiling is entirely disabled

---

## 3. Data Collection

### 3.1 Recording Schedule

Observations are recorded on a fixed interval via AppDaemon's `run_every` scheduler:

```python
# battery_optimizer.py:246-248
self.run_every(
    self.record_load_observation,
    self._next_interval_time(self.config.load_observation_minutes),
    self.config.load_observation_minutes * 60   # default: 15 min = 900 sec
)
```

**Recording frequency:** Every 15 minutes (configurable via `load_observation_minutes`).

### 3.2 Recording Process (`battery_optimizer.py:1739-1760`)

Each observation cycle:

1. Read current load from sensor via `_get_load_power()`
2. Align timestamp to slot boundary (e.g., 14:07 → 14:00)
3. Record actual load for prediction accuracy tracking
4. Store sample in `LoadProfile.record(dt, load_w)`
5. Record prediction for the *next* slot (for future accuracy comparison)
6. Persist to disk and update HA status sensors

### 3.3 Storage in LoadProfile (`load_profile.py:60-71`)

```python
def record(self, dt: datetime.datetime, load_w: float):
    if load_w <= 0:
        return
    slot = str(self._slot_index(dt))       # "0".."95"
    samples = self.stats.samples_by_slot.get(slot, [])
    samples.append(float(load_w))
    if len(samples) > self.max_samples:
        samples = samples[-self.max_samples:]   # keep most recent 60
    self.stats.samples_by_slot[slot] = samples
    self.stats.observation_count += 1
    self.stats.last_observation = dt.isoformat()
```

**Slot indexing:** `slot = (hour * 60 + minute) // 15`, yielding indices 0..95 for 96 daily slots.

---

## 4. Data Structure & Volume

### 4.1 In-Memory Model (`models.py:97-108`)

```python
@dataclass
class LoadProfileStats:
    samples_by_slot: Dict[str, List[float]]   # slot index → list of W samples
    observation_count: int = 0                 # total observations ever recorded
    last_observation: Optional[str] = None     # ISO timestamp
```

### 4.2 Data Volume Characteristics

| Metric | Value |
|--------|-------|
| Slots per day | 96 (1440 min / 15 min) |
| Max samples per slot | 60 (configurable: `load_profile_max_samples`) |
| Max total samples in memory | 96 × 60 = **5,760 float values** |
| Sample retention | Rolling window — oldest samples dropped when limit exceeded |
| Days of history per slot | 60 observations ÷ (96 obs/day × 1 slot) = **60 days** per slot |
| Typical JSON file size | ~50-100 KB |

### 4.3 Persistence

| Method | Location | Notes |
|--------|----------|-------|
| **Primary: File** | `/config/load_profile.json` (configurable) | No size limit; preferred |
| **Fallback: HA entity** | `input_text.battery_load_profile` | 255-char limit; disabled by default |

**JSON format:**
```json
{
  "version": 1,
  "slot_minutes": 15,
  "stats": {
    "samples_by_slot": {
      "0": [450.0, 478.2, 441.5, ...],
      "1": [460.0, 455.3, ...],
      ...
      "95": [520.0, 510.8, ...]
    },
    "observation_count": 4320,
    "last_observation": "2026-03-25T12:30:00+02:00"
  }
}
```

### 4.4 Slot Resolution Migration (`load_profile.py:99-145`)

When `slot_minutes` changes (e.g., 30 → 15), the system automatically migrates:
- Each old bucket splits into N sub-buckets (e.g., 1 × 30min → 2 × 15min)
- Sub-buckets receive the old bucket's mean as a single sample (low confidence)
- New real observations quickly replace these bootstrapped values

---

## 5. Prediction Algorithm

### 5.1 Quantile-Based Forecast (`load_profile.py:73-88`)

```python
def predict_kw(self, dt, quantile=0.75, correction_factor=1.0) -> float:
    slot = str(self._slot_index(dt))
    samples = self.stats.samples_by_slot.get(slot, [])
    if not samples:
        return max(0.0, self.default_load_w * correction_factor) / 1000.0
    q_value = _quantile(samples, quantile)
    confidence = min(1.0, len(samples) / self.min_samples)
    blended = (self.default_load_w * (1 - confidence)) + (q_value * confidence)
    return max(0.0, blended * correction_factor) / 1000.0
```

**Step-by-step:**

1. **Lookup slot**: Map datetime to slot index (0..95)
2. **No data fallback**: If no samples for this slot, return `default_load_w` (500W) scaled by correction factor
3. **Quantile calculation**: Compute 75th percentile (P75) from sorted samples using linear interpolation
4. **Confidence blending**:
   - `confidence = min(1.0, num_samples / 6)`
   - `blended = default_load × (1 - confidence) + P75 × confidence`
   - With 1 sample: 83% default + 17% observed
   - With 6+ samples: 100% observed quantile
5. **Correction factor**: Applied from prediction accuracy tracker (see §5.2)
6. **Convert W → kW** and return

### 5.2 Prediction Accuracy Tracker

A companion system (`LoadPredictionTracker`) compares predictions against actual observations:

```
At each 15-min observation:
  1. Record actual load for the just-completed slot
  2. Generate prediction for the next slot
  3. At next observation: compare prediction vs. actual → compute error ratio
```

The `correction_factor` adjusts predictions for systematic bias (e.g., if predictions consistently overestimate by 10%, the factor would be ~0.9). Persisted to `/config/prediction_tracker.json`.

### 5.3 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_quantile` | 0.75 | Percentile for forecast (conservative — overestimates 75% of the time) |
| `base_consumption_w` | 500 | Fallback when no samples available |
| `load_profile_max_samples` | 60 | Rolling window size per slot |
| `load_profile_min_samples` | 6 | Threshold for full confidence |
| `load_zero_floor_w` | 450 | Minimum load when sensor reads ≤ 0 |
| `load_observation_minutes` | 15 | Recording frequency |

---

## 6. Usage in DP Optimizer

### 6.1 Pre-computation (`dp_optimizer.py:218`)

Before the DP algorithm runs, load is predicted for every slot in the optimization horizon:

```python
load_kw = [self._predict_load_kw(p.time) for p in slots_sorted_by_time]
```

This produces a list of kW values (one per slot), passed into the DP engine.

### 6.2 Impact on Each Battery Mode

The load prediction affects the DP optimizer's cost/value calculation for each action at every time step:

#### HOLD (do nothing) — `dp_optimizer.py:448-457`

```python
hold_cost = buy_price * discharge_kwh
# where: discharge_kwh = min(load_kw[t], discharge_rate) * slot_hours * fraction
```

HOLD means the household load must be fully served by grid import. The cost of holding equals the grid electricity price × the load energy for this slot. **Higher predicted load → higher HOLD cost → stronger incentive to discharge instead.**

#### DISCHARGE (self-consumption) — `dp_optimizer.py:491-522`

```python
discharge_kwh = min(load_kw[t], discharge_rate) * slot_hours * fraction
```

Discharge is **capped by predicted load**: the battery can only displace grid import up to what the household actually consumes. If predicted load is 2 kW but discharge rate is 5.9 kW, only 2 kW is discharged.

The value of discharge = avoided grid import cost = `buy_price × discharge_kwh`.

**Impact:** Low predicted load → less discharge capacity per slot → discharge spread over more slots.

#### DISCHARGE with EXPORT — `dp_optimizer.py:528-560`

```python
load_kwh = load_kw[t] * slot_hours * fraction
exported_kwh = max(0.0, export_discharge_kwh - load_kwh)
```

When grid export is profitable, the battery discharges at full rate. Load is subtracted from total discharge to determine exportable surplus:

- Load 2 kW, discharge rate 5.9 kW → export 3.9 kW
- Load 5 kW, discharge rate 5.9 kW → export 0.9 kW

**Impact:** Higher predicted load → less exportable surplus → lower export revenue per slot.

#### CHARGE — `dp_optimizer.py:465-489`

```python
next_val = val - (buy_price * actual_charge_cost) - (buy_price * discharge_kwh)
```

During charging, the household load still must be served by grid import (same as HOLD). The total cost = charging cost + load cost. **Higher predicted load → charging slots become more expensive.**

### 6.3 Summary of Load Influence

| Aspect | Higher Load Prediction | Lower Load Prediction |
|--------|----------------------|---------------------|
| HOLD cost | Higher (more grid import) | Lower |
| DISCHARGE value | Higher (more avoided import) | Lower |
| DISCHARGE capacity | Capped at load (self-consumption) | Less discharge per slot |
| EXPORT surplus | Less exportable | More exportable |
| CHARGE total cost | Higher (load + charge cost) | Lower |
| Overall effect | Favors discharge during high-load periods | Favors export/hold during low-load periods |

---

## 7. Monitoring & Visibility

### 7.1 Home Assistant Entities

| Entity | Purpose |
|--------|---------|
| `sensor.load_profile_observation_count` | Total observations recorded (lifetime) |
| `sensor.load_profile_last_observation` | Timestamp of most recent observation |
| `sensor.battery_optimizer` (attribute: `load_profile_observations`) | Same count, in optimizer sensor |
| `sensor.battery_optimizer` (attribute: `load_profile_stats`) | Hourly aggregates for dashboard charts |

### 7.2 Hourly Statistics Exposure (`battery_optimizer.py:2009-2057`)

For dashboard visualization, the system aggregates 15-min slot data into hourly buckets with:
- `avg`: Mean consumption (W)
- `min` / `max`: Range observed
- `p25` / `p75`: Interquartile range
- `samples`: Number of samples in that hour

---

## 8. Data Quality & Risks

### 8.1 Warm-up Period

The system needs sufficient observations before predictions become reliable:

| Milestone | Observations | Duration |
|-----------|-------------|----------|
| First prediction possible | 1 per slot | ~1 day |
| Full confidence (per slot) | 6 per slot | ~6 days |
| Full rolling window | 60 per slot | ~60 days |

During warm-up, the confidence blending mechanism gradually transitions from the `default_load_w` fallback (500W) to empirical quantile data.

### 8.2 Identified Risks

| Risk | Mitigation | Residual Impact |
|------|-----------|-----------------|
| **Sensor unavailability** | Returns `None`, observation skipped | Missing data points, slightly less accurate profile |
| **Sensor reporting zero** | Floor at 450W or last known value | Prevents unrealistic zero-load slots; may overestimate during genuine low-use periods |
| **Seasonal variation** | Rolling window (60 samples) favors recent data | ~60-day lag for seasonal shifts; no explicit seasonality model |
| **Weekday/weekend patterns** | Not distinguished | Same profile used for all days of week |
| **Outlier observations** | P75 quantile is robust to outliers | Extreme values affect only if persistent (>25% of samples) |
| **Systematic prediction bias** | Correction factor from prediction tracker | Requires sufficient comparison history |

### 8.3 No Data Scenario

When the load profile has no data for a slot:
- Prediction falls back to `base_consumption_w / 1000.0` (default: 0.5 kW)
- This is a conservative assumption — 500W base load
- DP optimizer will make suboptimal but safe scheduling decisions

---

## 9. Complete Data Flow Diagram

```
 Growatt Inverter (CT Clamp)
        │
        ▼
 [sensor.growatt_house_consumption]     ← HA sensor, real-time Watts
        │
        ▼ (every 15 min)
 _get_load_power()                  ← Validation, zero-floor, last-known fallback
        │
        ├──► prediction_tracker.record_actual(now, kW)
        │
        ▼
 LoadProfile.record(dt, load_w)     ← Append to samples_by_slot[slot_index]
        │                              Keep last 60 samples per slot
        │
        ├──► _save_load_profile()   → /config/load_profile.json
        │
        ├──► _update_load_profile_sensors()
        │       → sensor.load_profile_observation_count
        │       → sensor.load_profile_last_observation
        │
        ▼ (on optimization run)
 LoadProfile.predict_kw(dt)         ← P75 quantile + confidence blending
        │                              + correction factor
        ▼
 DPOptimizer.optimize()
        │
        ├── load_kw[t] for each slot
        │
        ├── HOLD cost    = buy_price × min(load, discharge_rate) × slot_hours
        ├── DISCHARGE    = min(load, discharge_rate) × slot_hours  [capped by load]
        ├── EXPORT       = max(0, discharge - load) × slot_hours   [load subtracted]
        └── CHARGE cost  = charge_cost + buy_price × load × slot_hours
                │
                ▼
        Optimal Schedule (CHARGE / HOLD / DISCHARGE per slot)
                │
                ▼
        growatt_modbus/set_wit_mode (per-slot execution)
```

---

*Document generated: 2026-03-25*
