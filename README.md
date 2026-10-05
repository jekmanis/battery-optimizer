# Battery Optimizer for Growatt WIT Inverter

A Home Assistant add‑on that uses **Nord Pool** day‑ahead electricity prices (and optionally **Solcast** PV forecasts) to compute and execute a low‑cost battery **charge / hold / discharge** schedule for a Growatt **WIT** hybrid inverter.

It plans with dynamic programming over SOC, learns your house load and real charge rates over time, tracks the stored‑energy cost, and drives the inverter in real time through the Growatt integration's `set_wit_mode` service.

> ⚠️ This software actively controls battery hardware (grid charging, export, discharge). Use at your own risk and verify behaviour on your own system. See [Disclaimer](#disclaimer).

---

## Features

- **DP price optimizer** — dynamic programming over discretised SOC searches charge/hold/discharge sequences for every 15‑minute slot up to the end of the published price horizon (about 35 h once tomorrow's prices are in). The physics of every transition is exact; the search is not, because one path is kept per SOC bucket. See [what is exact and what is approximate](docs/scheduling-algorithm.md#conservative-quantization-bucket-label-plus-exact-path-energy).
- **Self‑learning** — learns actual charge rates per SOC and temperature band from observed behaviour (efficiency is configured, not learned — there is no independent AC meter to learn it from); builds a statistical, time‑of‑day **load profile**.
- **Temperature‑aware charge rates** — predicts slower charging when the battery is cold for more accurate scheduling.
- **PV‑aware** — uses Solcast forecasts and a live PV sensor to avoid grid‑charging when solar will cover it.
- **Battery cost tracking** — weighted-average landed cost of stored energy, persisted across restarts and exposed for reporting; the DP optimizes forecast cash flows directly.
- **Direct WIT control** — applies modes in real time through the Growatt integration's `set_wit_mode` service (grid_charge, discharge_to_load, max_export, hold, …).
- **Dashboard + manual controls** — HA package with enable/override toggles, manual mode select, force scripts, and rich schedule/status sensors.

---

## How it works

```
Nord Pool prices ─┐
Solcast PV       ─┼─► DP optimizer ─► schedule (15‑min slots to horizon)
learned load     ─┤        │
battery SOC/cost ─┘        └─► real‑time execution  → growatt_modbus/set_wit_mode
```

The optimizer re‑plans on a schedule and adapts when reality drifts from the plan (SOC deviation, new prices, load changes).

---

## Prerequisites

- **Home Assistant OS / Supervised** (the optimizer runs as a local add‑on).
- **Nord Pool** prices — the built‑in HA Nord Pool integration (config entry) or the [HACS Nord Pool](https://github.com/custom-components/nordpool) integration.
- **Growatt Modbus integration with WIT `set_wit_mode` support.** The stock upstream integration does **not** include `set_wit_mode`; this optimizer depends on the WIT‑enabled fork:
  **[jekmanis/Growatt_ModbusTCP](https://github.com/jekmanis/Growatt_ModbusTCP)** (branch `main`, v1.9.6 or later). It must expose the `growatt_modbus/set_wit_mode` service.
- *(Optional)* **Solcast PV Forecast** (HACS) for PV‑aware planning.

---

## Installation

### 1. Copy the add‑on to the HA `addons` share
`addon/battery_optimizer/` is a local Home Assistant add‑on ("app"); its image
also needs the optimizer code, which `scripts/deploy_addon.py` stages next to
it:

```bash
uv run python scripts/deploy_addon.py deploy --dry-run   # checks + file list
uv run python scripts/deploy_addon.py deploy             # copy, install, start
```

By hand: run `scripts/deploy_addon.py stage --out DIR` and copy `DIR` to
`\\<ha>\addons\battery_optimizer`, then Settings → Add‑ons → Add‑on Store →
⋮ → *Check for updates* → **Battery Optimizer** (under *Local add‑ons*) →
Install.

### 2. Configure the add‑on
Every option is the configuration key of the same name and every one is
optional; `addon/battery_optimizer/options.example.yaml` is the annotated
list. Set at minimum (Configuration tab, YAML mode):

```yaml
nordpool_config_entry: "YOUR_NORDPOOL_CONFIG_ENTRY_ID"   # built‑in Nord Pool
nordpool_area: LV

# Growatt sensors (note the device‑prefixed names from integration v0.6.7+)
soc_sensor:            sensor.growatt_battery_battery_soc
pv_power_sensor:       sensor.growatt_solar_solar_total_power
battery_temp_sensor:   sensor.growatt_battery_battery_temperature
battery_charge_sensor: sensor.growatt_battery_battery_charge_today
battery_discharge_sensor: sensor.growatt_battery_battery_discharge_today
load_power_sensor:     sensor.growatt_load_house_consumption

device_id: "YOUR_GROWATT_WIT_DEVICE_ID"   # Developer Tools → States → growatt device

# Match your system
battery_capacity_kwh: 14.3
charge_rate_kw: 4.5
discharge_rate_kw: 5.9

shadow_mode: false
```

No URL and no token: the Supervisor provides both (`SUPERVISOR_TOKEN`,
`http://supervisor/core`). A fresh install starts with `shadow_mode: true` —
nothing is sent to the inverter or written to an `input_*` helper, and every
published entity gets an `_shadow` suffix — so it can run next to another
instance until you switch it to live. See `addon/battery_optimizer/DOCS.md`.

**Finding `device_id`:** Developer Tools → States → open a `growatt_*` entity → copy the `device_id` attribute (or read it from the device page URL).

### 3. Install the HA package (entities, scripts, dashboard sensors)
```bash
cp homeassistant/packages/battery_optimizer.yaml /config/packages/
```
Ensure packages are enabled in `configuration.yaml`:
```yaml
homeassistant:
  packages: !include_dir_named packages
```

### 4. Restart
Restart Home Assistant, then (re)start the add‑on. Watch **Settings → Add‑ons → Battery Optimizer → Log** (or `scripts/deploy_addon.py logs`) for `Battery Optimizer version …`, `Direct control enabled via growatt_modbus/set_wit_mode` and the first optimization.

---

## Usage

### Automatic operation
- **Full optimization** daily at `tomorrow_prices_hour` + 15 min (14:15 with the default, after Nord Pool publishes tomorrow's prices) and at startup.
- **Schedule execution** every slot (15 min): the slot's mode is sent through `set_wit_mode` and verified after `verify_delay_seconds`.
- **Adaptive re‑evaluation** every `adaptive_recalc_minutes` (15): checks the price horizon and re‑plans on SOC deviation, PV shortfall or new prices. It never fetches prices itself; a bounded retry does that when the horizon is unusable.
- **Sampling**: PV power every `pv_sample_seconds` (60 s), battery temperature and load every slot; battery cost updates on every change of the inverter's energy counters.

### Manual controls

| Entity | Purpose |
|--------|---------|
| `input_boolean.battery_optimizer_enabled` | Master enable/disable |
| `input_boolean.battery_optimizer_override` | Enable manual override |
| `input_select.battery_manual_mode` | Auto / Charge / Hold / Discharge |

**Scripts:** `script.battery_force_charge`, `script.battery_force_discharge`, `script.battery_force_hold`, `script.battery_resume_auto`.

### Status & schedule sensors

The plan is published across three entities, because Home Assistant's recorder
stores no attributes at all once the serialized blob passes 16 384 bytes and the
per‑slot list alone exceeds that on a two‑day horizon:

| Entity | Contents | Recorded |
|--------|----------|----------|
| `sensor.battery_optimizer` | `current_mode`, `next_charge`, `next_discharge`, `battery_avg_cost`, `slot_minutes`, the schedule summary (`schedule_slots`, `schedule_start`/`schedule_end`, `charge_hours`/`discharge_hours`/`hold_hours`, `charge_slots_count`/`discharge_slots_count`), `schedule_entity`, plus decision‑transparency fields | Yes — kept under 8 KB on purpose |
| `sensor.battery_optimizer_schedule` | the per‑slot `schedule` list (`time`, `mode`, `wit_mode`, `wit_mode_name`, `export`, `reason`, `value`, `value_basis`), `slot_minutes`, `last_optimization`, the same summary | No — recorder‑excluded |
| `sensor.battery_optimizer_load_profile` | `load_profile_stats` (hourly aggregates for the dashboard chart), `slot_minutes`, `default_load_w`, `observations` | No — recorder‑excluded |

`sensor.battery_optimizer_schedule_markdown` (dashboard table) and
`sensor.battery_learning_stats` (which carries the temperature‑aware rate
tables) complete the set; the markdown sensor is recorder‑excluded too. Helper
template sensors (confidence, learned rate, profit, energy totals, schedule
hours, next charge/discharge times) are created by the HA package for
dashboards.

---

## Configuration reference

Common parameters (see `addon/battery_optimizer/options.example.yaml` for the full, commented list):

| Parameter | Example | Description |
|-----------|---------|-------------|
| `battery_capacity_kwh` | 14.3 | Usable battery capacity |
| `charge_rate_kw` / `discharge_rate_kw` | 4.5 / 5.9 | Power used for planning |
| `min_soc` / `max_soc` | 10 / 100 | SOC bounds (%) |
| `efficiency` / `inverter_efficiency` | 0.95 / 0.97 | Storage charge-retention factor / symmetric AC↔DC conversion factor (about 89.4% implied AC round trip) |
| `slot_minutes` | 15 | Plan resolution (matches Nord Pool 15‑min) |
| `grid_fee_eur_kwh` / `grid_export_fee_eur_kwh` | 0.052 / 0.02 | Import fees added / export fee subtracted |
| `import_price_multiplier` | 1.0 | Multiplier applied to spot plus import fees; use 1.21 only when those inputs exclude 21% VAT |
| `battery_wear_cost_eur_kwh` | 0.017 | Per‑kWh wear cost discouraging marginal cycling |
| `terminal_energy_value_eur_kwh` | `auto` | Values stored DC energy at the price horizon. `0` = no-salvage mode — see below |
| `pv_threshold_w` | 500 | PV above which grid charging pauses |
| `solcast_today_entity` / `_tomorrow_entity` | `sensor.solcast_*` | Optional PV forecast |
| `device_id` | `""` | **Empty = dry‑run** (logs decisions, no inverter writes) |
| `set_wit_mode_timeout_seconds` | 15 | Per‑call `hass_timeout`. This call **blocks its callback under the app lock** — see *Slow callbacks* |
| `verify_delay_seconds` | 90 | Delay before the first verify‑after‑set read of the configured `verify_source` (holding registers by default) |
| `verify_recheck_seconds` | 60 | Delay of the single re‑check performed after a resend |
| `verify_source` | `auto` | `registers` / `mode_sensor` / `none` / `auto` (registers whenever `device_id` is set, otherwise none — never the mode sensor by default) |
| `callback_warn_seconds` | 10 | Warn when one of this app's callbacks blocks for longer than this |
| `tomorrow_prices_hour` | 14 | Local hour from which tomorrow's intervals are expected. Before it, a today‑only reply is complete; from it, a missing tomorrow is an incomplete horizon |
| `price_retry_enabled` | true | Retry a failed or incomplete price fetch automatically — see *Price recovery* |
| `price_retry_delays_seconds` | `[30, 120, 300]` | Backoff for the 1st/2nd/3rd retry (list or `"30,120,300"`); every further attempt waits `price_retry_max_seconds` |
| `price_retry_max_seconds` | 900 | Cap for the backoff |
| `price_retain_max_age_hours` | 36 | How long already‑fetched **future** intervals stay reusable **without any non‑empty reply** — a backstop for a silent source, not a per‑interval age |

### Price recovery

A fetch that returns nothing — or one that returns today but no tomorrow after
`tomorrow_prices_hour` — used to leave the optimizer on an old or absent plan
until the next daily optimization: a slot with no entry simply applied
`HOLD/no_schedule`.

The app now judges *coverage* (current interval present, no gaps, horizon
reaching the end of the current publication window) and, when it is unusable,
schedules **one** retry on a bounded backoff. On success it rebuilds from the
current SOC and time and applies the result through the normal execution path,
so the enable switch and the manual override still decide whether anything is
sent to the inverter. While waiting, the safe `HOLD` stands — recovery never
invents a price.

**A price is never manufactured for the interval you are in.** When the fetched
data does not contain the current interval, planning starts at the next one it
does contain, and the current slot is resolved in one of two ways: the entry
already in the plan is kept if it was itself built from a published price, or
the slot applies `HOLD/no_price` until the retry brings the real price in.
Nothing is derived from yesterday's prices or from the neighbouring interval.
`sensor.battery_optimizer`'s `price_horizon` attribute reports this as
`current_slot_priced` and `current_slot_entry`
(`planned` / `retained` / `fallback`).

An incomplete horizon is *noted*, never acted on: missing tomorrow between
`tomorrow_prices_hour` and publication is normal, so the periodic pass records
it and still runs the reactive PV‑shortfall check.

Coverage state is published on `sensor.battery_optimizer` under the
`price_horizon` attribute (`ok`, `reason`, `horizon_end`, `required_end`,
`last_success_horizon_end`, `last_failure_reason`, `retry_pending`,
`retry_attempts`).

Next to it, `rate_refinement` reports how the current plan's charge rates were
settled: `branch` (`single_solve`, `converged`, `conservative_fallback`,
`degraded`), `passes`, and `shortfall_kwh`. `degraded` means the pack cannot
take the charge energy the plan was chosen on at the temperatures it reaches —
nothing published credits that energy, but the plan is no longer the cheapest
one. It is rare and is also logged at WARNING.

**Set a region time zone in Home Assistant** (Settings → System → General). The
add‑on adopts HA's zone; the "end of tomorrow" boundary needs real DST rules,
and when no usable zone is reported the app falls back to the current UTC
offset and warns once, and that boundary is an hour off on the two DST
transition days.

### End‑of‑horizon value (`0` = no‑salvage mode)

`terminal_energy_value_eur_kwh` prices whatever energy is still in the battery
when the price horizon ends. With `auto` it is derived from the median forecast
import price, discharge conversion and wear — a salvage value, not a terminal
SOC target.

Setting it to **`0` says stored energy is worthless at the horizon**, so the
optimal plan is always to spend it there. That shows up as every schedule
ending like:

```
07-30 00:30  DISCHARGE  ... (until depleted) [EXPORT] -> 11.2%
```

In practice this is usually harmless: those slots sit ~32 h out, and the daily
13:15 re‑optimization extends the horizon with tomorrow's prices long before
they execute.

**Neither setting is universally right**, so the app only states which mode is
active — at INFO, at startup and (rate‑limited) in the DP log:

```
INFO terminal_energy_value_eur_kwh=0 is no-salvage mode: ...
     Neither is universally correct — pick per installation.
```

| Setting | Failure mode |
|---|---|
| `0` | spends the battery at the horizon edge |
| `auto` | strands charge there; skips evening slots priced below the median |

On the reference installation `"auto"` was tried and reverted: it stranded ~77% SOC at the horizon edge and skipped evening slots priced below the median, which cost more than the end-of-horizon spend it prevented.

Choose per installation and record the reason next to the value in your options.

By default, spot prices and import fees are assumed to already use the desired
VAT basis. `import_price_multiplier` can apply VAT to the combined variable
import price when all those inputs are VAT-exclusive. Do not use it when the
source price or configured fees already include VAT. Import margins,
distribution charges, and export deductions are contract-specific; verify the
example values against your bill.

> **Upgrade note:** older releases stored a raw spot-price average in
> `input_number.battery_avg_cost`; the current tracker stores landed cost per
> battery kWh. The package includes `battery_cost_basis_version`, which is
> created at version 2 (current basis) — a legacy value is never converted
> automatically. To convert a raw-spot average from a pre-landed-cost install,
> set the helper to 1 and restart the add‑on once: the value is conservatively
> migrated as grid-charged energy and the helper is stamped back to 2 (look
> for the "Migrated legacy raw battery cost" log line to confirm it ran).
> Alternatively, just reset `input_number.battery_avg_cost` to a reasonable
> landed-cost estimate manually.

---

## Architecture

```
addon/battery_optimizer/           # The HA add-on: config.yaml (options schema), Dockerfile,
                                  # run.sh, DOCS.md, options.example.yaml
appdaemon/apps/                   # (historical name) the optimizer itself
├── battery_optimizer.py          # Orchestrator (scheduling, execution)
└── battery_optimizer_lib/
    ├── ha_host.py                # Host: HA websocket + REST, scheduler, worker pool
    ├── addon_main.py             # Add-on entry point: options -> config -> app
    ├── config.py                 # Typed config loader
    ├── callback_lock.py          # App‑wide re‑entrant lock behind every callback
    ├── models.py                 # BatteryMode, ScheduleEntry, … data types
    ├── dp_optimizer.py           # Dynamic‑programming SOC scheduler
    ├── slot_energy.py            # The one pure slot transition (named units)
    ├── soc_projection.py         # Shared slot‑SOC model, delegates to slot_energy
    ├── plan_validation.py        # Continuous replay of the final plan
    ├── learning_engine.py        # Charge‑rate / efficiency learning
    ├── thermal_model.py          # Shared battery temperature model (k1/k2)
    ├── ambient_service.py        # T_ambient(t) across the horizon
    ├── load_profile.py           # Statistical load forecasting
    ├── pv_profile.py             # Statistical PV production profile
    ├── load_prediction_tracker.py # Predicted vs actual load accuracy
    ├── pv_forecast_service.py    # Solcast PV forecast integration
    ├── pv_bias_tracker.py        # Sliding PV forecast bias
    ├── price_service.py          # Nord Pool price fetching
    ├── price_horizon.py          # Price recovery and horizon health
    ├── direct_control.py         # Real‑time control via set_wit_mode
    ├── cost_tracker.py           # Stored‑energy cost tracking
    ├── schedule_formatter.py     # Schedule → sensor/dashboard formatting
    ├── soc_deviation.py          # Detects unexpected SOC changes
    ├── slot_outcome_tracker.py   # Per‑slot outcome and mode compliance
    ├── ha_helpers.py             # HA state reading helpers
    └── timezone_utils.py         # TZ‑aware datetime helpers
homeassistant/packages/
└── battery_optimizer.yaml        # HA entities, scripts, automations, template sensors
docs/                             # Algorithm & analysis notes
tests/                            # pytest suite (library, host, add-on, shadow run)
```

See [docs/scheduling-algorithm.md](docs/scheduling-algorithm.md) for the optimizer internals.

---

## Development

Python project managed with [`uv`](https://docs.astral.sh/uv/); no enforced linter/formatter.

```bash
uv run python -m py_compile appdaemon/apps/battery_optimizer.py   # syntax check
uv run pytest tests/ -v                                           # run tests
uv run pytest tests/ --cov=appdaemon/apps --cov-report=term-missing
```

Library modules are tested standalone. `tests/fake_ha.py` is a fake Home Assistant (websocket + REST on a local socket); `conftest.py` exposes it and a connected host as fixtures, and `tests/test_addon_shadow_run.py` runs the real orchestrator against it in shadow mode.

---

## Troubleshooting

- **Logs:** Settings → Add‑ons → Battery Optimizer → Log, or `uv run python scripts/deploy_addon.py logs`.
- **Dry‑run:** set `device_id: ""` to log decisions without touching the inverter; `shadow_mode: true` additionally blocks every helper write and suffixes the published entities.
- **Entities `unavailable` / `not found`:** confirm the Growatt sensor names match your install (integration v0.6.7+ device‑prefixes them, e.g. `sensor.growatt_battery_battery_soc`).
- **`set_wit_mode` not found:** you're on the stock Growatt integration — install the [WIT fork](https://github.com/jekmanis/Growatt_ModbusTCP).
- **`set_wit_mode` timeouts:** many sequential VPP register writes on a busy Modbus link can exceed the host's default 10 s service window. The optimizer sets that per-call window from `set_wit_mode_timeout_seconds` (**default 15 s**) and inspects the service response. If the call still times out client-side (`ad_status: TIMEOUT`), the mode is treated as *unconfirmed* (logged at WARNING) rather than silently assumed applied — verify-after-set covers that case, which is why a short timeout is safe and a long one is not (it blocks every other callback). A confirmed failure (the service raised) is logged at ERROR and is **not** recorded as sent, so it is retried on the next slot instead of being masked by duplicate suppression.

- **Mode mismatches / "resending once":** `verify_delay_seconds` (default 90 s) after every mode change — including `passthrough` — DirectControl consults the source selected by `verify_source` (`auto` → the holding registers 30407-30410 / 30200-30201 whenever `device_id` is set; `mode_sensor` → the `inverter_mode_sensor` entity; `none` → no check). No entity is guessed: `auto` never falls back to the mode sensor. On a genuine mismatch it resends once and then re-checks **exactly once** after `verify_recheck_seconds` (default 60 s). If that second read still disagrees, the app logs an **ERROR** ("persistent mode mismatch after resend") and stops — never a third send, never a loop; the next slot retries normally.

  Counters live on `sensor.battery_inverter_control_health` (and as the `inverter_control_health` attribute of `sensor.battery_optimizer`). Use them to tell the two causes apart:

  | Symptom | Reading | Fix |
  |---|---|---|
  | HA modbus sensor merely lags | `resend_recovered_count` ≈ `resend_count`, `persistent_mismatch_count` = 0 | Raise `verify_delay_seconds` |
  | Inverter really drops the override | `persistent_mismatch_count` growing | Inverter/firmware/config, not timing |
  | Service itself failing | `resend_failed_count` growing | Check the Modbus connection |

  The sensor is created with `set_state`, so it disappears after an HA restart until the app republishes it — alert on trends, don't rely on its history.

- **Slow callbacks:** `set_wit_mode` is a **synchronous, blocking** service call made from a callback. The add-on runs callbacks on `worker_threads` threads (default 4), and the orchestrator serializes them itself: `_timed_callback` runs every callback under one app-wide re-entrant lock (`battery_optimizer_lib/callback_lock.py`), and the only region that drops it is the blocking `set_wit_mode` write in `_apply_mode_tracked` — so other callbacks keep running while the inverter write is in flight. The app measures its own callbacks (lock wait included), warns above `callback_warn_seconds` naming the offending callback, and after three overruns points at `set_wit_mode_timeout_seconds` and the inverter integration.

---

## Disclaimer

This project controls real battery and grid hardware. It is provided **as‑is, without warranty**. Incorrect configuration can cause unwanted grid import/export, battery wear, or missed savings. Test in dry‑run first and monitor before relying on it.
