# Battery Optimizer for Growatt WIT Inverter

An [AppDaemon](https://appdaemon.readthedocs.io/) application for Home Assistant that uses **Nord Pool** day‑ahead electricity prices (and optionally **Solcast** PV forecasts) to compute and execute an optimal battery **charge / hold / discharge** schedule for a Growatt **WIT** hybrid inverter.

It plans with dynamic programming over SOC, learns your house load and real charge rates over time, tracks the stored‑energy cost, and drives the inverter both in real time (via a `set_wit_mode` service) and autonomously (by writing the inverter's Time‑of‑Use registers so it keeps following the plan even if Home Assistant goes offline).

> ⚠️ This software actively controls battery hardware (grid charging, export, discharge). Use at your own risk and verify behaviour on your own system. See [Disclaimer](#disclaimer).

---

## Features

- **DP price optimizer** — dynamic programming over discretised SOC finds the cost‑optimal charge/hold/discharge sequence for the next ~48 h at 15‑minute resolution.
- **Self‑learning** — learns actual charge rates per SOC band and round‑trip efficiency from observed behaviour; builds a statistical, time‑of‑day **load profile**.
- **Temperature‑aware charge rates** — predicts slower charging when the battery is cold for more accurate scheduling.
- **PV‑aware** — uses Solcast forecasts and a live PV sensor to avoid grid‑charging when solar will cover it.
- **Battery cost tracking** — weighted-average landed cost of stored energy, persisted across restarts and exposed for reporting; the DP optimizes forecast cash flows directly.
- **Direct WIT control** — applies modes in real time through the Growatt integration's `set_wit_mode` service (grid_charge, discharge_to_load, max_export, hold, …).
- **Dashboard + manual controls** — HA package with enable/override toggles, manual mode select, force scripts, and rich schedule/status sensors.

---

## How it works

```
Nord Pool prices ─┐
Solcast PV       ─┼─► DP optimizer ─► schedule (96 × 15‑min slots)
learned load     ─┤        │
battery SOC/cost ─┘        └─► real‑time execution  → growatt_modbus/set_wit_mode
```

The optimizer re‑plans on a schedule and adapts when reality drifts from the plan (SOC deviation, new prices, load changes).

---

## Prerequisites

- **Home Assistant** with the **AppDaemon 4** add‑on.
- **Nord Pool** prices — the built‑in HA Nord Pool integration (config entry) or the [HACS Nord Pool](https://github.com/custom-components/nordpool) integration.
- **Growatt Modbus integration with WIT `set_wit_mode` support.** The stock upstream integration does **not** include `set_wit_mode`; this optimizer depends on the WIT‑enabled fork:
  **[jekmanis/Growatt_ModbusTCP](https://github.com/jekmanis/Growatt_ModbusTCP)** (branch `main`, v0.9.3+). It must expose the `growatt_modbus/set_wit_mode` service.
- *(Optional)* **Solcast PV Forecast** (HACS) for PV‑aware planning.
- A long‑lived HA access token (used by the app to read Nord Pool prices via the REST API).

---

## Installation

### 1. Install the AppDaemon add‑on
Settings → Add‑ons → Add‑on Store → **AppDaemon 4** → Install → enable *Start on boot* and *Watchdog*.

### 2. Deploy the app + library
Copy the app **and** its `battery_optimizer_lib/` package into your AppDaemon apps directory (e.g. `/addon_configs/<appdaemon>/apps/` or `/config/appdaemon/apps/`, depending on your install):

```bash
cp -r appdaemon/apps/battery_optimizer.py \
      appdaemon/apps/battery_optimizer_lib \
      <your_appdaemon>/apps/
```

### 3. Configure the app
Copy the example config and fill in your values:

```bash
cp appdaemon/apps/apps.yaml.example <your_appdaemon>/apps/apps.yaml
```

Edit `apps.yaml` and set at minimum:

```yaml
battery_optimizer:
  ha_url: "http://homeassistant.local:8123"
  ha_token: "REPLACE_WITH_YOUR_HA_LONG_LIVED_TOKEN"

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
```

> 🔒 `apps.yaml` holds your HA token and is **gitignored**. Only `apps.yaml.example` is committed — never commit your real `apps.yaml`.

**Finding `device_id`:** Developer Tools → States → open a `growatt_*` entity → copy the `device_id` attribute (or read it from the device page URL).

### 4. Install the HA package (entities, scripts, dashboard sensors)
```bash
cp homeassistant/packages/battery_optimizer.yaml /config/packages/
```
Ensure packages are enabled in `configuration.yaml`:
```yaml
homeassistant:
  packages: !include_dir_named packages
```

### 5. Restart
Restart Home Assistant, then restart the AppDaemon add‑on. Watch **Settings → Add‑ons → AppDaemon → Log** for `Direct control enabled via growatt_modbus/set_wit_mode` and the first optimization.

---

## Usage

### Automatic operation
- **Full optimization** daily at ~13:15 (after Nord Pool publishes tomorrow's prices) and at startup.
- **Adaptive re‑evaluation** every 15 minutes (re‑plans if prices/SOC/load drift).
- **Safety checks** every 5 minutes.

### Manual controls

| Entity | Purpose |
|--------|---------|
| `input_boolean.battery_optimizer_enabled` | Master enable/disable |
| `input_boolean.battery_optimizer_override` | Enable manual override |
| `input_select.battery_manual_mode` | Auto / Charge / Hold / Discharge |

**Scripts:** `script.battery_force_charge`, `script.battery_force_discharge`, `script.battery_force_hold`, `script.battery_resume_auto`.

### Status & schedule sensors
`sensor.battery_optimizer` carries the live plan as attributes: `current_mode`, `schedule` (per‑slot list), `slot_minutes`, `next_charge`, `next_discharge`, `battery_avg_cost`, plus decision‑transparency fields. Helper template sensors (confidence, learned rate, profit, energy totals, schedule hours, next charge/discharge times) are created by the HA package for dashboards.

---

## Configuration reference

Common parameters (see `apps.yaml.example` for the full, commented list):

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
| `terminal_energy_value_eur_kwh` | `auto` | Values stored DC energy at the price horizon; `0` restores legacy depletion behavior |
| `pv_threshold_w` | 500 | PV above which grid charging pauses |
| `solcast_today_entity` / `_tomorrow_entity` | `sensor.solcast_*` | Optional PV forecast |
| `device_id` | `""` | **Empty = dry‑run** (logs decisions, no inverter writes) |

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
> set the helper to 1 and restart AppDaemon once: the value is conservatively
> migrated as grid-charged energy and the helper is stamped back to 2 (look
> for the "Migrated legacy raw battery cost" log line to confirm it ran).
> Alternatively, just reset `input_number.battery_avg_cost` to a reasonable
> landed-cost estimate manually.

---

## Architecture

```
appdaemon/apps/
├── battery_optimizer.py          # AppDaemon orchestrator (scheduling, execution)
├── apps.yaml.example             # Config template (copy to apps.yaml)
└── battery_optimizer_lib/
    ├── config.py                 # Typed config loader
    ├── models.py                 # BatteryMode, ScheduleEntry, … data types
    ├── dp_optimizer.py           # Dynamic‑programming SOC scheduler
    ├── learning_engine.py        # Charge‑rate / efficiency learning
    ├── load_profile.py           # Statistical load forecasting
    ├── pv_forecast_service.py    # Solcast PV forecast integration
    ├── price_service.py          # Nord Pool price fetching
    ├── direct_control.py         # Real‑time control via set_wit_mode
    ├── cost_tracker.py           # Stored‑energy cost tracking
    ├── schedule_formatter.py     # Schedule → sensor/dashboard formatting
    ├── soc_deviation.py          # Detects unexpected SOC changes
    ├── charge_rate_utils.py      # Temperature‑aware rate computation
    ├── ha_helpers.py             # HA state reading helpers
    └── timezone_utils.py         # TZ‑aware datetime helpers
homeassistant/packages/
└── battery_optimizer.yaml        # HA entities, scripts, automations, template sensors
docs/                             # Algorithm & analysis notes
tests/                            # pytest suite (library modules)
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

`conftest.py` mocks the AppDaemon runtime so the `battery_optimizer_lib` modules can be tested standalone. The orchestrator (`battery_optimizer.py`) is validated via dry‑run (`device_id: ""`).

---

## Troubleshooting

- **Logs:** Settings → Add‑ons → AppDaemon → Log.
- **Dry‑run:** set `device_id: ""` to log decisions without touching the inverter.
- **Entities `unavailable` / `not found`:** confirm the Growatt sensor names match your install (integration v0.6.7+ device‑prefixes them, e.g. `sensor.growatt_battery_battery_soc`).
- **`set_wit_mode` not found:** you're on the stock Growatt integration — install the [WIT fork](https://github.com/jekmanis/Growatt_ModbusTCP).
- **`set_wit_mode` timeouts:** many sequential VPP register writes on a busy Modbus link can exceed AppDaemon's default 10 s service window. The optimizer now raises that per-call window to 30 s (`hass_timeout`) and inspects the service response. If AppDaemon still times out client-side (returns `None`), the mode is treated as *unconfirmed* (logged at WARNING) rather than silently assumed applied. About 90 s after every mode change (including `passthrough`), DirectControl reads the integration's **Inverter Mode** sensor (`sensor.growatt_inverter_mode` by default, overridable via `inverter_mode_sensor`) and, on a genuine mismatch, resends the command exactly once. A confirmed failure (the service raised) is logged at ERROR and is **not** recorded as sent, so it is retried on the next slot instead of being masked by duplicate suppression.

---

## Disclaimer

This project controls real battery and grid hardware. It is provided **as‑is, without warranty**. Incorrect configuration can cause unwanted grid import/export, battery wear, or missed savings. Test in dry‑run first and monitor before relying on it.
