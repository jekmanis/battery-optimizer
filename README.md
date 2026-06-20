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
- **Battery cost tracking** — weighted‑average cost of stored energy, persisted across restarts, used to avoid selling/using energy below what it cost.
- **Rolling TOU sync** — writes the schedule into the inverter's TOU registers with a rolling day boundary so it survives an HA outage (see [Rolling TOU Schedule Sync](#rolling-tou-schedule-sync)).
- **Direct WIT control** — applies modes in real time through the Growatt integration's `set_wit_mode` service (grid_charge, discharge_to_load, max_export, hold, …).
- **Dashboard + manual controls** — HA package with enable/override toggles, manual mode select, force scripts, and rich schedule/status sensors.

---

## How it works

```
Nord Pool prices ─┐
Solcast PV       ─┼─► DP optimizer ─► schedule (96 × 15‑min slots)
learned load     ─┤        │
battery SOC/cost ─┘        ├─► real‑time execution  → growatt_modbus/set_wit_mode
                           └─► autonomous fallback   → inverter TOU registers (rolling sync)
```

The optimizer re‑plans on a schedule and adapts when reality drifts from the plan (SOC deviation, new prices, load changes).

---

## Prerequisites

- **Home Assistant** with the **AppDaemon 4** add‑on.
- **Nord Pool** prices — the built‑in HA Nord Pool integration (config entry) or the [HACS Nord Pool](https://github.com/custom-components/nordpool) integration.
- **Growatt Modbus integration with WIT `set_wit_mode` support.** The stock upstream integration does **not** include `set_wit_mode`; this optimizer depends on the WIT‑enabled fork:
  **[jekmanis/Growatt_ModbusTCP](https://github.com/jekmanis/Growatt_ModbusTCP)** (branch `main`, v0.9.3+). It must expose the services `growatt_modbus/set_wit_mode`, `write_register`, `write_registers`, and `get_register_data`.
- *(Optional)* **Solcast PV Forecast** (HACS) for PV‑aware planning.
- A long‑lived HA access token (used by the app to read Nord Pool prices and write registers via the REST API).

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
`sensor.battery_optimizer` carries the live plan as attributes: `current_mode`, `schedule` (per‑slot list), `slot_minutes`, `next_charge`, `next_discharge`, `battery_avg_cost`, plus decision‑transparency fields. Helper template sensors (confidence, learned rate, efficiency, profit, energy totals, schedule hours) are created by the HA package for dashboards.

---

## Rolling TOU Schedule Sync

The optimizer syncs its schedule to the inverter's Time‑of‑Use (TOU) registers so the inverter keeps following the plan even if Home Assistant goes offline.

**The problem:** TOU registers store time‑of‑day only (00:00–23:59), not dates. After midnight, "afternoon" TOU entries would still reflect yesterday's plan until the next sync.

**The solution — rolling boundary:** every cycle the schedule is written using the start of the currently active TOU period as a boundary:
- hours **before** the boundary (already passed today) → use **tomorrow's** schedule
- hours **from** the boundary onward → use **today's** schedule

```
Period 00:00-05:59: boundary 00:00 → all hours = today
Period 06:00-13:59: boundary 06:00 → 00:00-05:59 = tomorrow, rest = today
Period 14:00-18:59: boundary 14:00 → 00:00-13:59 = tomorrow, rest = today
Period 19:00-23:59: boundary 19:00 → 00:00-18:59 = tomorrow, rest = today
```

So if HA dies at 18:00, the inverter already holds a valid plan for the rest of today **and** tomorrow's early hours. Writes are skipped when the proposed TOU is byte‑identical to what's already programmed, avoiding needless 20‑second write cycles.

---

## Configuration reference

Common parameters (see `apps.yaml.example` for the full, commented list):

| Parameter | Example | Description |
|-----------|---------|-------------|
| `battery_capacity_kwh` | 14.3 | Usable battery capacity |
| `charge_rate_kw` / `discharge_rate_kw` | 4.5 / 5.9 | Power used for planning |
| `min_soc` / `max_soc` | 10 / 100 | SOC bounds (%) |
| `efficiency` / `inverter_efficiency` | 0.95 / 0.97 | Round‑trip / AC‑DC efficiency |
| `slot_minutes` | 15 | Plan resolution (matches Nord Pool 15‑min) |
| `grid_fee_eur_kwh` / `grid_export_fee_eur_kwh` | 0.052 / 0.02 | Import fees added / export fee subtracted |
| `battery_wear_cost_eur_kwh` | 0.017 | Per‑kWh wear cost discouraging marginal cycling |
| `pv_threshold_w` | 500 | PV above which grid charging pauses |
| `solcast_today_entity` / `_tomorrow_entity` | `sensor.solcast_*` | Optional PV forecast |
| `modbus_hub` | `wit_inverter` | Enables TOU sync (empty = mode commands only) |
| `device_id` | `""` | **Empty = dry‑run** (logs decisions, no inverter writes) |

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
    ├── tou_sync.py               # Inverter TOU register sync (raw Modbus)
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
- **`set_wit_mode` timeouts:** many sequential VPP register writes on a busy Modbus link can exceed AppDaemon's 10 s service window; the command usually still applies. Verify the inverter mode actually changed.
- **TOU write failures ("Illegal data value" / "overlap"):** the inverter validates all 20 period registers; the optimizer clears them before each write and retries with backoff. Concurrent reads from the HA coordinator can cause transient bus contention.

---

## Disclaimer

This project controls real battery and grid hardware. It is provided **as‑is, without warranty**. Incorrect configuration can cause unwanted grid import/export, battery wear, or missed savings. Test in dry‑run first and monitor before relying on it.
