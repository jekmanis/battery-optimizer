# Battery Optimizer

Plans a Growatt WIT battery's charge / hold / discharge schedule from Nord Pool
prices, load and PV forecasts, and executes it through
`growatt_modbus/set_wit_mode`.

## How it talks to Home Assistant

- Websocket `ws://supervisor/core/websocket`: state cache, `state_changed`
  listeners, service calls (prices, inverter commands, register read-back).
- REST `POST http://supervisor/core/api/states/<entity>`: the published
  entities (`sensor.battery_optimizer`, `sensor.battery_optimizer_schedule`,
  `sensor.battery_optimizer_load_profile`,
  `sensor.battery_optimizer_schedule_markdown`,
  `sensor.battery_inverter_control_health`, ...). Same names as before, so the
  dashboards and `homeassistant/packages/battery_optimizer.yaml` are unchanged.
- Authentication is the Supervisor's `SUPERVISOR_TOKEN`; no long-lived token
  is configured anywhere.

## Options

Every optimizer option is optional: an option you leave out takes the code
default. `options.example.yaml` in the repository documents every key. Set
them in the Configuration tab, or from a file:

```
uv run python scripts/deploy_addon.py export-options options.json   # current options
uv run python scripts/deploy_addon.py options options.json          # set + restart
```

A legacy `apps.yaml` app block converts with
`uv run python scripts/addon_options.py apps.yaml [--live] --out options.json`.

Add-on-level options:

| option | meaning |
|---|---|
| `shadow_mode` | `true`: parallel run. `device_id` is forced empty (dry run), every service except the read-only ones (`nordpool/get_price_indices_for_date`, `weather/get_forecasts`, `growatt_modbus/get_register_data`) is suppressed - including every `input_*` helper write - and every published entity gets `entity_suffix`. A fresh install starts in shadow mode. |
| `entity_suffix` | suffix for shadow entities, default `_shadow` |
| `log_level` | `debug` / `info` / `warning` / `error` |
| `worker_threads` | callback worker threads, default 4 |

## Data

`/config` in the container is `/addon_configs/local_battery_optimizer` on the
host (SMB `\\<ha>\addon_configs\local_battery_optimizer`). The learning data,
load profile, PV profile and prediction tracker JSON files live there under the
same `/config/...` paths, and are part of every HA backup.

## Threads and blocking

Callbacks run on `worker_threads` threads, concurrently; the app serializes its
own state with one lock and releases it only around the blocking
`set_wit_mode` write (`set_wit_mode_timeout_seconds`, default 15 s). A
`call_service` that HA does not answer in time is reported as UNCONFIRMED, not
as a failure; register verification decides.

## History and rollback

Until 2026-10-05 the optimizer ran on AppDaemon. That add-on
(`a0d7b954_appdaemon`) is kept installed, stopped and `boot: manual` - with
`auto` an HA reboot would start a second instance commanding the inverter.

Rollback, two commands each way:

1. `uv run python scripts/deploy_addon.py stop` and
   `uv run python scripts/deploy_addon.py boot manual` (this add-on), or set
   `shadow_mode: true` to keep it running as a shadow.
2. `uv run python scripts/deploy_addon.py boot auto --slug a0d7b954_appdaemon`
   and `uv run python scripts/deploy_addon.py start --slug a0d7b954_appdaemon`.

The old instance resumes from its own JSON files, which are as old as the
cutover; to carry the learned data back, copy the four JSON files into
`\\<ha>\addon_configs\a0d7b954_appdaemon\` before starting it.
