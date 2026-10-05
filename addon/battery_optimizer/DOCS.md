# Battery Optimizer

Plans a Growatt WIT battery's charge / hold / discharge schedule from Nord Pool
prices, load and PV forecasts, and executes it through
`growatt_modbus/set_wit_mode`. Same optimizer that ran under AppDaemon; this
add-on is only its host.

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

Every optimizer option is the apps.yaml key of the same name, and every one is
optional: an option you leave out takes the code default. See
`options.example.yaml` in the repository. Convert an existing apps.yaml with

```
uv run python scripts/addon_options.py apps.yaml [--live] --out options.json
```

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
same `/config/...` paths apps.yaml used, and are part of every HA backup.

## Threads and blocking

Callbacks run on `worker_threads` threads, concurrently; the app serializes its
own state with one lock and releases it only around the blocking
`set_wit_mode` write (`set_wit_mode_timeout_seconds`, default 15 s). A
`call_service` that HA does not answer in time is reported as UNCONFIRMED, not
as a failure; register verification decides.

## Cutover from AppDaemon

1. Stop the AppDaemon add-on (keep it installed: it is the rollback).
2. Copy the four JSON files from `\\<ha>\addon_configs\a0d7b954_appdaemon\`
   into `\\<ha>\addon_configs\local_battery_optimizer\` (fresh copies - the
   shadow run kept its own).
3. Set the options with `shadow_mode: false` and the real `device_id`
   (`scripts/addon_options.py <live apps.yaml> --live --out live.json`, then
   `scripts/deploy_addon.py options live.json`).
4. Check: `sensor.battery_optimizer` shows this add-on's `app_version`;
   `sensor.battery_inverter_control_health` shows register matches and no
   `persistent_mismatch_count`; the log has no Traceback.
5. Delete the leftover `*_shadow` entities (they are not refreshed any more
   and disappear at the next HA restart).

## Rollback

One command each way, nothing to copy back:

1. `uv run python scripts/deploy_addon.py stop` (this add-on), or set
   `shadow_mode: true` to keep it running as a shadow.
2. `uv run python scripts/deploy_addon.py start --slug a0d7b954_appdaemon`.

AppDaemon resumes from its own JSON files, which are older than this add-on's
by the length of the live run; the learning engine catches up within a day.
If the add-on's learned data should survive the rollback, copy the four JSON
files back into `\\<ha>\addon_configs\a0d7b954_appdaemon\` while AppDaemon is
still stopped.
