# scripts/

Tooling for the Home Assistant add-on (`addon/battery_optimizer/`). Everything
that talks to HA reads the **admin** token from `~/.ha_token` (first line),
never prints it, and authenticates **once** per run — repeated 401s can get the
client IP banned.

## `deploy_addon.py`

Builds, copies and (re)installs the add-on. A local add-on is a directory in
the HA `addons` share (`\\192.168.77.167\addons\battery_optimizer`); the
Supervisor builds the image from it.

```bash
uv run python scripts/deploy_addon.py deploy --dry-run        # every check, prints the file list, writes nothing
uv run python scripts/deploy_addon.py deploy                  # the real thing
uv run python scripts/deploy_addon.py deploy --options o.json # ... and set the options
uv run python scripts/deploy_addon.py status                  # both add-ons: state, version, shadow_mode
uv run python scripts/deploy_addon.py logs [--lines 300] [--slug SLUG]
uv run python scripts/deploy_addon.py options o.json          # set options + restart
uv run python scripts/deploy_addon.py export-options o.json   # current options -> file
uv run python scripts/deploy_addon.py start|stop|restart [--slug SLUG]
uv run python scripts/deploy_addon.py boot auto|manual [--slug SLUG]  # exactly one instance boots with HA
uv run python scripts/deploy_addon.py seed [--force]          # copy the rollback instance's JSON state in
uv run python scripts/deploy_addon.py restore <backup-dir>
uv run python scripts/deploy_addon.py stage --out DIR         # build context only (e.g. for `docker build`)
```

`deploy`, in order:

1. **git** — refuses a dirty tree without `--allow-dirty`; prints branch@commit.
2. **tests** — `pytest tests/ -q` (`--skip-tests` to skip), then a syntax check
   of every shipped module.
3. **stage** — `addon/battery_optimizer/` plus the optimizer in `app/`
   (`battery_optimizer.py`, `battery_optimizer_lib/*.py`, never `__pycache__`);
   `run.sh`/`Dockerfile` forced to LF; `config.yaml`'s `version` set to
   `APP_VERSION`, so the version the Supervisor shows is the one the app logs.
4. **backup** — the current share copy to
   `\\<ha>\share\battery_optimizer_backups\addon-<ts>\`, newest 5 kept.
   **Never under `addons/`**: the Supervisor scans that tree for `config.yaml`,
   and a backup there would be a second add-on with the same slug — the same
   shadowing trap that once made AppDaemon import a backup out of `apps/`.
5. **copy** — mirror into the share, pruning files the stage no longer has,
   then SHA256-verify every file.
6. **Supervisor** — through HA's websocket `supervisor/api` (the REST
   `/api/hassio/addons/<slug>/info` path answers 401 for every token here):
   store reload, then install (first time), update (version changed) or
   rebuild (same version), options if given, then start/restart and wait for
   `started`.
7. **log check** — `GET /api/hassio/addons/<slug>/logs` with
   `Range: entries=:-400:` (works from Python and Git Bash `curl`; Windows
   PowerShell 5.1 refuses the header) until the
   `Battery Optimizer version <APP_VERSION>` line appears; any `Traceback`,
   `ModuleNotFoundError`, `ImportError` or `TypeError` fails the check.

SHA256 proves the bytes on the share; the version line and
`sensor.battery_optimizer.attributes.app_version` prove what runs.

## `addon_options.py`

Converts a legacy `apps.yaml` app block (the pre-add-on format) into add-on
options: host-only keys (`module`, `class`, `pin_app`, `pin_thread`) and the
connection keys (`ha_url`, `ha_token`) are dropped, every other key must be in
the add-on schema, and values are coerced the way the Supervisor coerces them
(a fractional value under an `int` type is an error, not a silent truncation).

```bash
uv run python scripts/addon_options.py apps.yaml                 # shadow options, printed redacted
uv run python scripts/addon_options.py apps.yaml --live --out o.json
```

A shadow conversion sets `shadow_mode: true`, `entity_suffix: _shadow` and
`device_id: ""`.

## `compare_shadow.py`

Prints the live plan (`sensor.battery_optimizer_schedule`) and the shadow plan
(`sensor.battery_optimizer_schedule_shadow`) side by side, matched by UTC
instant, and checks that the current and next 4 slot modes agree and every
mode's total over the common range is within one slot. Exit code 0 = pass.

## `smoke_config.py`

```bash
uv run python scripts/smoke_config.py addon/battery_optimizer/options.example.yaml
uv run python scripts/smoke_config.py <options.yaml | legacy apps.yaml>
```

Imports `battery_optimizer` and every module in `battery_optimizer_lib`, loads
the file (flat add-on options, or the `battery_optimizer` app of a legacy
apps.yaml) and builds `BatteryOptimizerConfig.from_args()` plus
`AmbientServiceConfig` and `PvForecastServiceConfig`. For a legacy apps.yaml it
also converts it to add-on options and requires the IDENTICAL config. Prints a
redacted summary, lists keys the loader does not read (typos, stale settings)
and supported keys left at their defaults, and exits non-zero on any failure.

## `profile_dp.py`, `clean_learning_data.py`

Offline tools on copies of the persisted JSON files; see their docstrings. The
state files live in `\\<ha>\addon_configs\local_battery_optimizer\`, the
options come from `deploy_addon.py export-options`.
