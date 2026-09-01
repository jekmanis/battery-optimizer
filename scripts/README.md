# scripts/

Deployment tooling for the AppDaemon share. Both scripts are read-only until
you drop `-DryRun`.

## `deploy.ps1`

Automates the manual procedure in CLAUDE.md § "Deployment to the HA machine".
Windows PowerShell 5.1 compatible (no `&&`/`||`, no ternary, no `??`).

```powershell
# rehearsal: every check runs, nothing is written to the share
.\scripts\deploy.ps1 -DryRun

# real deploy, pausing twice so you stop/start the add-on in the HA UI
.\scripts\deploy.ps1

# real deploy, unattended (stop/start via HA's Supervisor proxy)
.\scripts\deploy.ps1 -HaToken $env:HA_TOKEN -NoPause

# roll back
.\scripts\deploy.ps1 -Restore '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps\backup-20260902-015714'
```

Steps, in order:

1. **git** — refuses a dirty working tree without `-AllowDirty`, prints the
   branch and commit that is about to be deployed.
2. **share** — `Test-Path` on the UNC path and on the LIVE `apps.yaml`.
3. **tests** — `uv run pytest tests/ -q` (skip with `-SkipTests`).
4. **compile** — `uv run python -m py_compile` on the orchestrator and every
   library module.
5. **smoke test** — copies the LIVE `apps.yaml` to `$env:TEMP`, runs
   `smoke_config.py` on the copy, deletes the copy. The unit suite does not
   cover `battery_optimizer.py`, so this is the only check that the deployed
   config still loads.
6. **backup** — `<share>\backup-<yyyyMMdd-HHmmss>\` with the current
   `battery_optimizer.py`, `battery_optimizer_lib\` and a `deployed-commit.txt`
   naming the commit/branch/user. Keeps the `-KeepBackups` newest (default 5)
   and only ever prunes directories matching `backup-<8 digits>-<6 digits>`.
7. **stop** — `POST <HaUrl>/api/hassio/addons/<slug>/stop` with the HA
   long-lived token as Bearer (HA proxies it to the Supervisor), then polls
   `GET .../info` until `state = stopped`, 60 s timeout. Without `-HaToken` the
   script prints what to do in the HA UI and waits for Enter.
8. **copy** — `battery_optimizer.py` plus every `*.py` under
   `battery_optimizer_lib\` (never `__pycache__`, `.pyc` or tests), then
   deletes `.py` files on the share that no longer exist in the repo, so a
   removed module cannot be imported by stale code.
9. **pycache** — removes every `__pycache__` directory under the share's apps
   directory.
10. **verify** — SHA256 of each deployed file against its repo original; any
    mismatch is listed and the script fails.
11. **start** — starts the add-on (API or pause), then reminds you to watch the
    AppDaemon log for `ModuleNotFoundError` / `TypeError` in the first minute.

If anything fails between the stop and a successful verify, the script prints
the backup directory and the exact `-Restore` command to undo the deploy.

`apps.yaml` on the share is **never written** — it holds the HA long-lived
token. It is only read, and the temp copy made for the smoke test is deleted in
a `finally` block (the script warns loudly if the delete fails).

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `-Share` | `\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps` | AppDaemon apps directory |
| `-DryRun` | off | run every check, print the plan, write nothing |
| `-SkipTests` | off | skip pytest (compile + smoke test still run) |
| `-AllowDirty` | off | allow deploying an uncommitted tree |
| `-HaToken` | *empty* | HA long-lived token; enables API stop/start |
| `-HaUrl` | `http://192.168.33.167:8123` | HA base URL (the Supervisor proxy is under `/api/hassio`) |
| `-AddonSlug` | `a0d7b954_appdaemon` | add-on slug |
| `-NoPause` | off | never prompt; requires `-HaToken` |
| `-Restore` | *empty* | path of a `backup-<ts>` directory to roll back to |
| `-KeepBackups` | `5` | how many `backup-*` directories to keep |

## `smoke_config.py`

Standalone; `deploy.ps1` calls it, but it is useful on its own:

```powershell
uv run python scripts/smoke_config.py appdaemon/apps/apps.yaml.example
```

It imports `tests/conftest.py` to install the same
`appdaemon.plugins.hass.hassapi` mock the unit suite uses, imports
`battery_optimizer` and every module in `battery_optimizer_lib`, then loads the
given YAML, finds the app whose `module` is `battery_optimizer`, and builds
`BatteryOptimizerConfig.from_args()` plus `AmbientServiceConfig` and
`PvForecastServiceConfig`. It prints a redacted summary (any key containing
`token`/`key`/`password`/`secret` is shown as `<set>`/`<empty>`), lists config
keys present in the YAML that this version of the loader does not read (typos,
stale settings) and supported keys absent from it (defaults apply), and exits
non-zero on any failure.

Requires PyYAML, declared in the project's `dev` extra
(`uv pip install pyyaml` if your environment predates it).
