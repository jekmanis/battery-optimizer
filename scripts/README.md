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

# one-off: evacuate legacy backup-* directories out of apps\ first
.\scripts\deploy.ps1 -MoveStrayBackups -HaToken $env:HA_TOKEN -NoPause

# roll back
.\scripts\deploy.ps1 -Restore '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\backups\battery_optimizer\backup-20260902-015714'
```

### Nothing but the app may hold `*.py` under `apps\`

AppDaemon 4.5 discovers apps with `app_dir.rglob("*.py")` and calls
`sys.path.insert(0, <dir>)` for **every** directory below the apps directory
that contains `.py` files and has no `__init__.py`. Such a directory therefore
sits at the **front** of `sys.path` and wins every `import battery_optimizer` /
`import battery_optimizer_lib`.

That is not theoretical. Until 2026-09-02 this script wrote its backup to
`apps\backup-<ts>\`. After the 01:59 deploy the add-on imported
`apps\backup-20260902-015911\` — the *previous* commit — while the SHA256
verification of `apps\` passed, because the files in `apps\` really were
correct; the wrong ones were simply imported from the sibling directory. The
only symptoms were an old log wording (`Failed to apply mode — will retry next
slot`) and `sensor.battery_inverter_control_health` missing the attributes the
new commit had added.

Consequences for this script:

* backups go **outside** the apps directory, to
  `<share-root>\backups\battery_optimizer\backup-<ts>\` (`-BackupRoot`), and
  `-BackupRoot` pointing back inside `apps\` is refused;
* a pre-flight scan (step 3, runs in `-DryRun` too) aborts the deploy on any
  `.py` under `apps\` that is not `battery_optimizer.py`, not inside
  `battery_optimizer_lib\` and not in `-AllowedApps` (default `hello.py`), and
  on any other directory below `apps\` that contains `.py` files;
* `-MoveStrayBackups` **moves** `backup-*` directories found inside `apps\` to
  the backup root instead of aborting. They are never deleted;
* `-Restore` accepts a backup in either location and, after copying it back,
  evacuates any `backup-*` directory still sitting inside `apps\`.

Harmless neighbours that the scan deliberately tolerates: `apps.yaml`,
`apps.yaml.bak-*` (not `.py`), `__pycache__\` (removed anyway),
`battery_optimizer_lib\` (a real package — it has `__init__.py`, so AppDaemon
does not prepend it to `sys.path`), and `hello.py` (the stock sample app).
`<share-root>\backup-2026-07-28-pre-fixes\` is outside `apps\` and is fine.

Steps, in order:

1. **git** — refuses a dirty working tree without `-AllowDirty`, prints the
   branch and commit that is about to be deployed.
2. **share** — `Test-Path` on the UNC path and on the LIVE `apps.yaml`.
3. **strays** — the shadowing scan described above. Runs in `-DryRun` too: a
   backup directory under `apps\` makes the whole deploy a lie, so the
   rehearsal has to surface it as loudly as the real run.
4. **tests** — `uv run pytest tests/ -q` (skip with `-SkipTests`).
5. **compile** — `uv run python -m py_compile` on the orchestrator and every
   library module.
6. **smoke test** — copies the LIVE `apps.yaml` to `$env:TEMP`, runs
   `smoke_config.py` on the copy, deletes the copy. The unit suite does not
   cover `battery_optimizer.py`, so this is the only check that the deployed
   config still loads.
7. **backup** — `<share-root>\backups\battery_optimizer\backup-<yyyyMMdd-HHmmss>\`
   with the current `battery_optimizer.py`, `battery_optimizer_lib\` and a
   `deployed-commit.txt` naming the commit/branch/user. Keeps the
   `-KeepBackups` newest (default 5) and only ever prunes directories matching
   `backup-<8 digits>-<6 digits>` **in the backup root** — the live `apps\` is
   never enumerated for pruning.
8. **stop** — `POST <HaUrl>/api/hassio/addons/<slug>/stop` with the HA
   long-lived token as Bearer (HA proxies it to the Supervisor), then polls
   `GET .../info` until `state = stopped`, 60 s timeout. Without `-HaToken` the
   script prints what to do in the HA UI and waits for Enter.
9. **copy** — `battery_optimizer.py` plus every `*.py` under
   `battery_optimizer_lib\` (never `__pycache__`, `.pyc` or tests), then
   deletes `.py` files on the share that no longer exist in the repo, so a
   removed module cannot be imported by stale code.
10. **pycache** — removes every `__pycache__` directory under the share's apps
    directory.
11. **verify** — SHA256 of each deployed file against its repo original; any
    mismatch is listed and the script fails.
12. **stamp** — sets `LastWriteTime = now` on every copied file. `Copy-Item`
    preserves the *source* mtime, so without this the share shows the git
    checkout's timestamps and "when was this deployed?" is unanswerable from
    `ls -l`; AppDaemon's own mtime-based change detection is likewise fed a
    time in the past. Stamping happens **after** the hash check, so it can
    never mask a bad copy. A `<backup-root>\last-deploy.txt` records the
    commit, branch, time, user and the backup directory of the latest deploy.
13. **start** — starts the add-on (API or pause).
14. **post-deploy check** (best effort, `-SkipPostCheck` to disable) — with
    `-HaToken`, reads `GET /api/hassio/addons/<slug>/logs?lines=400` back for
    up to `-PostCheckTimeoutSeconds` (default 90) and reports:
    * `Initializing Battery Optimizer` appearing **after** the restart (the log
      is diffed against an anchor line captured just before the start, so no
      timestamp parsing is involved);
    * `ModuleNotFoundError` / `ImportError` / `Traceback` / `SyntaxError` in the
      new lines;
    * AppDaemon's `Starting apps with N worker threads` line, warning if
      `N < 2` — this app makes blocking `set_wit_mode` calls from callbacks and
      needs `appdaemon: total_threads: 4` in `appdaemon.yaml`.

    It never fails the deploy: the HA API may be unreachable from the deploying
    machine, and a warning is the right outcome there.

If anything fails between the stop and a successful verify, the script prints
the backup directory and the exact `-Restore` command to undo the deploy.

**SHA256 proves the bytes on the share; it does not prove AppDaemon imported
them.** Always confirm the running code is the new one — the post-deploy check
does the log half, and a version marker visible in HA (e.g. an attribute that
only the new commit adds to `sensor.battery_inverter_control_health`) does the
rest.

`apps.yaml` on the share is **never written** — it holds the HA long-lived
token. It is only read, and the temp copy made for the smoke test is deleted in
a `finally` block (the script warns loudly if the delete fails).

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `-Share` | `\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps` | AppDaemon apps directory |
| `-BackupRoot` | `<parent of -Share>\backups\battery_optimizer` | where `backup-<ts>` directories go; refused if inside `-Share` |
| `-DryRun` | off | run every check, print the plan, write nothing |
| `-SkipTests` | off | skip pytest (compile + smoke test still run) |
| `-AllowDirty` | off | allow deploying an uncommitted tree |
| `-AllowedApps` | `hello.py` | other AppDaemon apps allowed to sit directly in `apps\` |
| `-MoveStrayBackups` | off | move `backup-*` directories out of `apps\` instead of aborting |
| `-HaToken` | *empty* | HA long-lived token; enables API stop/start and the post-deploy log check |
| `-HaUrl` | `http://192.168.33.167:8123` | HA base URL (the Supervisor proxy is under `/api/hassio`) |
| `-AddonSlug` | `a0d7b954_appdaemon` | add-on slug |
| `-NoPause` | off | never prompt; requires `-HaToken` |
| `-Restore` | *empty* | path (or bare name) of a `backup-<ts>` directory to roll back to; old in-`apps\` paths accepted |
| `-KeepBackups` | `5` | how many `backup-*` directories to keep in the backup root |
| `-SkipPostCheck` | off | do not read the add-on log back after the start |
| `-PostCheckTimeoutSeconds` | `90` | how long to poll the add-on log |

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
