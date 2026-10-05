# Open questions (add-on migration)

## Q1 - S6 cutover: approve? (asked 2026-10-05 ~11:40, Ask-first)

The cutover does three Ask-first things at once: it stops the AppDaemon add-on,
gives the add-on the real `device_id` (it then commands the inverter), and lets
it write the shared helpers (`input_number.battery_avg_cost`,
`input_number.battery_cost_basis_version`, `input_boolean.battery_optimizer_override`
on manual "Auto").

Precondition: S5 passes (`scripts/compare_shadow.py` after the 14:15 run, log
clean, 8+ slots).

Exact steps (from the worktree):

```
uv run python scripts/deploy_addon.py stop                              # shadow add-on
uv run python scripts/deploy_addon.py stop --slug a0d7b954_appdaemon    # stays installed
uv run python scripts/deploy_addon.py seed                              # fresh JSON state
# live options from a temp copy of the live apps.yaml (copy deleted after):
uv run python scripts/addon_options.py <copy of apps.yaml> --live --out <tmp>/live.json
uv run python scripts/deploy_addon.py options <tmp>/live.json --no-restart
uv run python scripts/deploy_addon.py start
```

Verify: `sensor.battery_optimizer.app_version == 2026-10-05.1`; two consecutive
slot commands verified by registers; `sensor.battery_inverter_control_health`
`persistent_mismatch_count == 0`; log clean.

Rollback (one command each, nothing to copy):

```
uv run python scripts/deploy_addon.py stop
uv run python scripts/deploy_addon.py start --slug a0d7b954_appdaemon
```

Answer (2026-10-05, in session): "Yes, once S5 passes" - cut over only after S5 passes.
