# Repository Guidelines

## Project Structure & Module Organization
- `appdaemon/apps/battery_optimizer.py` holds the main optimizer logic (the orchestrator). The directory name is historical; it runs as a Home Assistant add-on.
- `appdaemon/apps/battery_optimizer_lib/` houses helper modules (learning, load profile, price service, direct inverter control, models) and the add-on host (`ha_host.py`, `addon_main.py`).
- `addon/battery_optimizer/` is the add-on (manifest + options schema, Dockerfile, run script, `DOCS.md`); `options.example.yaml` is the annotated options template.
- `scripts/` holds the deploy tooling (`deploy_addon.py`, `addon_options.py`, `compare_shadow.py`, `smoke_config.py`, `profile_dp.py`); see `scripts/README.md`.
- `homeassistant/packages/battery_optimizer.yaml` defines Home Assistant entities, automations, and scripts.
- `docs/` stores design notes (for example, `docs/scheduling-algorithm.md`).
- `tests/` contains pytest coverage for scheduling logic and helpers.
- `README.md` documents setup and usage; `CLAUDE.md` captures architecture notes for agents.

## Build, Test, and Development Commands
- `uv run python -m py_compile appdaemon/apps/battery_optimizer.py` - quick syntax check for the main app.
- `uv run python script.py` - run ad-hoc scripts in the repo's Python environment.
- `uv run pytest tests/ -v` - run the test suite.
- `uv run python scripts/deploy_addon.py deploy --dry-run`, then `... deploy` - stage, copy to the `addons` share, rebuild and restart the add-on. See CLAUDE.md, "Deployment to the HA machine".
- `cp homeassistant/packages/battery_optimizer.yaml /config/packages/` - deploy the HA package.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and snake_case naming (`battery_optimizer.py`, `calculate_schedule`).
- YAML files use 2-space indentation; keep Home Assistant entity names consistent with existing patterns (e.g., `input_boolean.battery_optimizer_enabled`).
- Prefer small, well-named helper methods over long inline blocks in `battery_optimizer.py`.
- No formatter or linter is enforced in this repo; keep changes tidy and readable.

## Testing Guidelines
- Run unit tests via `uv run pytest tests/ -v` when touching scheduling, inverter control, or learning logic.
- Validate changes in dry-run (`device_id: ""`) or shadow mode (`shadow_mode: true`) and review the add-on log (`deploy_addon.py logs`).
- Use the `sensor.battery_optimizer` attributes to confirm schedule outputs and mode transitions.

## Commit & Pull Request Guidelines
- Commit messages in history are short and descriptive; follow that pattern (e.g., "Schedule optimizations").
- PRs should include: a concise description, the motivation or linked issue, and a brief testing note (what you validated in HA / the add-on).
- Include screenshots only if UI entities or dashboards are changed.

## Architecture & Ops Notes
- Core dependencies: Home Assistant (Supervisor add-on), Nord Pool integration, Growatt Modbus integration; `websocket-client` + `requests` in the image.
- Runtime cadence: full optimization daily at `tomorrow_prices_hour` + 15 min (14:15 by default) and at startup; schedule execution every slot (15 min); adaptive re-evaluation every `adaptive_recalc_minutes` (15); PV sampling every 60 s; price recovery on a bounded backoff. There is no separate safety-check job and nothing runs hourly.
- Dynamic config is read from HA `input_number.*` entities; key outputs surface on `sensor.battery_optimizer`.
- Inverter control goes through the `growatt_modbus/set_wit_mode` HA service (`battery_optimizer_lib/direct_control.py`); no raw register writes and no Time-of-Use programming. The plan is executed slot by slot; nothing runs autonomously on the inverter if the add-on is down.

## Configuration & Safety Notes
- This project controls a real Growatt inverter; keep device safety in mind and test in dry-run first.
- Document any new config key in `addon/battery_optimizer/options.example.yaml` and the schema in `addon/battery_optimizer/config.yaml` (`tests/test_addon_options.py` fails until both have it); set live values in the add-on's options.
