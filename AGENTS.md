# Repository Guidelines

## Project Structure & Module Organization
- `appdaemon/apps/battery_optimizer.py` holds the main AppDaemon optimizer logic.
- `appdaemon/apps/battery_optimizer_lib/` houses helper modules (learning, load profile, price service, direct inverter control, models).
- `appdaemon/apps/apps.yaml` contains AppDaemon configuration and entity wiring.
- `homeassistant/packages/battery_optimizer.yaml` defines Home Assistant entities, automations, and scripts.
- `docs/` stores design notes (for example, `docs/scheduling-algorithm.md`).
- `tests/` contains pytest coverage for scheduling logic and helpers.
- `README.md` documents setup and usage; `CLAUDE.md` captures architecture notes for agents.

## Build, Test, and Development Commands
- `uv run python -m py_compile appdaemon/apps/battery_optimizer.py` - quick syntax check for the main app.
- `uv run python script.py` - run ad-hoc scripts in the repo's Python environment.
- `uv run pytest tests/ -v` - run the test suite.
- `cp appdaemon/apps/battery_optimizer.py /config/appdaemon/apps/` - deploy the app to Home Assistant.
- `cp homeassistant/packages/battery_optimizer.yaml /config/packages/` - deploy the HA package.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and snake_case naming (`battery_optimizer.py`, `calculate_schedule`).
- YAML files use 2-space indentation; keep Home Assistant entity names consistent with existing patterns (e.g., `input_boolean.battery_optimizer_enabled`).
- Prefer small, well-named helper methods over long inline blocks in `battery_optimizer.py`.
- No formatter or linter is enforced in this repo; keep changes tidy and readable.

## Testing Guidelines
- Run unit tests via `uv run pytest tests/ -v` when touching scheduling, inverter control, or learning logic.
- Validate changes by running Home Assistant in "dry-run" mode (`device_id: ""` in `apps.yaml`) and reviewing AppDaemon logs.
- Use the `sensor.battery_optimizer` attributes to confirm schedule outputs and mode transitions.

## Commit & Pull Request Guidelines
- Commit messages in history are short and descriptive; follow that pattern (e.g., "Schedule optimizations").
- PRs should include: a concise description, the motivation or linked issue, and a brief testing note (what you validated in HA/AppDaemon).
- Include screenshots only if UI entities or dashboards are changed.

## Architecture & Ops Notes
- Core dependencies: AppDaemon 4, Home Assistant, Nord Pool integration, Growatt Modbus integration.
- Runtime cadence: full optimization at 13:15 daily, adaptive refresh every 30 minutes, safety checks every 5 minutes, and hourly mode application.
- Dynamic config is read from HA `input_number.*` entities; key outputs surface on `sensor.battery_optimizer`.
- Inverter control goes through the `growatt_modbus/set_wit_mode` HA service (`battery_optimizer_lib/direct_control.py`); no raw register writes.

## Configuration & Safety Notes
- This project controls a real Growatt inverter; keep device safety in mind and test in dry-run first.
- Document any new config keys in `appdaemon/apps/apps.yaml` and `README.md`.
