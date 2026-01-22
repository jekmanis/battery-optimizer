# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Battery Optimizer for Growatt WIT Inverter - a Home Assistant AppDaemon application that uses Nord Pool electricity price forecasts to optimize battery charging/discharging schedules.

## Architecture

### File Structure
- `appdaemon/apps/battery_optimizer.py` - Main optimization engine (~1,200 lines)
- `appdaemon/apps/apps.yaml` - AppDaemon configuration with all parameters
- `homeassistant/packages/battery_optimizer.yaml` - HA entities, automations, sensors

### Key Code Sections in battery_optimizer.py

| Lines | Section | Purpose |
|-------|---------|---------|
| 158-530 | Price Fetching | Nord Pool API/sensor data retrieval |
| 535-700 | Optimization Algorithm | SOC-aware scheduling with timeline simulation |
| 720-920 | Schedule Execution | Timing, adaptive optimization, recalculation |
| 940-980 | Device Control | Growatt Modbus register writes |
| 985-1025 | Manual Override | User intervention handling |
| 1030-1150 | Battery Cost Tracking | Weighted average cost with persistence |
| 1155-1220 | Helper Methods | SOC reading, dynamic config from HA entities |

### Data Models
- `BatteryMode` enum: HOLD (0), CHARGE (1), DISCHARGE (2)
- `PricePoint` dataclass: Hour + price
- `ScheduleEntry` dataclass: Hour + mode + reason

## Core Algorithm

### Scheduling Logic (`find_optimal_schedule`)
1. Select N cheapest hours for CHARGE (N = hours needed to reach max_soc)
2. Calculate discharge threshold: `blended_cost / efficiency + grid_fee`
3. Find candidate DISCHARGE hours above threshold
4. **SOC Timeline Simulation**: Walk through hours chronologically, only allow discharge when `simulated_soc - drain >= min_soc`
5. Remaining hours are HOLD

This prevents scheduling discharge before charge has raised SOC sufficiently.

### Battery Cost Tracking
- **Weighted average**: `(old_energy × old_cost + new_energy × charge_price) / total_energy`
- **SOC-based tracking**: Measures actual SOC changes, not theoretical charging
- **Persistence**: Stored in `input_number.battery_avg_cost` (survives restarts)
- **Blended threshold**: Combines existing battery cost with planned charge cost

### Discharge Rate Assumption
Uses `base_consumption` (default 500W) for discharge calculations, not max inverter rate. At 500W average house load, a 14.3 kWh battery lasts ~25 hours, not 3 hours.

### Dynamic Configuration
These values read from HA entities at runtime (adjustable without restart):
- `input_number.battery_min_soc` → min_soc
- `input_number.battery_max_soc` → max_soc
- `input_number.battery_pv_threshold` → pv_threshold
- `input_number.battery_avg_cost` → battery cost persistence

### Scheduled Tasks
- **13:15 daily**: Full optimization (after Nord Pool prices publish)
- **Startup**: Initial optimization
- **Every 30 min**: Adaptive re-evaluation + battery cost tracking
- **Every 5 min**: Safety checks
- **Hourly**: Mode execution + battery cost update

## Development

This is a Python AppDaemon project - no build/test/lint commands exist.

### Deployment
```bash
# Copy app to AppDaemon
cp appdaemon/apps/battery_optimizer.py /config/appdaemon/apps/

# Copy configuration
cp appdaemon/apps/apps.yaml /config/appdaemon/apps/

# Copy HA package
cp homeassistant/packages/battery_optimizer.yaml /config/packages/
```

### Testing
Manual verification via Home Assistant UI and AppDaemon logs. Set `device_id: ""` in apps.yaml for dry-run mode (simulates without controlling inverter).

### Key Sensor Attributes
`sensor.battery_optimizer` exposes:
- `current_mode`, `schedule`, `next_charge`, `next_discharge`
- `battery_avg_cost` - weighted average cost of energy in battery
- `discharge_threshold` - current price threshold for discharge decisions

## Key Dependencies
- AppDaemon 4
- Home Assistant
- Nord Pool integration (built-in or HACS)
- Growatt Modbus integration
