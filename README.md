# Battery Optimizer for Growatt WIT Inverter

An AppDaemon-based optimizer that uses Nord Pool price forecasts to schedule optimal battery charge/hold/discharge periods.

## Prerequisites

- Home Assistant with AppDaemon add-on installed
- [Nord Pool integration](https://github.com/custom-components/nordpool) (HACS)
- Growatt Modbus integration with VPP support

## Installation

### Step 1: Install AppDaemon Add-on

If not already installed:

1. Go to **Settings > Add-ons > Add-on Store**
2. Search for "AppDaemon"
3. Install "AppDaemon 4"
4. Start the add-on
5. Enable "Start on boot" and "Watchdog"

### Step 2: Copy AppDaemon App Files

Copy the app files to your AppDaemon apps directory:

```bash
# From your Home Assistant config directory
cp appdaemon/apps/battery_optimizer.py /config/appdaemon/apps/
cp appdaemon/apps/apps.yaml /config/appdaemon/apps/
```

**Or manually:**
- Copy `appdaemon/apps/battery_optimizer.py` to `/config/appdaemon/apps/`
- Copy contents of `appdaemon/apps/apps.yaml` to your existing `/config/appdaemon/apps/apps.yaml`

### Step 3: Copy Home Assistant Package

Copy the HA package file:

```bash
cp homeassistant/packages/battery_optimizer.yaml /config/packages/
```

### Step 4: Enable Packages in Home Assistant

Add this to your `configuration.yaml` if not already present:

```yaml
homeassistant:
  packages: !include_dir_named packages
```

Or include the specific file:

```yaml
homeassistant:
  packages:
    battery_optimizer: !include packages/battery_optimizer.yaml
```

### Step 5: Configure the App

Edit `/config/appdaemon/apps/apps.yaml` and update these values:

```yaml
battery_optimizer:
  # Your Nord Pool sensor (check your actual sensor name)
  nordpool_sensor: sensor.nordpool_kwh_lv_eur_3_10

  # Your Growatt sensors
  soc_sensor: sensor.growatt_battery_soc
  pv_power_sensor: sensor.growatt_pv_power

  # IMPORTANT: Set your Growatt device ID
  device_id: "your_device_id_here"

  # Adjust battery parameters to match your system
  battery_capacity_kwh: 14.3
  charge_rate_kw: 4.5
  # ... etc
```

**Finding your device_id:**
1. Go to **Developer Tools > States**
2. Search for your Growatt entities
3. Look for the device_id in the entity attributes

### Step 6: Restart

1. Restart Home Assistant: **Settings > System > Restart**
2. Restart AppDaemon add-on: **Settings > Add-ons > AppDaemon > Restart**

## Usage

### Automatic Mode

The optimizer runs automatically:
- Full optimization at 13:15 daily (after Nord Pool publishes tomorrow's prices)
- Adaptive re-evaluation every 30 minutes
- Safety checks every 5 minutes

### Manual Controls

Use these entities to control the optimizer:

| Entity | Purpose |
|--------|---------|
| `input_boolean.battery_optimizer_enabled` | Enable/disable optimizer |
| `input_boolean.battery_optimizer_override` | Enable manual override |
| `input_select.battery_manual_mode` | Select mode: Auto/Charge/Hold/Discharge |

**Quick Scripts:**
- `script.battery_force_charge` - Force charge mode
- `script.battery_force_discharge` - Force discharge mode
- `script.battery_force_hold` - Force hold mode
- `script.battery_resume_auto` - Return to automatic

### Viewing the Schedule

The schedule is exposed via `sensor.battery_optimizer` with attributes:
- `current_mode` - Current battery mode
- `schedule` - Full schedule array
- `next_charge` - Next scheduled charge time
- `next_discharge` - Next scheduled discharge time

## Rolling TOU Schedule Sync

The optimizer syncs its schedule to the inverter's Time-of-Use (TOU) registers, allowing the inverter to operate autonomously even if Home Assistant goes offline.

### The Problem

TOU registers only support time-of-day values (00:00-23:59), not specific dates. After midnight, "afternoon hours" in the TOU would still reflect yesterday's schedule until the next sync.

### The Solution: Rolling Boundary

Every 30 minutes, the optimizer updates the TOU schedule using a **rolling boundary** — the start time of the currently active TOU period:

- **Hours before the boundary** (already passed today) → use **tomorrow's** schedule
- **Hours from the boundary onward** (still to come) → use **today's** schedule

**Example:** If the current TOU period is DISCHARGE 14:00-18:59, and it's now 16:00 Tuesday:
```
00:00-13:59 → Wednesday's schedule (these hours next execute tomorrow)
14:00-23:59 → Tuesday's schedule (these hours execute today)
```

The boundary stays at 14:00 until the period ends at 19:00, then jumps to the next period's start time. This avoids unnecessary writes mid-period.

**Progression through the day:**
```
Period 00:00-05:59: boundary=00:00 → all hours use today's schedule
Period 06:00-13:59: boundary=06:00 → 00:00-05:59=tomorrow, 06:00-23:59=today
Period 14:00-18:59: boundary=14:00 → 00:00-13:59=tomorrow, 14:00-23:59=today
Period 19:00-23:59: boundary=19:00 → 00:00-18:59=tomorrow, 19:00-23:59=today
```

### Offline Resilience

If Home Assistant goes offline at 18:00, the inverter already has:
- Valid schedule for 18:00-23:59 (today's remaining hours)
- Valid schedule for 00:00-17:59 (tomorrow's early hours)

The inverter can operate correctly through midnight and into the next morning without any intervention.

### Efficient Updates

The optimizer only writes to the inverter when actual behavior would change. It compares the proposed schedule with the current TOU minute-by-minute and skips writes if they're identical, avoiding unnecessary 20-second write cycles.

## File Structure

```
battery-optimizer/
├── appdaemon/
│   └── apps/
│       ├── battery_optimizer.py   # Main optimizer app
│       └── apps.yaml              # AppDaemon configuration
├── homeassistant/
│   └── packages/
│       └── battery_optimizer.yaml # HA entities, automations, scripts
└── README.md
```

## Troubleshooting

### Check AppDaemon Logs

```
Settings > Add-ons > AppDaemon > Log
```

### Verify Sensors Exist

Check these sensors are available in Developer Tools > States:
- Your Nord Pool sensor
- `sensor.growatt_battery_soc`
- `sensor.growatt_pv_power`

### Test Without Device Control

Leave `device_id` empty to run in "dry run" mode - the optimizer will log decisions without sending commands to the inverter.

### TOU Period Write Failures

If you see errors like "Failed to write to registers starting at 304XX" or "Illegal data value":

**Root Cause:** The Growatt firmware validates TOU period writes against ALL 20 period registers, not just the active ones. Stale data from previous schedules can cause "overlap validation" failures.

**Solution:** The optimizer now clears ALL 20 period registers before writing a new schedule. This fix ensures clean writes even when the number of periods changes between schedule updates.

**If problems persist:**
1. Check AppDaemon logs for specific error messages
2. The optimizer uses exponential backoff retries (0.7s, 1.05s, 1.58s delays)
3. Verify the Growatt Modbus integration isn't showing connection errors
4. Concurrent Modbus reads (from the HA coordinator) can occasionally cause bus contention - the retries should handle this

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `battery_capacity_kwh` | 14.3 | Total battery capacity |
| `charge_rate_kw` | 4.5 | Charge power for calculations |
| `discharge_rate_kw` | 4.5 | Discharge power for calculations |
| `min_soc` | 10 | Minimum SOC reserve (%) |
| `max_soc` | 100 | Maximum SOC target (%) |
| `efficiency` | 0.95 | Round-trip efficiency |
| `base_consumption_w` | 500 | Base house consumption |
| `grid_fee_eur_kwh` | 0.05 | Fixed grid fees per kWh |
| `battery_wear_cost_eur_kwh` | 0.00 | Per-kWh wear cost added to discharge cost |
| `pv_threshold_w` | 500 | PV power to trigger solar override |
| `battery_temp_sensor` | `` | Battery temperature sensor for temp-aware charge rate learning (optional) |

## Scheduling Algorithm Highlights

The optimizer uses dynamic programming to find the optimal charge/hold/discharge schedule. Key features:

- **SOC-aware optimization**: Tracks battery state through time to ensure feasible schedules
- **Partial charge support**: Can "top off" the battery when a full charge slot would exceed `max_soc`, utilizing cheap prices fully
- **Load profile learning**: Builds statistical model of household consumption by time-of-day
- **Temperature-aware charge rates**: Learns actual charge rates by SOC and temperature for accurate scheduling
- **Two-pass optimization**: Re-runs with projected battery costs when charging significantly changes the cost basis
- **Survival-first charging**: Guarantees minimum charge slots to avoid hitting `min_soc`

See [docs/scheduling-algorithm.md](docs/scheduling-algorithm.md) for full algorithm documentation.
