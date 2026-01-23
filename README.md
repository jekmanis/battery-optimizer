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
