"""
TOU (Time-of-Use) sync module for Growatt inverter control.

Handles:
- Converting battery optimizer schedules to TOU periods
- Syncing schedules to inverter registers via Modbus
- Direct VPP mode control for CHARGE/DISCHARGE/HOLD
- Reading current TOU configuration from inverter
"""

import datetime
from typing import Callable, Dict, List, Optional, Any

from .models import BatteryMode, ScheduleEntry, TouPeriod

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# VPP Register addresses
VPP_CONTROL_AUTHORITY = 30100
VPP_REMOTE_POWER_ENABLE = 30407
VPP_REMOTE_POWER_PERCENT = 30409
VPP_AC_CHARGE_ENABLE = 30410
VPP_TOU_NUM_PERIODS = 30411
VPP_TOU_PERIOD1_BASE = 30412
VPP_DEFAULT_MODE = 30476
MAX_TOU_PERIODS = 20


class TouSyncManager:
    """
    Manager for syncing battery schedules to Growatt inverter TOU registers.

    Uses the Growatt Modbus integration to write TOU period registers.
    This allows the inverter to operate autonomously even if HA goes offline.
    """

    def __init__(
        self,
        device_id: str,
        slot_minutes: int,
        ha_url: str,
        ha_token: str,
        call_service_func: Callable,
        get_datetime_func: Callable,
        get_timezone_func: Callable,
        sleep_func: Callable,
        create_task_func: Callable,
        log_func: Callable,
        get_schedule_func: Callable[[], Dict[datetime.datetime, ScheduleEntry]] = None,
    ):
        """
        Initialize the TOU sync manager.

        Args:
            device_id: Growatt device ID for Modbus commands
            slot_minutes: Schedule slot size in minutes
            ha_url: Home Assistant URL for REST API calls
            ha_token: Long-lived access token for HA API
            call_service_func: Callback to call HA services
            get_datetime_func: Callback to get current datetime
            get_timezone_func: Callback to get local timezone
            sleep_func: Async sleep function
            create_task_func: Async task creation function
            log_func: Callback for logging
            get_schedule_func: Callback to get latest schedule (ensures fresh data at execution time)
        """
        self.device_id = device_id
        self.slot_minutes = slot_minutes
        self.ha_url = ha_url.rstrip("/") if ha_url else ""
        self.ha_token = ha_token

        self.call_service = call_service_func
        self.datetime = get_datetime_func
        self.get_timezone = get_timezone_func
        self.sleep = sleep_func
        self.create_task = create_task_func
        self.log = log_func
        self.get_schedule = get_schedule_func or (lambda: {})

        # Sync state
        self._tou_sync_in_progress = False
        self._tou_sync_pending = False
        self._tou_sync_pending_kwargs: Optional[Dict[str, Any]] = None

    def schedule_to_tou_periods(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        boundary_minute: int = None
    ) -> List[TouPeriod]:
        """
        Convert a schedule to TOU periods for inverter programming.

        Walks forward from the current time slot, consolidating contiguous
        same-mode slots into periods, collecting up to MAX_TOU_PERIODS (20).
        The result is then sorted by clock time for writing to the inverter.

        Since the system re-syncs every 15 minutes, far-future periods that
        didn't fit will be picked up by later syncs as the window advances.

        All modes are written as TOU periods:
        - CHARGE: +100% (or configured power)
        - DISCHARGE: -100% (or configured power)
        - HOLD: +1% charge (firmware quirk creates TRUE standby)

        CRITICAL: TOU periods CANNOT overlap! End times use XX:59 format.

        Args:
            schedule: Schedule dict mapping datetime to ScheduleEntry
            boundary_minute: Minutes since midnight (0-1439) for rolling day boundary.

        Returns:
            List of TouPeriod objects (max 20 periods), sorted by start time
        """
        if not schedule:
            return []

        local_tz = self.get_timezone()
        now = self.datetime()
        if not isinstance(now, datetime.datetime):
            now = datetime.datetime.now()
        if local_tz is not None:
            if now.tzinfo is not None:
                now = now.astimezone(local_tz)
            else:
                now = now.replace(tzinfo=local_tz)
        today = now.date()
        tomorrow = today + datetime.timedelta(days=1)

        def get_local_dt(hour_dt):
            """Convert to local timezone if needed."""
            if hour_dt.tzinfo is not None and local_tz is not None:
                return hour_dt.astimezone(local_tz)
            return hour_dt

        # Build time-of-day schedule from today and tomorrow
        time_of_day_map = {}

        for hour_dt, entry in schedule.items():
            local_hour = get_local_dt(hour_dt)
            entry_date = local_hour.date()

            # Only include today and tomorrow
            if entry_date not in (today, tomorrow):
                continue

            minutes = local_hour.hour * 60 + local_hour.minute

            # Decide which day's entry to use for this time-of-day
            if boundary_minute is not None:
                prefer_tomorrow = minutes < boundary_minute

                if minutes in time_of_day_map:
                    existing_date = time_of_day_map[minutes][1]
                    if prefer_tomorrow and entry_date == tomorrow:
                        time_of_day_map[minutes] = (entry, entry_date)
                    elif not prefer_tomorrow and entry_date == today:
                        time_of_day_map[minutes] = (entry, entry_date)
                else:
                    time_of_day_map[minutes] = (entry, entry_date)
            else:
                # Default behavior: prefer today's entry for conflicts
                if minutes in time_of_day_map:
                    existing_date = time_of_day_map[minutes][1]
                    if entry_date < existing_date:
                        time_of_day_map[minutes] = (entry, entry_date)
                else:
                    time_of_day_map[minutes] = (entry, entry_date)

        if not time_of_day_map:
            self.log("No schedule entries for today/tomorrow, skipping TOU sync")
            return []

        self.log(f"TOU sync: {len(time_of_day_map)} time slots from schedule "
                 f"(today: {sum(1 for _, d in time_of_day_map.values() if d == today)}, "
                 f"tomorrow: {sum(1 for _, d in time_of_day_map.values() if d == tomorrow)})")

        def get_power_for_mode(mode: BatteryMode) -> int:
            """Get TOU power value for mode. HOLD uses +1% (true standby)."""
            if mode == BatteryMode.CHARGE:
                return 100
            elif mode == BatteryMode.DISCHARGE:
                return -100
            else:  # HOLD
                return 1  # +1% charge = TRUE HOLD (firmware quirk)

        # Sort entries in forward order starting from reference_minute
        reference_minute = boundary_minute if boundary_minute is not None else (now.hour * 60 + now.minute)
        sorted_minutes = sorted(time_of_day_map.keys(),
                                key=lambda m: (m - reference_minute) % 1440)

        def close_period(start, end_slot, power, into):
            """Close a period, splitting across midnight if needed."""
            period_end = end_slot + self.slot_minutes - 1
            if period_end > 1439:
                period_end = 1439

            if start <= period_end:
                # Normal period (no midnight wrap)
                into.append(TouPeriod(start=start, end=period_end, power=power))
            else:
                # Period wraps around midnight — split into two
                into.append(TouPeriod(start=start, end=1439, power=power))
                if len(into) < MAX_TOU_PERIODS:
                    into.append(TouPeriod(start=0, end=period_end, power=power))

        # Walk forward, consolidating contiguous same-mode slots into periods.
        # Stop once we've collected MAX_TOU_PERIODS periods.
        periods = []
        current_period_mode = None
        current_period_start = None
        prev_minutes = None

        for hour_minutes in sorted_minutes:
            entry, _ = time_of_day_map[hour_minutes]
            mode = entry.mode

            # Detect gap: entries are non-contiguous (accounting for day wrap)
            has_gap = (prev_minutes is not None and
                       (hour_minutes - prev_minutes) % 1440 > self.slot_minutes)

            if current_period_mode is None:
                # First entry
                current_period_mode = mode
                current_period_start = hour_minutes
            elif current_period_mode != mode or has_gap:
                # Mode changed or gap — close current period
                close_period(current_period_start, prev_minutes,
                             get_power_for_mode(current_period_mode), periods)
                if len(periods) >= MAX_TOU_PERIODS:
                    break
                current_period_mode = mode
                current_period_start = hour_minutes

            prev_minutes = hour_minutes

        # Close the last open period (if we didn't hit the limit mid-loop)
        if len(periods) < MAX_TOU_PERIODS and current_period_mode is not None:
            close_period(current_period_start, prev_minutes,
                         get_power_for_mode(current_period_mode), periods)

        # Trim to MAX_TOU_PERIODS (close_period may have added a split pair)
        if len(periods) > MAX_TOU_PERIODS:
            periods = periods[:MAX_TOU_PERIODS]

        # Sort by clock time for writing to inverter
        periods.sort(key=lambda p: p.start)

        # Firmware requires period 1 to start at 00:00 — pad with HOLD if needed
        if periods and periods[0].start > 0 and len(periods) < MAX_TOU_PERIODS:
            periods.insert(0, TouPeriod(start=0, end=periods[0].start - 1, power=1))

        self.log(f"TOU periods: {len(periods)} (forward from "
                 f"{reference_minute//60:02d}:{reference_minute%60:02d})")

        return periods

    def read_current_tou_periods(self) -> Optional[List[TouPeriod]]:
        """
        Read current TOU periods from inverter registers.

        Returns:
            List of TouPeriod objects, or None if read fails
        """
        if not self.device_id:
            return None

        try:
            num_periods_data = self._read_modbus_registers(VPP_TOU_NUM_PERIODS, 1)
            if not num_periods_data:
                return None

            num_periods = num_periods_data[0]
            if num_periods == 0:
                return []

            period_regs = self._read_modbus_registers(VPP_TOU_PERIOD1_BASE, num_periods * 3)
            if not period_regs or len(period_regs) < num_periods * 3:
                return None

            periods = []
            for i in range(num_periods):
                base = i * 3
                start = period_regs[base]
                end = period_regs[base + 1]
                power_raw = period_regs[base + 2]
                # Convert unsigned to signed
                power = power_raw if power_raw <= 32767 else power_raw - 65536
                periods.append(TouPeriod(start=start, end=end, power=power))

            return periods

        except Exception as e:
            self.log(f"Failed to read current TOU periods: {e}", level="DEBUG")
            return None

    def schedule_tou_sync(
        self,
        boundary_minute: int = None,
        skip_fit_check: bool = False,
        allow_queue: bool = True,
        reason: str = ""
    ):
        """Schedule a TOU sync, avoiding overlapping register writes.

        Note: Schedule is fetched fresh via get_schedule() at execution time,
        ensuring the latest schedule is always used even if queued.
        """
        if self._tou_sync_in_progress:
            if allow_queue:
                self._tou_sync_pending = True
                # Don't store schedule - it will be fetched fresh at execution time
                self._tou_sync_pending_kwargs = {
                    "boundary_minute": boundary_minute,
                    "skip_fit_check": skip_fit_check,
                    "reason": reason,
                }
                if reason:
                    self.log(f"TOU sync already in progress; queued ({reason})", level="DEBUG")
            else:
                if reason:
                    self.log(f"TOU sync already in progress; skipping ({reason})", level="DEBUG")
            return

        self._tou_sync_in_progress = True
        self._tou_sync_pending = False
        self._tou_sync_pending_kwargs = None
        if reason:
            self.log(f"Scheduling TOU sync ({reason})", level="INFO")
        self.create_task(self._run_tou_sync(
            boundary_minute=boundary_minute,
            skip_fit_check=skip_fit_check
        ))

    async def _run_tou_sync(
        self,
        boundary_minute: int = None,
        skip_fit_check: bool = False
    ):
        """Run TOU sync with cleanup. Fetches fresh schedule at execution time."""
        try:
            # Get fresh schedule at execution time, not when sync was scheduled
            schedule = self.get_schedule()
            await self.sync_schedule_to_inverter(
                schedule=schedule,
                boundary_minute=boundary_minute,
                skip_fit_check=skip_fit_check
            )
        finally:
            self._tou_sync_in_progress = False
            if self._tou_sync_pending:
                pending_kwargs = self._tou_sync_pending_kwargs or {}
                self._tou_sync_pending = False
                self._tou_sync_pending_kwargs = None
                self.schedule_tou_sync(**pending_kwargs)

    def check_and_sync_rolling_tou(self):
        """
        Check if TOU schedule needs rolling update and sync if needed.

        Fetches fresh schedule via get_schedule() to ensure latest data.
        """
        schedule = self.get_schedule()
        if not schedule:
            return
        if self._tou_sync_in_progress:
            self.log("TOU sync already in progress; skipping rolling check", level="DEBUG")
            return

        existing_periods = self.read_current_tou_periods()
        if existing_periods is None:
            self.log("Cannot read current TOU for rolling check", level="DEBUG")
            return

        now = self.datetime()
        local_tz = self.get_timezone()
        if now.tzinfo is not None and local_tz:
            now = now.astimezone(local_tz)
        current_minute = now.hour * 60 + now.minute

        # Find boundary from active period
        boundary_minute = 0
        for period in existing_periods:
            if period.start <= current_minute <= period.end:
                boundary_minute = period.start
                break

        new_periods = self.schedule_to_tou_periods(schedule, boundary_minute=boundary_minute)
        if not new_periods:
            return

        # Bidirectional fit check for behavioral equivalence
        if (self._new_schedule_fits_in_existing(new_periods, existing_periods) and
            self._new_schedule_fits_in_existing(existing_periods, new_periods)):
            return

        self.log(f"Rolling TOU update: boundary={boundary_minute//60:02d}:{boundary_minute%60:02d}, "
                 f"updating {len(new_periods)} periods")
        self.schedule_tou_sync(
            boundary_minute=boundary_minute,
            skip_fit_check=True,
            allow_queue=False,
            reason="rolling_update"
        )

    def _new_schedule_fits_in_existing(
        self,
        new_periods: List[TouPeriod],
        existing_periods: List[TouPeriod]
    ) -> bool:
        """
        Check if new TOU schedule fits inside existing schedule.

        Returns True if new schedule fits (no write needed).
        """
        if not existing_periods:
            return False
        if not new_periods:
            return False

        def build_minute_map(periods: List[TouPeriod]) -> Dict[int, int]:
            minute_map = {}
            for period in periods:
                for minute in range(period.start, period.end + 1):
                    if minute <= 1439:
                        minute_map[minute] = period.power
            return minute_map

        existing_map = build_minute_map(existing_periods)
        new_map = build_minute_map(new_periods)

        for minute, new_power in new_map.items():
            existing_power = existing_map.get(minute)
            if existing_power is None or existing_power != new_power:
                return False

        return True

    async def sync_schedule_to_inverter(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        boundary_minute: int = None,
        skip_fit_check: bool = False
    ) -> bool:
        """
        Sync a schedule to the inverter's TOU registers.

        Returns True if sync succeeded, False otherwise.
        """
        if not self.device_id:
            self.log("No device_id configured, cannot sync TOU schedule", level="WARNING")
            return False

        try:
            periods = self.schedule_to_tou_periods(schedule, boundary_minute=boundary_minute)
            num_periods = len(periods)

            # Check if write needed
            if not skip_fit_check:
                existing_periods = self.read_current_tou_periods()
                if existing_periods is not None and self._new_schedule_fits_in_existing(periods, existing_periods):
                    self.log(f"TOU schedule unchanged or fits inside existing ({num_periods} periods) - skipping write")
                    return True

            self.log(f"Syncing {num_periods} TOU periods to inverter")

            # Step 1: Enable VPP control
            if not await self._write_register_with_retry(VPP_CONTROL_AUTHORITY, 1):
                self.log("Failed to enable VPP control", level="ERROR")
                return False

            # Step 2: Enable AC charging
            await self._write_register_with_retry(VPP_AC_CHARGE_ENABLE, 1)
            await self.sleep(0.3)

            # Step 3: Disable remote control so TOU takes precedence
            await self._write_register_with_retry(VPP_REMOTE_POWER_ENABLE, 0)
            await self.sleep(0.3)

            # Step 4: Set default mode to HOLD
            await self._write_register_with_retry(VPP_DEFAULT_MODE, 0)
            await self.sleep(0.3)

            # Step 5: Clear and zero out period registers
            self.log("Clearing TOU schedule and zeroing period registers...")
            await self._write_register_with_retry(VPP_TOU_NUM_PERIODS, 0, verify=False)
            await self.sleep(0.5)

            clear_values = [0] * (MAX_TOU_PERIODS * 3)
            try:
                cleared = await self._write_registers_with_retry(VPP_TOU_PERIOD1_BASE, clear_values, max_retries=2)
                if cleared:
                    self.log(f"Cleared {MAX_TOU_PERIODS} TOU period registers", level="DEBUG")
                else:
                    self.log("Bulk zeroing of TOU period registers failed (continuing)", level="WARNING")
            except Exception as e:
                self.log(f"Bulk zeroing of TOU period registers failed: {e}", level="WARNING")
            await self.sleep(0.5)

            # Step 6: Write periods sequentially
            write_failures = 0
            for i, period in enumerate(periods):
                base_addr = VPP_TOU_PERIOD1_BASE + (i * 3)
                power_unsigned = period.power if period.power >= 0 else 65536 + period.power

                success = await self._write_registers_with_retry(
                    base_addr, [period.start, period.end, power_unsigned]
                )

                if success:
                    if not await self._write_register_with_retry(VPP_TOU_NUM_PERIODS, i + 1):
                        self.log(f"Failed to set num_periods to {i+1}", level="WARNING")
                        success = False
                    else:
                        self.log(f"TOU Period {i+1}: {period.start//60:02d}:{period.start%60:02d} - "
                                 f"{period.end//60:02d}:{period.end%60:02d}, power={period.power}%")

                if not success:
                    write_failures += 1
                    self.log(f"TOU Period {i+1} write FAILED", level="ERROR")

                await self.sleep(0.5)

            if write_failures > 0:
                self.log(f"TOU sync failed: {write_failures} period(s) failed to write", level="ERROR")
                await self._write_register_with_retry(VPP_TOU_NUM_PERIODS, 0, verify=False)
                return False

            # All writes succeeded and were individually verified
            self.log(f"TOU sync complete: {num_periods} periods written and verified")
            return True

        except Exception as e:
            self.log(f"Error syncing TOU schedule to inverter: {e}", level="ERROR")
            return False

    def set_mode(self, mode: BatteryMode, power_percent: int = 100) -> bool:
        """
        Set the battery mode via VPP protocol registers.

        Mode Mapping:
        - CHARGE: Remote control with positive power
        - DISCHARGE: Remote control with negative power
        - HOLD: TOU with +1% charge (firmware quirk for true standby)

        Returns True if mode was set successfully.
        """
        if not self.device_id:
            self.log(f"No device_id configured, would set mode to {mode.name}", level="WARNING")
            return False

        try:
            # Enable VPP control authority
            self.call_service("growatt_modbus/write_register",
                device_id=self.device_id,
                register=VPP_CONTROL_AUTHORITY,
                value=1
            )

            if mode == BatteryMode.HOLD:
                return self._set_hold_mode()
            elif mode == BatteryMode.CHARGE:
                return self._set_charge_mode(power_percent)
            elif mode == BatteryMode.DISCHARGE:
                return self._set_discharge_mode(power_percent)

            return False

        except Exception as e:
            self.log(f"Error setting battery mode: {e}", level="ERROR")
            return False

    def _set_hold_mode(self) -> bool:
        """Set HOLD mode via TOU +1% workaround."""
        self.log("Setting HOLD mode via TOU +1% workaround (true standby)")

        try:
            self.call_service("growatt_modbus/write_register",
                device_id=self.device_id,
                register=VPP_AC_CHARGE_ENABLE,
                value=1
            )
        except Exception as e:
            self.log(f"AC charge enable (30410) failed: {e}", level="WARNING")

        now = self.datetime()
        local_tz = self.get_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        current_minutes = now.hour * 60 + now.minute

        start_min = max(0, current_minutes - 5)
        end_min = min(1439, current_minutes + 120)

        # Clear and zero periods
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_TOU_NUM_PERIODS,
            value=0
        )
        try:
            self.call_service("growatt_modbus/write_registers",
                device_id=self.device_id,
                register=VPP_TOU_PERIOD1_BASE,
                values=[0] * (MAX_TOU_PERIODS * 3)
            )
        except Exception as e:
            self.log(f"Bulk zeroing of TOU registers failed: {e}", level="WARNING")

        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_TOU_NUM_PERIODS,
            value=1
        )
        self.call_service("growatt_modbus/write_registers",
            device_id=self.device_id,
            register=VPP_TOU_PERIOD1_BASE,
            values=[start_min, end_min, 1]  # +1% = HOLD
        )

        self.log(f"Set battery mode to HOLD via TOU {start_min//60:02d}:{start_min%60:02d}-"
                 f"{end_min//60:02d}:{end_min%60:02d} @ +1%")
        return True

    def _set_charge_mode(self, power_percent: int) -> bool:
        """Set CHARGE mode via remote control."""
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_TOU_NUM_PERIODS,
            value=0
        )
        try:
            self.call_service("growatt_modbus/write_register",
                device_id=self.device_id,
                register=VPP_AC_CHARGE_ENABLE,
                value=1
            )
        except Exception as e:
            self.log(f"AC charge enable (30410) failed: {e}", level="WARNING")

        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_REMOTE_POWER_ENABLE,
            value=1
        )
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_REMOTE_POWER_PERCENT,
            value=power_percent
        )
        self.log(f"Set battery mode to CHARGE at {power_percent}%")
        return True

    def _set_discharge_mode(self, power_percent: int) -> bool:
        """Set DISCHARGE mode via remote control."""
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_TOU_NUM_PERIODS,
            value=0
        )
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_REMOTE_POWER_ENABLE,
            value=1
        )
        power_value = 65536 - power_percent
        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=VPP_REMOTE_POWER_PERCENT,
            value=power_value
        )
        self.log(f"Set battery mode to DISCHARGE at {power_percent}%")
        return True

    def _read_modbus_registers(self, address: int, count: int = 1) -> Optional[List[int]]:
        """Read holding registers via growatt_modbus service."""
        if not self.device_id:
            return None

        # Prefer REST API
        rest_values = self._read_modbus_registers_rest(address, count)
        if rest_values is not None:
            return rest_values

        # Fallback to AppDaemon call_service
        try:
            result = self.call_service(
                "growatt_modbus/get_register_data",
                device_id=self.device_id,
                register_type="holding",
                start_address=address,
                count=count
            )
            if result and isinstance(result, dict) and result.get("success"):
                return result.get("values")
            return None
        except Exception as e:
            self.log(f"Failed to read registers {address}-{address+count-1}: {e}", level="WARNING")
            return None

    def _read_modbus_registers_rest(self, address: int, count: int) -> Optional[List[int]]:
        """Read holding registers via HA REST API."""
        if not REQUESTS_AVAILABLE:
            return None
        if not self.ha_url or not self.ha_token:
            return None

        try:
            url = f"{self.ha_url}/api/services/growatt_modbus/get_register_data?return_response"
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "device_id": self.device_id,
                "register_type": "holding",
                "start_address": address,
                "count": count
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                self.log(f"REST API read returned status {response.status_code}", level="DEBUG")
                return None

            data = response.json()
            if isinstance(data, dict):
                if "service_response" in data:
                    data = data["service_response"]
                elif "response" in data:
                    data = data["response"]

            if isinstance(data, dict) and data.get("success"):
                return data.get("values")
            return None

        except requests.exceptions.RequestException as e:
            self.log(f"REST API read failed: {e}", level="DEBUG")
            return None

    async def _write_register_with_retry(
        self,
        address: int,
        value: int,
        max_retries: int = 3,
        verify: bool = True
    ) -> bool:
        """Write single register with retry logic and optional verification."""
        for attempt in range(max_retries):
            try:
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id, register=address, value=value)
                await self.sleep(0.5)

                if verify:
                    readback = self._read_modbus_registers(address, 1)
                    if readback and len(readback) > 0:
                        if readback[0] == value:
                            return True
                        else:
                            self.log(f"Register {address} verify failed: wrote {value}, read {readback[0]}",
                                    level="WARNING")
                    else:
                        self.log(f"Register {address} verify read failed (attempt {attempt+1})",
                                level="WARNING")
                else:
                    return True

            except Exception as e:
                self.log(f"Register {address} write attempt {attempt+1} failed: {e}", level="WARNING")

            if attempt < max_retries - 1:
                await self.sleep(0.5)

        return False

    async def _write_registers_with_retry(
        self,
        address: int,
        values: List[int],
        max_retries: int = 3
    ) -> bool:
        """Write multiple registers atomically with retry logic and verification."""
        base_delay = 0.7

        for attempt in range(max_retries):
            try:
                self.call_service("growatt_modbus/write_registers",
                    device_id=self.device_id, register=address, values=values)

                delay = base_delay * (1.5 ** attempt)
                await self.sleep(delay)

                readback = self._read_modbus_registers(address, len(values))
                if readback and len(readback) == len(values):
                    if readback == values:
                        return True
                    else:
                        self.log(f"Registers {address}-{address+len(values)-1} verify failed",
                                level="WARNING")
                else:
                    self.log(f"Registers {address} verify read failed (attempt {attempt+1})",
                            level="WARNING")

            except Exception as e:
                error_str = str(e)
                if "Illegal data value" in error_str or "exception 3" in error_str.lower():
                    self.log(f"Registers {address} write rejected by firmware (Modbus exception 3)",
                            level="WARNING")
                else:
                    self.log(f"Registers {address} write attempt {attempt+1} failed: {e}",
                            level="WARNING")

            if attempt < max_retries - 1:
                retry_delay = 1.0 * (2 ** attempt)
                await self.sleep(retry_delay)

        return False
