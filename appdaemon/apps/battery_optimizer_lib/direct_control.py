"""Direct inverter control via growatt_modbus/set_wit_mode service.

Sends mode commands to the WIT inverter via HA service calls
instead of writing TOU registers.

Verify-after-set is a *hook*, not a fixed rule
----------------------------------------------
Whether a sent mode "took" is answered by a pluggable strategy (a callable
returning a :class:`VerificationOutcome`), because the obvious source of truth
is not always one. On the reference installation the integration's Inverter Mode
sensor read ``Passthrough`` for 73 consecutive verifications of commands the
battery demonstrably executed (grid charge lifted SOC 9 -> 21 %, max_export
dumped 0.7 kWh in 4 min, hold held SOC flat for 27 min under load). Every one of
those was a false mismatch, and every false mismatch cost a blocking ~10 s
resend on the AppDaemon callback thread — 36 useless inverter writes in 9 h.

The cause was found by probing the inverter: growatt_modbus computes that sensor
from holding registers 30100, 30200-30201 and 30407-30410, but reads them behind
a ``_failed_optional_holding_addrs`` blacklist that is never cleared. A single
transient read failure on 2026-09-01T03:46:34Z blacklisted all three blocks, and
the sensor has reported its dataclass default ever since. The REGISTERS were
right the whole time.

Hence two strategies:

* :class:`RegisterVerifier` — reads those registers back directly through
  ``growatt_modbus/get_register_data``, bypassing the blacklist. This is the
  RECOMMENDED source (``verify_source: registers``, and what "auto" picks
  whenever a device_id exists).
* :class:`ModeSensorVerifier` — the entity comparison. Correct only where that
  entity is not frozen; it must be chosen explicitly.

Verification is OFF when neither can be built. Physical plausibility (battery
power sign/magnitude, SOC slope) is deliberately NOT a strategy: the probe
measured -39.7 W under a -100 % discharge command at 12 % SOC against a 10 %
cutoff, so a power check would have failed a perfectly good command.

Thread safety
-------------
AppDaemon dispatches this app's callbacks across several worker threads
(``total_threads`` + ``pin_app: false``), so ``apply_mode``, ``release_control``
and the verification timer can all run at once. DirectControl owns two locks:

* ``_io_lock`` (plain Lock) — one ``set_wit_mode`` in flight at a time. Held
  across ``apply_mode_with_outcome`` from the duplicate check through the
  last-sent record, across ``release_control``, and across the resend inside
  ``_run_verification``. Holding it over BOTH the duplicate decision and the
  send is the point: reading "last sent" before a call that can take 15 s and
  writing it after let two concurrent same-mode applies both reach the inverter.
* ``_state_lock`` (RLock) — ``_last_mode_*``, ``_last_params``, ``_verify_timer``,
  ``_verify_generation``, the verifier reference and every counter. Never held
  across a service call, so ``get_diagnostics()`` returns immediately even while
  a 15 s send is in flight.

**LOCK ORDER: app lock -> _io_lock -> _state_lock.** DirectControl must never
call anything that takes the app lock while holding either of its own locks.
The one place that could: ``_verify_mode`` reports its duration through
``app.record_external_callback_duration`` (which the app runs under its lock) —
so that call lives in ``_verify_mode``'s ``finally``, strictly AFTER
``_run_verification`` has returned and released both locks.

A verification's READS deliberately run outside ``_io_lock``. They are read-only
(``get_register_data``), and an apply arriving mid-verification must not queue
behind a diagnostic read; the resend that may follow does take ``_io_lock``.
Timer identity is a generation counter rather than the ``run_in`` handle,
because ``cancel_timer`` on thread A cannot stop a callback already dequeued on
thread B: ``_schedule_verification`` and ``_cancel_verification`` bump
``_verify_generation``, and a callback whose stamp no longer matches returns
without clearing a newer handle and without resending a superseded mode.
"""

import datetime
import enum
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .models import BatteryMode, ScheduleEntry

# Conventional entity id of the integration's "Inverter Mode" sensor
# (GrowattWitModeStatusSensor). Its friendly name is "<entry> Inverter Mode"
# and, because that sensor does NOT use has_entity_name, the entity_id is
# derived directly from that slugified name. The id therefore depends on the
# config entry NAME:
#   entry "Growatt"     -> sensor.growatt_inverter_mode      (this deployment)
#   entry "Growatt WIT" -> sensor.growatt_wit_inverter_mode
#
# This is documentation of the usual id — a value to COPY into
# ``inverter_mode_sensor`` — and NOT an automatic fallback any more. It used to
# be one, which meant every deployment silently verified against an entity that
# may not reflect the active override at all; see the module docstring.
DEFAULT_MODE_STATUS_ENTITY = "sensor.growatt_inverter_mode"

# Expected value of the Inverter Mode sensor for each mode string we send.
# Mirrors _compute_wit_mode_status() in the integration's coordinator.
MODE_STATUS_MAP = {
    "grid_charge": "Grid Charge",
    "hold": "Preserve SOC",
    "preserve_soc": "Preserve SOC",
    "max_export": "Max Export",
    "discharge_to_grid": "Discharge to Grid",
    "discharge_to_load": "Discharge to Load",
    "passthrough": "Passthrough",
}

# The status the mode sensor reports when NO override is active — and also what
# a sensor that does not track the override reports permanently. A MATCH on this
# value is therefore not evidence that the command arrived: it is what we would
# read either way. Only `release_control()` expects it, so only release_control
# is affected; see VerificationOutcome.probative.
NON_PROBATIVE_STATUS = "Passthrough"

# Modes whose expected state is indistinguishable from "no override at all".
# A MATCH on one of these is not evidence that the command landed, whatever the
# strategy: an inverter that never received the release reads the same as one
# that did.
NON_PROBATIVE_MODES = ("passthrough",)

# ---------------------------------------------------------------------------
# WIT VPP control registers (holding), as written by growatt_modbus'
# set_wit_mode (custom_components/growatt_modbus/diagnostic.py:1307-1560) and
# decoded by its coordinator (_compute_wit_mode_status, coordinator.py:525-570).
# These are the ONLY source that reflects the active override on this
# installation: the integration's Inverter Mode sensor is computed from the same
# registers but behind a `_failed_optional_holding_addrs` blacklist that is
# never cleared, so a single transient read failure (2026-09-01T03:46:34Z) froze
# it at its dataclass default "Passthrough" indefinitely. Reading the registers
# directly bypasses the blacklist.
# ---------------------------------------------------------------------------
REG_CONTROL_AUTHORITY = 30100   # 1 = VPP has control (not read: costs a call)
REG_EXPORT_LIMIT_ENABLE = 30200
REG_EXPORT_LIMIT_RATE = 30201
REG_REMOTE_ENABLE = 30407       # 1 = remote power override active
REG_REMOTE_DURATION = 30408     # minutes; does NOT count down -> informational
REG_REMOTE_POWER = 30409        # signed % (u16 two's complement): + charge, - discharge
REG_AC_CHARGE_ENABLE = 30410    # 0 disabled, 1 pv_priority, 2 ac_priority
REG_PRIORITY_MODE = 30476       # 0 Load First, 1 Battery First (not read)

# Two contiguous blocks = exactly two blocking service calls per verification.
REG_BLOCK_CONTROL = (REG_REMOTE_ENABLE, 4)   # 30407, 30408, 30409, 30410
REG_BLOCK_EXPORT = (REG_EXPORT_LIMIT_ENABLE, 2)  # 30200, 30201

# diagnostic.py:209-213 AC_CHARGE_MODE_MAP
AC_CHARGE_MODE_VALUES = {"disabled": 0, "pv_priority": 1, "ac_priority": 2}

# Default delay before verifying a sent mode against the inverter's reported
# status. The status sensor is recomputed on each coordinator poll (~30-60s), so
# 90s gives at least one poll cycle after the write settles. Overridable via
# config.verify_delay_seconds: a lagging sensor and a lost command look
# identical at a fixed delay.
VERIFY_DELAY_SECONDS = 90

# Default delay for the SINGLE re-check performed after a resend. Shorter than
# the first check — the resend goes out immediately, only one poll cycle is
# needed to see it.
VERIFY_RECHECK_SECONDS = 60

# Per-call websocket timeout for set_wit_mode (AppDaemon >= 4.4 HASS kwarg).
# The handler performs 6-9 sequential Modbus writes behind a shared lock and can
# exceed AppDaemon's 10s default. This call is SYNCHRONOUS on the callback
# thread, so every second here blocks every other callback of this app; the
# unconfirmed path is safe because the request is already on the wire when the
# timeout fires (see _call_set_wit_mode). Overridable via
# config.set_wit_mode_timeout_seconds.
SET_WIT_MODE_TIMEOUT_SECONDS = 15

# ``ad_status`` values the AppDaemon HASS plugin stamps onto a service-call
# response. Both mean "we stopped waiting", not "it did not happen": the request
# JSON is written to the websocket BEFORE the plugin starts awaiting the
# matching response future, so a timeout says nothing about whether Home
# Assistant ran the service.
UNCONFIRMED_AD_STATUSES = ("TIMEOUT", "TERMINATING")


def _ad_status_of(response) -> Optional[str]:
    """Upper-cased ``ad_status`` from a service response, top level or nested.

    AppDaemon stamps ``ad_status`` onto the envelope it returns, but WHERE it
    lands depends on the AD version and on whether the service declared a
    response: it can sit at the top level or one level down under ``result``.
    Both service calls in this module must look in both places, and they used
    to disagree — ``_call_set_wit_mode`` checked the nested copy,
    ``RegisterVerifier._read_block`` only the top-level one. A nested TIMEOUT
    therefore fell through the read path into the ordinary payload parsing,
    where a missing ``values`` key made it indistinguishable from a clean read
    that simply returned nothing.

    Returns None when there is no ``ad_status`` string anywhere.
    """
    if not isinstance(response, dict):
        return None
    status = response.get("ad_status")
    if not isinstance(status, str):
        inner = response.get("result")
        status = inner.get("ad_status") if isinstance(inner, dict) else None
    return status.upper() if isinstance(status, str) else None


class ApplyOutcome(enum.Enum):
    """What actually happened to one ``apply_mode`` command.

    The boolean returned by ``apply_mode`` cannot separate these, and THREE of
    them are True: a dry run, a duplicate that was never transmitted, and a
    client-side timeout nobody confirmed. Treating all three as "the inverter
    obeyed" is what let a hung growatt_modbus publish climbing apply_successes
    while the "inverter is NOT following the schedule" escalation could never
    fire — every call timed out at ``hass_timeout``, logged a WARNING and
    returned True.
    """

    SENT = "sent"                        # confirmed by the service response
    UNCONFIRMED_TIMEOUT = "unconfirmed"  # client-side timeout, outcome unknown
    SKIPPED_DUPLICATE = "duplicate"      # identical command, nothing transmitted
    DRY_RUN = "dry_run"                  # device_id == "" — nothing transmitted
    FAILED = "failed"                    # confirmed failure

    @property
    def confirmed(self) -> bool:
        """True only when the inverter actually acknowledged the command."""
        return self is ApplyOutcome.SENT


class VerificationResult(enum.Enum):
    """Verdict of one verification strategy about one sent mode."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


@dataclass
class VerificationOutcome:
    """What a verification strategy observed, in reportable form.

    Attributes:
        result: MATCH / MISMATCH / UNVERIFIABLE.
        source: Human-readable identity of what was consulted, e.g.
            ``mode_sensor:sensor.growatt_inverter_mode``. It goes into every log
            line and into ``get_diagnostics()`` so a reader never has to guess
            which entity disagreed.
        observed: The raw value that was read, as a string (None if nothing
            could be read).
        expected: The value the strategy was looking for.
        probative: False when a MATCH would have occurred anyway. The mode
            sensor reporting ``Passthrough`` for a passthrough command is the
            case that matters: that is also the reading of a sensor that ignores
            overrides entirely, so counting it as "verified" would manufacture
            evidence.
        detail: Free-text reason, mainly for UNVERIFIABLE.
    """

    result: VerificationResult
    source: str
    observed: Optional[str] = None
    expected: Optional[str] = None
    probative: bool = True
    detail: str = ""


# A verification strategy is any callable taking the mode string that was sent
# and the params that were sent with it, and returning a VerificationOutcome.
Verifier = Callable[[str, dict], VerificationOutcome]


class ModeSensorVerifier:
    """Verify a sent mode against the integration's "Inverter Mode" sensor.

    This is ONE strategy, not the definition of verification. It answers "does
    the entity that is supposed to mirror the active override agree?" — which is
    evidence only when that entity actually tracks the override.

    A register read or a physical-plausibility check (SOC slope, battery power
    sign, energy moved over the slot) is a drop-in replacement: build any
    callable ``(mode_str, params) -> VerificationOutcome`` and hand it to
    ``DirectControl(..., verifier=...)`` or :meth:`DirectControl.set_verifier`.
    The bounded two-step ladder in ``_verify_mode`` is strategy-independent.
    """

    def __init__(self, app, entity_id: str):
        self.app = app
        self.entity_id = entity_id

    @property
    def source(self) -> str:
        return f"mode_sensor:{self.entity_id}"

    def __call__(self, mode_str: str, params: dict) -> VerificationOutcome:
        expected = MODE_STATUS_MAP.get(mode_str)
        if expected is None:
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                detail=f"no expected status is mapped for mode '{mode_str}'",
            )

        state = self.app.get_state(self.entity_id)

        if state is None or state in ("unknown", "unavailable"):
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                observed=None if state is None else str(state),
                expected=expected,
                detail=f"{self.entity_id} is {state}",
            )

        observed = str(state)
        if observed == expected:
            return VerificationOutcome(
                result=VerificationResult.MATCH,
                source=self.source,
                observed=observed,
                expected=expected,
                # A 'Passthrough' match is what we'd read from a sensor that
                # tracks nothing, so it proves nothing.
                probative=(expected != NON_PROBATIVE_STATUS),
            )

        return VerificationOutcome(
            result=VerificationResult.MISMATCH,
            source=self.source,
            observed=observed,
            expected=expected,
        )



def decode_signed_power(raw: int) -> int:
    """Two's-complement decode of register 30409 (coordinator.py:533-537).

    The register is an unsigned 16-bit word: 65436 means -100 % (full
    discharge), 100 means +100 % (full charge). Comparing the raw word without
    decoding would call a -100 % discharge a "65436 % charge".
    """
    raw = int(raw)
    return raw - 65536 if raw > 32767 else raw


def decode_wit_mode_status(regs: dict) -> str:
    """Reproduce the integration's mode-status decode from raw registers.

    Verbatim port of ``GrowattCoordinator._compute_wit_mode_status``
    (coordinator.py:525-570), so the log can show the words the Inverter Mode
    sensor *would* be showing if it were not frozen. Reporting only — the
    MATCH/MISMATCH decision compares registers, not this string.
    """
    remote_enable = int(regs.get(REG_REMOTE_ENABLE, 0) or 0)
    power_signed = decode_signed_power(regs.get(REG_REMOTE_POWER, 0) or 0)
    export_enable = int(regs.get(REG_EXPORT_LIMIT_ENABLE, 0) or 0)
    export_rate = int(regs.get(REG_EXPORT_LIMIT_RATE, 0) or 0)

    if not remote_enable:
        return "Passthrough"
    if power_signed > 1:
        return "Grid Charge"
    if power_signed in (0, 1):
        return "Preserve SOC"
    if power_signed < 0:
        export_allowed = (not export_enable) or (export_rate > 0)
        if export_allowed and abs(power_signed) == 100:
            return "Max Export"
        if export_allowed:
            return "Discharge to Grid"
        return "Discharge to Load"
    return "Passthrough"


def expected_registers(mode_str: str, params: dict):
    """Registers ``set_wit_mode`` will have written for this exact command.

    Derived from the params WE SENT rather than from a static per-mode table,
    because the same mode writes different registers depending on export_rate:
    ``discharge_to_grid`` with ``export_rate=40`` leaves 30200=1 / 30201=40,
    while ``export_rate=100`` leaves 30200=0. A fixed table would have to guess,
    and a wrong guess here is a false mismatch — the exact failure this whole
    change exists to stop.

    Mirrors diagnostic.py:1307-1560 step for step. Each value is a TUPLE of
    acceptable values, because one register has a documented firmware fallback:
    30410=2 (ac_priority) is retried as 1 when the inverter rejects it
    (diagnostic.py:1400-1405), so both are correct.

    Returns None for a mode string the service does not implement (-> the
    verification is UNVERIFIABLE, never a MISMATCH).
    """
    if mode_str not in MODE_STATUS_MAP:
        return None

    power = int(params.get("power_percent", 100) or 100)
    export_rate = params.get("export_rate")
    ac_mode = params.get("ac_charge_mode")

    exp = {}

    # --- Step 2: AC charge mode (30410) ---
    if ac_mode is not None:
        ac_val = AC_CHARGE_MODE_VALUES.get(ac_mode)
        if ac_val == 2:
            # Documented fallback: ac_priority downgrades to pv_priority when
            # the firmware rejects 2.
            exp[REG_AC_CHARGE_ENABLE] = (2, 1)
        elif ac_val is not None:
            exp[REG_AC_CHARGE_ENABLE] = (ac_val,)
    elif mode_str == "grid_charge":
        exp[REG_AC_CHARGE_ENABLE] = (1,)
    else:
        exp[REG_AC_CHARGE_ENABLE] = (0,)

    # --- Step 4: export limit (30200 / 30201) ---
    if export_rate is not None:
        if int(export_rate) >= 100:
            exp[REG_EXPORT_LIMIT_ENABLE] = (0,)
        else:
            exp[REG_EXPORT_LIMIT_ENABLE] = (1,)
            exp[REG_EXPORT_LIMIT_RATE] = (int(export_rate),)
    elif mode_str == "discharge_to_load":
        # Zero export is CRITICAL here: without it the battery dumps to grid.
        exp[REG_EXPORT_LIMIT_ENABLE] = (1,)
        exp[REG_EXPORT_LIMIT_RATE] = (0,)
    else:
        exp[REG_EXPORT_LIMIT_ENABLE] = (0,)
    # NOTE: 30201 is only compared when 30200 is expected to be 1. When the
    # limiter is off the service never writes the rate, so whatever stale value
    # sits there is not evidence of anything.

    # --- Step 5: battery power command (30407 / 30409) ---
    if mode_str == "passthrough":
        exp[REG_REMOTE_ENABLE] = (0,)
        exp[REG_REMOTE_POWER] = (0,)
    elif mode_str in ("hold", "preserve_soc"):
        # +1 %, not 0: 30409=0 clips PV export and 30407=0 still discharges
        # under Load First. A 1 % charge command is what actually idles the pack.
        exp[REG_REMOTE_ENABLE] = (1,)
        exp[REG_REMOTE_POWER] = (1,)
    elif mode_str == "grid_charge":
        exp[REG_REMOTE_ENABLE] = (1,)
        exp[REG_REMOTE_POWER] = (power,)
    else:  # discharge_to_load / discharge_to_grid / max_export
        p = 100 if mode_str == "max_export" else power
        exp[REG_REMOTE_ENABLE] = (1,)
        exp[REG_REMOTE_POWER] = (65536 - p,)

    return exp


class RegisterVerifier:
    """Verify a sent mode by reading the VPP control registers back.

    This is the RECOMMENDED strategy on this installation, because the obvious
    alternative is broken at its source: ``sensor.growatt_inverter_mode`` is
    computed from these very registers, but the integration reads them through a
    ``_failed_optional_holding_addrs`` blacklist that is never cleared. One
    transient failure on 2026-09-01T03:46:34Z blacklisted 30100, 30200-30201 and
    30407-30410, and the sensor has been frozen at its dataclass default
    ("Passthrough") ever since — hence 73/73 false mismatches on 2026-09-02
    while the registers themselves were correct (probe at 09:25, 9 min into a
    discharge_to_load: 30407=1, 30409=65436 = -100 %, 30200=1, 30201=0).

    Two reads per verification, both holding registers:
      30407 count 4 -> remote_enable, duration_min, power_percent(u16), ac_charge
      30200 count 2 -> export_limit_enable, export_limit_rate

    Each read is a SYNCHRONOUS blocking service call on the AppDaemon callback
    thread, exactly like set_wit_mode, so it carries the same ``hass_timeout``
    and the enclosing ``_verify_mode`` reports its wall time through
    ``record_external_callback_duration``. 30100 (control authority) and 30476
    (priority mode) are deliberately NOT read: each would be a third and fourth
    blocking call and neither changes the verdict.

    Anything that is not a clean read — service error, timeout, None, missing or
    short ``values`` list — is UNVERIFIABLE, never a MISMATCH. A source that
    cannot answer must not trigger a resend; that conflation is what turned a
    frozen sensor into 36 useless inverter writes.

    NOT used, on purpose: physical plausibility (battery power sign/magnitude,
    SOC slope). The probe measured only -39.7 W under a -100 % discharge_to_load
    command at 12 % SOC against a 10 % cutoff — a correct command that any
    power-magnitude check would have called a failure.
    """

    def __init__(self, app, device_id: str,
                 timeout: int = SET_WIT_MODE_TIMEOUT_SECONDS):
        self.app = app
        self.device_id = device_id
        self.timeout = int(timeout)
        # Why the last read failed, so an UNVERIFIABLE verdict can SAY whether
        # the read timed out, was refused, or came back malformed. Every caller
        # runs under DirectControl's _io_lock, so a plain attribute is enough.
        self._last_read_error: Optional[str] = None

    @property
    def source(self) -> str:
        return "registers:30407-30410,30200-30201"

    # -- reading ------------------------------------------------------------

    def _read_block(self, start_address: int, count: int):
        """Read one holding-register block. Returns a list, or None on any doubt.

        Service schema (growatt_modbus/services.yaml, diagnostic.py:189-195):
          device_id (required), register_type in {input, holding},
          start_address (required), count (required).
        NOTE the field is ``start_address``, not ``address``.

        ``return_response=True`` is safe to pass and is NOT service data: it is a
        formal parameter of AppDaemon's ``HassPlugin.call_plugin_service``
        (hassplugin.py:727), bound before ``service_data`` is assembled — the
        same mechanism as ``hass_timeout``. It is passed explicitly rather than
        relying on AppDaemon's auto-enable for SupportsResponse.OPTIONAL, which
        only fires once the plugin has learned the service's response metadata.
        """
        try:
            result = self.app.call_service(
                "growatt_modbus/get_register_data",
                device_id=self.device_id,
                register_type="holding",
                start_address=start_address,
                count=count,
                return_response=True,
                hass_timeout=self.timeout,
            )
        except Exception as e:
            self._last_read_error = f"{type(e).__name__}: {e}"
            self.app.log(
                f"DirectControl: register read {start_address}+{count} "
                f"failed: {e}",
                level="DEBUG",
            )
            return None

        if result is None:
            self._last_read_error = "no response from the AppDaemon HASS plugin"
            return None

        # Same envelope handling as _call_set_wit_mode: ad_status can sit at the
        # top level or nested under "result", and either way it means "we
        # stopped waiting", not "the read came back empty".
        ad_status = _ad_status_of(result)
        if ad_status in UNCONFIRMED_AD_STATUSES:
            self._last_read_error = (
                f"AppDaemon ad_status={ad_status} after {self.timeout}s "
                f"(read timeout, not a mismatch)"
            )
            return None

        payload = self._service_payload(result)
        if payload is None:
            self._last_read_error = "no {'values': [...]} payload in the response"
            return None
        if payload.get("success") is False:
            self._last_read_error = str(payload.get("error") or "success=False")
            return None

        values = payload.get("values")
        if not isinstance(values, (list, tuple)) or len(values) < count:
            self._last_read_error = f"short or missing values list: {values!r}"
            return None
        try:
            return [int(v) for v in values[:count]]
        except (TypeError, ValueError):
            self._last_read_error = f"non-integer register values: {values!r}"
            return None

    @staticmethod
    def _service_payload(result, depth: int = 0):
        """Dig the integration's {"success":..., "values":[...]} out.

        AppDaemon hands back Home Assistant's websocket envelope, whose depth
        varies by version: the response dict can sit at the top level, under
        ``result``, or under ``result`` -> ``response``.
        """
        if not isinstance(result, dict) or depth > 4:
            return None
        if "values" in result:
            return result
        for key in ("result", "response"):
            found = RegisterVerifier._service_payload(result.get(key), depth + 1)
            if found is not None:
                return found
        return None

    def _read_failure_detail(self, start_address: int, count: int) -> str:
        """Name the block AND why it could not be read."""
        detail = f"could not read holding {start_address}+{count}"
        if self._last_read_error:
            detail = f"{detail}: {self._last_read_error}"
        return detail

    # -- verdict ------------------------------------------------------------

    def __call__(self, mode_str: str, params: dict) -> VerificationOutcome:
        if not self.device_id:
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                detail="no device_id (dry run)",
            )

        expected = expected_registers(mode_str, params)
        if expected is None:
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                detail=f"no register expectation for mode '{mode_str}'",
            )

        control = self._read_block(*REG_BLOCK_CONTROL)
        if control is None:
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                detail=self._read_failure_detail(*REG_BLOCK_CONTROL),
            )

        export = self._read_block(*REG_BLOCK_EXPORT)
        if export is None:
            return VerificationOutcome(
                result=VerificationResult.UNVERIFIABLE,
                source=self.source,
                detail=self._read_failure_detail(*REG_BLOCK_EXPORT),
            )

        observed_regs = {
            REG_REMOTE_ENABLE: control[0],
            REG_REMOTE_DURATION: control[1],
            REG_REMOTE_POWER: control[2],
            REG_AC_CHARGE_ENABLE: control[3],
            REG_EXPORT_LIMIT_ENABLE: export[0],
            REG_EXPORT_LIMIT_RATE: export[1],
        }

        differences = []
        for reg, accepted in expected.items():
            if (reg == REG_EXPORT_LIMIT_RATE
                    and 1 not in expected.get(REG_EXPORT_LIMIT_ENABLE, (1,))):
                # Limiter off -> the service never wrote the rate, so a stale
                # value there is not evidence.
                continue
            actual = observed_regs.get(reg)
            if actual not in accepted:
                accepted_text = "/".join(str(v) for v in accepted)
                differences.append(f"{reg}={actual} (expected {accepted_text})")

        observed = self._describe(observed_regs)
        expected_text = self._describe_expected(expected)

        if differences:
            return VerificationOutcome(
                result=VerificationResult.MISMATCH,
                source=self.source,
                observed=observed,
                expected=expected_text,
                detail="; ".join(differences),
            )

        return VerificationOutcome(
            result=VerificationResult.MATCH,
            source=self.source,
            observed=observed,
            expected=expected_text,
            # An inverter that never got the release reads exactly like one that
            # did: all zeros.
            probative=(mode_str not in NON_PROBATIVE_MODES),
        )

    @staticmethod
    def _describe(regs: dict) -> str:
        """Raw registers plus the mode string the frozen sensor should show."""
        raw = regs.get(REG_REMOTE_POWER, 0)
        signed = decode_signed_power(raw)
        power = f"{REG_REMOTE_POWER}={raw}"
        if signed != raw:
            power += f"({signed}%)"
        return (
            f"{REG_REMOTE_ENABLE}={regs.get(REG_REMOTE_ENABLE)} "
            f"{REG_REMOTE_DURATION}={regs.get(REG_REMOTE_DURATION)} "
            f"{power} "
            f"{REG_AC_CHARGE_ENABLE}={regs.get(REG_AC_CHARGE_ENABLE)} "
            f"{REG_EXPORT_LIMIT_ENABLE}={regs.get(REG_EXPORT_LIMIT_ENABLE)} "
            f"{REG_EXPORT_LIMIT_RATE}={regs.get(REG_EXPORT_LIMIT_RATE)} "
            f"-> {decode_wit_mode_status(regs)}"
        )

    @staticmethod
    def _describe_expected(expected: dict) -> str:
        parts = []
        for reg, accepted in sorted(expected.items()):
            parts.append(f"{reg}={'/'.join(str(v) for v in accepted)}")
        return " ".join(parts)


class DirectControl:
    """Sends mode commands to WIT inverter via HA service calls."""

    def __init__(self, app, config, verify_enabled: bool = True,
                 verifier: Optional[Verifier] = None):
        """
        Args:
            app: AppDaemon app instance (for call_service, get_state, log,
                 run_in, cancel_timer). If it exposes
                 ``record_external_callback_duration(name, seconds)``, the
                 verification callback's wall time is reported through it.
            config: BatteryOptimizerConfig instance
            verify_enabled: Master switch for verify-after-set. Default True, so
                existing behaviour is unchanged for anyone who has configured a
                verification source. Set False to run the schedule without any
                verify-after-set at all (no timers, no resends), which is the
                right setting when no entity or register on this installation is
                known to reflect the active override.
            verifier: Verification strategy — a callable
                ``(mode_str, params) -> VerificationOutcome``. When omitted, a
                :class:`ModeSensorVerifier` is built from
                ``config.inverter_mode_sensor``; if that is EMPTY, verification
                is disabled. It deliberately no longer falls back to
                ``DEFAULT_MODE_STATUS_ENTITY``: that fallback made every
                deployment verify against an entity that may not track the
                override, and each false mismatch costs a blocking ~10 s resend
                on the AppDaemon callback thread.
        """
        self.app = app
        self.config = config

        # --- locks (see "Thread safety" in the module docstring) -----------
        # ORDER: app lock -> _io_lock -> _state_lock. Never the reverse, and
        # never call into the app's locked surface while holding either.
        # One inverter write at a time; also makes the duplicate decision
        # atomic with the send it is deciding about.
        self._io_lock = threading.Lock()
        # Re-entrant: properties that take it are read from methods that
        # already hold it. Never held across a service call.
        self._state_lock = threading.RLock()

        self._last_mode_sent: Optional[str] = None
        self._last_mode_time: Optional[datetime.datetime] = None
        self._last_params: dict = {}
        # Handle of the pending one-shot verification timer (from run_in), or
        # None. Superseded whenever a new mode is applied.
        self._verify_timer = None
        # Monotonic stamp identifying the CURRENT verification. The run_in
        # handle alone is not enough: cancel_timer from one thread cannot stop
        # a callback another thread has already dequeued, so that orphan would
        # null a newer handle and resend a superseded mode. Every schedule and
        # every cancel bumps this; a callback carrying an older stamp is inert.
        self._verify_generation = 0
        # Whether the "cannot verify — source unreadable" condition has been
        # logged at WARNING yet. First occurrence per app start is a WARNING (a
        # wrong entity id must be visible); the rest are DEBUG.
        self._verify_unreadable_warned = False

        # Last error text the service call reported, for the failure logs.
        self._last_service_error: Optional[str] = None
        # Source string the strategy last reported, for strategies that don't
        # declare a static `.source` (e.g. a plain function).
        self._last_verification_source: Optional[str] = None

        # Timing (configurable — a lagging source must be compensable from
        # apps.yaml, not by editing this module). MUST be set before the
        # verifier is built: RegisterVerifier borrows _set_mode_timeout for its
        # own blocking reads.
        self._verify_delay = int(
            getattr(config, "verify_delay_seconds", VERIFY_DELAY_SECONDS)
        )
        self._verify_recheck_delay = int(
            getattr(config, "verify_recheck_seconds", VERIFY_RECHECK_SECONDS)
        )
        self._set_mode_timeout = int(
            getattr(config, "set_wit_mode_timeout_seconds",
                    SET_WIT_MODE_TIMEOUT_SECONDS)
        )

        # Verification strategy. None => nothing to verify against.
        self._verify_enabled = bool(verify_enabled)
        self._verifier: Optional[Verifier] = (
            verifier if verifier is not None else self._default_verifier()
        )

        # Diagnostics counters. They exist to separate "the HA sensor lags"
        # (mismatch_count high, resend_recovered_count high, persistent 0) from
        # "the inverter really falls back to Passthrough" (persistent grows)
        # from "the verification source tracks nothing" (recovered 0, persistent
        # ~= mismatch/2 while the battery physically obeys).
        self._mismatch_count = 0
        self._resend_count = 0
        self._resend_recovered_count = 0
        self._resend_failed_count = 0
        self._persistent_mismatch_count = 0
        self._unverifiable_count = 0
        self._verified_count = 0
        self._unprobative_match_count = 0
        self._last_mismatch: Optional[dict] = None

        # Outcome of the last apply_mode command, and a tally per outcome.
        # These are what separate "the inverter acknowledged N commands" from
        # "N commands timed out unconfirmed" / "N were never transmitted".
        self.last_apply_outcome: Optional[ApplyOutcome] = None
        self._apply_outcome_counts: dict = {}

    # ------------------------------------------------------------------
    # Verification wiring
    # ------------------------------------------------------------------

    def _default_verifier(self) -> Optional[Verifier]:
        """Build the strategy named by ``config.verify_source``.

        Values (read defensively with getattr, so an older config object still
        works):

          "registers"    RegisterVerifier — reads the VPP control registers back
                         through growatt_modbus/get_register_data. RECOMMENDED,
                         and the only source on this installation that is known
                         to reflect the active override.
          "mode_sensor"  ModeSensorVerifier against config.inverter_mode_sensor.
                         Only correct where that entity is not frozen by the
                         integration's never-cleared read blacklist.
          "none"         No verification.
          unset/"auto"   registers when a device_id exists, otherwise nothing.

        "auto" deliberately does NOT fall back to the mode sensor even when
        inverter_mode_sensor is set: that entity read 'Passthrough' through
        73/73 verifications of commands the battery executed, and each false
        mismatch costs a blocking ~10 s resend. Choosing it must be explicit.
        """
        source = getattr(self.config, "verify_source", "") or "auto"
        source = str(source).strip().lower()

        entity = getattr(self.config, "inverter_mode_sensor", "") or ""
        entity = entity.strip() if isinstance(entity, str) else ""
        device = getattr(self.config, "device_id", "") or ""

        if source in ("none", "off", "disabled", "false"):
            return None

        if source == "mode_sensor":
            if not entity:
                self.app.log(
                    "DirectControl: verify_source='mode_sensor' but "
                    "inverter_mode_sensor is empty — verification disabled",
                    level="WARNING",
                )
                return None
            return ModeSensorVerifier(self.app, entity)

        if source == "registers":
            if not device:
                return None  # dry run: nothing to read from
            return RegisterVerifier(self.app, device, self._set_mode_timeout)

        # auto
        if device:
            return RegisterVerifier(self.app, device, self._set_mode_timeout)
        return None

    def set_verifier(self, verifier: Optional[Verifier]) -> None:
        """Replace the verification strategy at runtime.

        Cancels any pending check first: a timer scheduled under the old
        strategy would evaluate the new one at a delay chosen for the old one.
        Pass None to stop verifying.
        """
        with self._state_lock:
            self._cancel_verification()
            self._verifier = verifier
            self._last_verification_source = None

    @property
    def verification_enabled(self) -> bool:
        """True when verify-after-set will actually run."""
        with self._state_lock:
            return bool(self._verify_enabled and self._verifier is not None)

    @property
    def verification_source(self) -> Optional[str]:
        """Identity of the verification source, or None when disabled.

        Prefers what the strategy declares (``.source``), then what it last
        actually reported, then the callable's own name — so a bare function
        used as a strategy is still identifiable in the diagnostics before it
        has run once.
        """
        with self._state_lock:
            if not self.verification_enabled:
                return None
            return (
                getattr(self._verifier, "source", None)
                or self._last_verification_source
                or getattr(self._verifier, "__name__", None)
                or type(self._verifier).__name__
            )

    @property
    def device_id(self) -> str:
        return self.config.device_id

    def _duration_for_slot(self) -> int:
        """Override duration: slot_minutes + safety buffer.

        If the optimizer misses a refresh, the override expires and
        the inverter reverts to its panel-configured base mode.
        """
        return self.config.slot_minutes + self.config.direct_control_buffer_minutes

    def apply_mode(self, entry: ScheduleEntry) -> bool:
        """Send mode command to inverter via set_wit_mode service.

        Backward-compatible boolean wrapper around
        :meth:`apply_mode_with_outcome`: False only for a CONFIRMED failure.
        Callers that need to distinguish "the inverter acknowledged" from
        "nothing was transmitted" or "we never found out" must use
        ``apply_mode_with_outcome`` (or read ``last_apply_outcome``).

        Args:
            entry: Schedule entry with mode, and optional export_rate
                   and ac_charge_mode.

        Returns:
            True unless the service call confirmed a failure.
        """
        return self.apply_mode_with_outcome(entry) is not ApplyOutcome.FAILED

    def apply_mode_with_outcome(self, entry: ScheduleEntry) -> ApplyOutcome:
        """Send mode command to inverter and report what actually happened.

        Concurrency: ``_io_lock`` is taken once the command has been built and
        is held through the duplicate check, the send and the last-sent record.
        That makes the duplicate decision atomic with the send it is about — a
        competing apply waits on ``_io_lock``, not on any app lock, and the last
        one to COMPLETE is the mode that stands. Building the command (HA state
        reads) stays outside the lock so it does not widen that window.

        Pending-timer handling, per exit path:
          * DRY_RUN — returns before any send. No timer can be pending, because
            a dry run never sends and so never schedules one.
          * SKIPPED_DUPLICATE — returns WITHOUT cancelling, deliberately: the
            pending check is for this very same mode, which is still the mode we
            want. Cancelling would throw away the only evidence we were going to
            collect about it.
          * every other path — cancels BEFORE sending, so even a confirmed
            failure cannot leave a stale timer that later resends a superseded
            mode.

        Args:
            entry: Schedule entry with mode, and optional export_rate
                   and ac_charge_mode.

        Returns:
            The :class:`ApplyOutcome` for this command. Also stored on
            ``self.last_apply_outcome`` and counted in ``get_diagnostics()``.
        """
        if not self.device_id:
            self.app.log(
                f"DirectControl: dry-run {entry.mode.name} ({entry.reason})"
            )
            return self._record_outcome(ApplyOutcome.DRY_RUN)

        mode = entry.mode
        duration = self._duration_for_slot()

        mode_str = {
            BatteryMode.CHARGE: self._resolve_charge_mode(entry),
            BatteryMode.DISCHARGE: self._resolve_discharge_mode(entry),
            BatteryMode.HOLD: "hold",
        }.get(mode, "hold")

        params = {
            "device_id": self.device_id,
            "mode": mode_str,
            "duration_minutes": duration,
        }

        # Power percent
        params["power_percent"] = self.config.default_power_percent

        # Export rate
        if entry.export_rate is not None:
            params["export_rate"] = entry.export_rate

        # AC charge mode
        ac_mode = self._ac_charge_mode_for_entry(entry)
        if ac_mode:
            params["ac_charge_mode"] = ac_mode

        # SOC limits
        if mode == BatteryMode.CHARGE:
            params["charge_cutoff_soc"] = self._get_max_soc()

        if mode == BatteryMode.DISCHARGE:
            params["discharge_cutoff_soc"] = self._get_min_soc()

        # One send at a time, and the duplicate decision inside the same
        # critical section as the send it guards.
        with self._io_lock:
            return self._apply_locked(mode_str, params, duration)

    def _apply_locked(self, mode_str: str, params: dict,
                      duration: int) -> ApplyOutcome:
        """apply_mode_with_outcome's critical section, with _io_lock held."""
        # Duplicate detection
        if self._is_duplicate(mode_str, params):
            self.app.log(
                f"DirectControl: skipping duplicate {mode_str} "
                f"(last sent {self._seconds_since_last():.0f}s ago)",
                level="DEBUG",
            )
            # NOTE: the pending verification (if any) is for this same mode and
            # is deliberately left running.
            return self._record_outcome(ApplyOutcome.SKIPPED_DUPLICATE)

        # Supersede any verification pending from a previous send BEFORE we send
        # (regardless of this send's outcome). Otherwise a confirmed failure
        # here would return without cancelling, and the stale timer could later
        # resend the now-superseded older mode. This also bumps the generation,
        # so a verification callback already dequeued on another worker thread
        # becomes inert.
        self._cancel_verification()

        self.app.log(
            f"DirectControl: {mode_str} "
            f"power={params.get('power_percent', '-')}% "
            f"duration={duration}min "
            f"export={params.get('export_rate', '-')} "
            f"ac={params.get('ac_charge_mode', '-')} "
            f"soc=[{params.get('discharge_cutoff_soc', '-')}"
            f"-{params.get('charge_cutoff_soc', '-')}]"
        )

        outcome = self._call_set_wit_mode(params)

        if outcome is False:
            # Confirmed failure: do NOT record last-sent, so duplicate
            # suppression can't mask an immediate resend on the next slot.
            self.app.log(
                f"DirectControl: set_wit_mode reported failure for {mode_str}"
                f"{self._error_suffix()}; not recording last-sent so a resend "
                "can correct it",
                level="ERROR",
            )
            return self._record_outcome(ApplyOutcome.FAILED)

        # outcome is True (confirmed) or None (unconfirmed — the request was on
        # the wire but no acknowledgement came back). In both cases record the
        # last-sent marker so the schedule isn't spammed with resends.
        with self._state_lock:
            self._last_mode_sent = mode_str
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()

        if outcome is None:
            self.app.log(
                f"DirectControl: {mode_str} unconfirmed (no acknowledgement "
                f"within {self._set_mode_timeout}s); the command was already "
                f"sent to Home Assistant and usually still executes"
                f"{self._verify_followup_suffix()}",
                level="WARNING",
            )

        self._schedule_verification(mode_str, params)
        return self._record_outcome(
            ApplyOutcome.SENT if outcome is True
            else ApplyOutcome.UNCONFIRMED_TIMEOUT
        )

    def _record_outcome(self, outcome: ApplyOutcome) -> ApplyOutcome:
        """Store and count one apply outcome, then return it unchanged.

        ``+= 1`` on a dict entry is a read-modify-write and loses increments
        under concurrent dispatch — and these counters gate the app's "3
        failures in a row" escalation.
        """
        with self._state_lock:
            self.last_apply_outcome = outcome
            self._apply_outcome_counts[outcome] = (
                self._apply_outcome_counts.get(outcome, 0) + 1
            )
        return outcome

    def _error_suffix(self) -> str:
        """`` (<error text>)`` when the last service call reported one."""
        return f" ({self._last_service_error})" if self._last_service_error else ""

    def _verify_followup_suffix(self) -> str:
        """Say whether anything will actually check up on an unconfirmed send."""
        if self.verification_enabled:
            return f"; will verify against {self.verification_source}"
        return "; verification is disabled, so this will not be checked"

    def release_control(self) -> bool:
        """Release all overrides — inverter reverts to base mode."""
        if not self.device_id:
            self.app.log("DirectControl: dry-run release_control (passthrough)")
            return True

        params = {"device_id": self.device_id, "mode": "passthrough"}

        # Same critical section as apply_mode: a release and an apply must not
        # interleave, and the last one to COMPLETE is the mode that stands.
        with self._io_lock:
            return self._release_locked(params)

    def _release_locked(self, params: dict) -> bool:
        """release_control's critical section, with _io_lock held."""
        # Cancel any pending verification from a previous send before sending,
        # so a failed release can't leave a stale timer that resends an older
        # mode ~90s later.
        self._cancel_verification()

        outcome = self._call_set_wit_mode(params)

        if outcome is False:
            self.app.log(
                f"DirectControl: release failed{self._error_suffix()}",
                level="ERROR",
            )
            return False

        with self._state_lock:
            self._last_mode_sent = "passthrough"
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()

        if outcome is None:
            self.app.log(
                f"DirectControl: passthrough unconfirmed (no acknowledgement "
                f"within {self._set_mode_timeout}s); the command was already "
                f"sent to Home Assistant and usually still executes"
                f"{self._verify_followup_suffix()}",
                level="WARNING",
            )
        else:
            self.app.log("DirectControl: released all overrides (passthrough)")

        # Verification still runs, but a MATCH here is marked non-probative: the
        # expected status IS the source's idle/no-override reading, so agreement
        # is not evidence that the release arrived. A MISMATCH (the source still
        # reports an active override) remains real evidence and is worth a
        # resend.
        self._schedule_verification("passthrough", params)
        return True

    def _call_set_wit_mode(self, params: dict) -> Optional[bool]:
        """Call the set_wit_mode service and classify the outcome.

        Returns:
            True  - confirmed success (handler returned success=True, or a
                    truthy response we couldn't disprove)
            False - confirmed failure: the handler raised (exception here), or
                    Home Assistant answered with success=False while the
                    round-trip itself completed normally.
            None  - unconfirmed: we stopped waiting, but the request was already
                    written to the websocket, so the service very likely ran.

        Why a timeout is *unconfirmed*, not a failure (AppDaemon 4.5.x, verified
        against the installed package):

        ``HassPlugin.websocket_send_json`` awaits the response future with
        ``asyncio.wait_for(..., timeout=hass_timeout)`` AFTER it has already
        done ``await self.ws.send_json(request)``. On ``asyncio.TimeoutError`` it
        does NOT raise and does NOT return None — it logs
        ``Timed out [0:00:15] waiting for request: {'type': 'call_service' ...}``
        and returns ``{"success": False, "ad_status": "TIMEOUT",
        "ad_duration": ...}``. When the real answer arrives later, the plugin
        logs ``Request already timed out for <id>``.

        Reading only ``success`` therefore classified every timeout as a
        confirmed failure: in the 2026-09-02 log four timeouts (06:15, 06:30,
        06:53, 06:54) produced "set_wit_mode reported failure" while the
        UNCONFIRMED_TIMEOUT branch never fired once. That is backwards — the
        "Request already timed out" pairs prove Home Assistant *did* answer.
        ``ad_status`` is what separates the two, so it is checked first.

        ``ad_status: TERMINATING`` (the wait was cancelled during AppDaemon
        shutdown) is unconfirmed for the same reason: the request was on the
        wire.

        A ``None`` return is still possible and still unconfirmed — the plugin
        returns None when the websocket is not connected, and
        ``run_coroutine_threadsafe`` returns None when AppDaemon's own
        ``internal_function_timeout`` (60 s) expires first.

        Threading: every caller holds ``_io_lock``, which is also what
        serializes ``_last_service_error`` with the log line that reads it. No
        lock is taken here — this is the blocking call the locks exist to bound.
        """
        self._last_service_error = None
        try:
            # hass_timeout: a formal parameter of the AppDaemon HASS plugin
            #   (>= 4.4). It is consumed by the plugin, NOT forwarded as
            #   service data.
            # No return_response/return_result kwarg is passed: set_wit_mode is
            #   registered SupportsResponse.OPTIONAL, and AppDaemon auto-enables
            #   return_response for such services, so call_service already
            #   surfaces the handler's response dict and propagates handler
            #   exceptions. Passing an unknown kwarg like return_result would be
            #   forwarded to HA's strict voluptuous schema and fail every call
            #   with "extra keys not allowed".
            result = self.app.call_service(
                "growatt_modbus/set_wit_mode",
                hass_timeout=self._set_mode_timeout,
                **params,
            )
        except Exception as e:
            self._last_service_error = f"{type(e).__name__}: {e}"
            self.app.log(
                f"DirectControl: set_wit_mode failed: {e}",
                level="ERROR",
            )
            return False

        if result is None:
            self._last_service_error = "no response from the AppDaemon HASS plugin"
            return None

        # Response shape from the handler on success:
        #   {"success": True, "mode_applied": ..., "registers_written": {...},
        #    "timestamp": ..., "override_expires": ...}
        # Tolerate the payload being nested under a "result" key on some AD
        # versions.
        if isinstance(result, dict):
            inner = result.get("result")
            inner = inner if isinstance(inner, dict) else {}

            ad_status = _ad_status_of(result)
            if ad_status in UNCONFIRMED_AD_STATUSES:
                self._last_service_error = (
                    f"AppDaemon ad_status={ad_status} after "
                    f"{self._set_mode_timeout}s"
                )
                return None

            success = result.get("success")
            if success is None:
                success = inner.get("success")
            if success is False:
                error = result.get("error", inner.get("error"))
                self._last_service_error = (
                    str(error) if error else "success=False"
                )
                return False

        return True

    def _cancel_verification(self) -> None:
        """Cancel any pending verification timer and invalidate its generation.

        ``cancel_timer`` alone is not enough under multi-thread dispatch: it
        cannot stop a callback AppDaemon has already handed to another worker.
        Bumping the generation makes such an in-flight callback inert, so it
        neither clears a newer handle nor resends the mode we just superseded.
        """
        with self._state_lock:
            self._verify_generation += 1
            if self._verify_timer is not None:
                handle, self._verify_timer = self._verify_timer, None
                try:
                    self.app.cancel_timer(handle)
                except Exception:
                    pass

    def _schedule_verification(
        self, mode_str: str, params: dict, attempt: int = 1
    ) -> None:
        """Schedule a one-shot verification after a mode was sent.

        Supersedes any previously pending verification, so a mode applied
        between send and verify cancels the stale check. A no-op when no
        verification source is configured — an unverified send is far cheaper
        than a false mismatch, which costs a blocking resend.

        Args:
            attempt: 1 for the check after the original send (delayed by
                verify_delay_seconds), 2 for the single re-check after a resend
                (delayed by verify_recheck_seconds). Attempt is capped at 2 in
                _verify_mode, so this can never become a resend loop.
        """
        with self._state_lock:
            self._cancel_verification()          # also bumps the generation
            if not self.verification_enabled:
                return
            generation = self._verify_generation
            delay = (self._verify_delay if attempt <= 1
                     else self._verify_recheck_delay)
            try:
                self._verify_timer = self.app.run_in(
                    self._verify_mode,
                    delay,
                    mode_str=mode_str,
                    params=params.copy(),
                    attempt=attempt,
                    generation=generation,
                )
            except Exception as e:
                self.app.log(
                    f"DirectControl: could not schedule verification: {e}",
                    level="ERROR",
                )
                self._verify_timer = None

    def _report_duration(self, name: str, seconds: float) -> None:
        """Feed this callback's wall time into the app's slow-callback advice.

        ``_verify_mode`` runs on the AppDaemon callback thread and its resend is
        a blocking ``set_wit_mode``: the 2026-09-02 log has eight verifications
        at 10-16 s that never showed up in the app's "Callback ... took Ns"
        accounting, because only methods decorated with the app's own
        ``@_timed_callback`` were measured. Callbacks this module registers
        itself must report themselves.

        Optional by design: ``getattr`` so DirectControl still works against an
        app (or test double) that has no such method.
        """
        record = getattr(self.app, "record_external_callback_duration", None)
        if record is None:
            return
        try:
            record(name, seconds)
        except Exception:
            pass

    def _verify_mode(self, kwargs=None) -> None:
        """AppDaemon scheduler entry point for verification (timed).

        ``_report_duration`` MUST stay here rather than inside
        ``_run_verification``: it calls back into the app, which runs it under
        the APP lock. Doing that while holding ``_io_lock``/``_state_lock``
        would invert the documented lock order (app -> io -> state) and deadlock
        against any callback that holds the app lock and then asks for
        ``get_diagnostics()``. By the time this ``finally`` runs,
        ``_run_verification`` has released both.
        """
        started = time.monotonic()
        try:
            self._run_verification(kwargs)
        finally:
            self._report_duration(
                "DirectControl._verify_mode", time.monotonic() - started
            )

    def _run_verification(self, kwargs=None) -> None:
        """Verify the inverter reached the last-sent mode; resend once if not.

        Attempt ladder (bounded — never a loop, max 2 checks and 2 sends):
          attempt 1: mismatch -> WARNING, resend once, schedule attempt 2
          attempt 2: match    -> INFO "recovered after resend"
                     mismatch -> ERROR, NO further resend, NO further timer

        The second check is what makes the diagnostics meaningful: without it we
        never learned whether the resend helped, so a lagging HA modbus sensor
        was indistinguishable from an inverter that genuinely drops back to
        Passthrough — and both are indistinguishable from a verification source
        that never tracked the override in the first place, which is why every
        log line names the source and the raw value it returned.

        Threading: the verification READ runs with no lock held. It is
        read-only, and an apply arriving mid-verification must not queue behind
        a diagnostic read — only the resend that may follow takes ``_io_lock``.
        Everything touching shared state takes ``_state_lock`` briefly.
        """
        kwargs = kwargs or {}
        mode_str = kwargs.get("mode_str")
        params = kwargs.get("params", {})
        attempt = int(kwargs.get("attempt", 1) or 1)
        generation = kwargs.get("generation")

        with self._state_lock:
            if generation is not None and generation != self._verify_generation:
                # Superseded while this callback sat in a worker queue. Return
                # WITHOUT clearing _verify_timer — that handle belongs to a
                # newer verification now.
                return
            self._verify_timer = None
            if generation is not None:
                # CLAIM this verification by bumping the generation and taking
                # the new value as our stamp. Two things need this:
                #   * a duplicate dispatch of the same callback (AppDaemon can
                #     hand a dequeued timer to more than one worker) would
                #     otherwise pass the guard too and double-count — six
                #     workers on one attempt-2 callback logged six persistent
                #     mismatches and six ERRORs for one event;
                #   * the resend below re-checks this stamp, so any apply that
                #     supersedes us mid-verification still invalidates it.
                self._verify_generation += 1
                generation = self._verify_generation
            verifier = self._verifier
            enabled = self._verify_enabled

        try:
            if not enabled or verifier is None:
                return

            outcome = verifier(mode_str, params)
            with self._state_lock:
                self._last_verification_source = outcome.source

            if outcome.result is VerificationResult.UNVERIFIABLE:
                # Cannot verify — don't resend blindly. Warn the FIRST time so a
                # wrong/misconfigured source is visible; stay DEBUG after that to
                # avoid log spam when it is merely briefly offline.
                detail = outcome.detail or "source returned no usable value"
                with self._state_lock:
                    self._unverifiable_count += 1
                    first_unreadable = not self._verify_unreadable_warned
                    self._verify_unreadable_warned = True
                if first_unreadable:
                    self.app.log(
                        f"DirectControl: cannot verify {mode_str} — "
                        f"{outcome.source}: {detail}. If this persists, check "
                        "that inverter_mode_sensor points at an entity that "
                        "reflects the ACTIVE override (verification is skipped "
                        "until it reads a value; set it to \"\" to turn "
                        "verification off deliberately).",
                        level="WARNING",
                    )
                else:
                    self.app.log(
                        f"DirectControl: cannot verify {mode_str} — "
                        f"{outcome.source}: {detail}",
                        level="DEBUG",
                    )
                return

            if outcome.result is VerificationResult.MATCH:
                if not outcome.probative:
                    # Agreement that would have happened anyway. Counting it as
                    # "verified" (or as "recovered after resend") would turn the
                    # absence of information into evidence.
                    with self._state_lock:
                        self._unprobative_match_count += 1
                    self.app.log(
                        f"DirectControl: {mode_str} — {outcome.source} reports "
                        f"'{outcome.observed}' as expected, but that is also "
                        "what this source reports when no override is tracked; "
                        "not counted as verified",
                        level="DEBUG",
                    )
                    return

                with self._state_lock:
                    self._verified_count += 1
                    if attempt > 1:
                        self._resend_recovered_count += 1
                if attempt > 1:
                    self.app.log(
                        f"DirectControl: {mode_str} recovered after resend — "
                        f"{outcome.source} now reports '{outcome.observed}'"
                    )
                else:
                    self.app.log(
                        f"DirectControl: verified {mode_str} — "
                        f"{outcome.source} reports '{outcome.observed}'",
                        level="DEBUG",
                    )
                return

            # MISMATCH
            with self._state_lock:
                self._mismatch_count += 1
                self._last_mismatch = {
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "mode": mode_str,
                    "expected": outcome.expected,
                    "actual": outcome.observed,
                    "attempt": attempt,
                    "source": outcome.source,
                }
                if attempt >= 2:
                    self._persistent_mismatch_count += 1

            if attempt >= 2:
                # Already resent once and the source still disagrees. Escalate
                # and STOP (no third send, no third timer).
                self.app.log(
                    f"DirectControl: persistent mode mismatch after resend — "
                    f"{outcome.source} reports '{outcome.observed}', expected "
                    f"'{outcome.expected}' for {mode_str}. Not resending again "
                    f"(the next slot retries). Two causes fit and this source "
                    f"cannot separate them: (a) the inverter is not applying or "
                    f"is dropping the override, or (b) this verification source "
                    f"does not track the override. Check whether the battery "
                    f"physically followed the command (SOC trend, battery "
                    f"power) before treating it as (a).",
                    level="ERROR",
                )
                return

            # Mismatch — resend the same params ONCE, bypassing duplicate
            # suppression by clearing the last-sent timestamp.
            self.app.log(
                f"DirectControl: mode mismatch — {outcome.source} reports "
                f"'{outcome.observed}', expected '{outcome.expected}' for "
                f"{mode_str}; resending once",
                level="WARNING",
            )
            # The resend IS a send: it takes _io_lock so it cannot interleave
            # with an apply on another thread. The verification READ above
            # deliberately did not.
            with self._io_lock:
                with self._state_lock:
                    if (generation is not None
                            and generation != self._verify_generation):
                        # An apply won the race for _io_lock while we waited and
                        # superseded this mode (its _cancel_verification bumped
                        # past our claim). Resending now would put the inverter
                        # back into the mode the schedule just left.
                        return
                    self._last_mode_time = None  # bypass _is_duplicate
                    self._resend_count += 1
                outcome_send = self._call_set_wit_mode(params)
                self._after_resend(outcome_send, mode_str, params)

        except Exception as e:
            self.app.log(
                f"DirectControl: verification error for {mode_str}: {e}",
                level="ERROR",
            )

    def _after_resend(self, outcome_send, mode_str: str, params: dict) -> None:
        """Record the resend's outcome. Called with _io_lock held."""
        if outcome_send is False:
            with self._state_lock:
                self._resend_failed_count += 1
            self.app.log(
                f"DirectControl: resend of {mode_str} failed"
                f"{self._error_suffix()}; leaving last-sent cleared so the "
                "next slot re-sends instead of suppressing it as a duplicate",
                level="ERROR",
            )
            # _last_mode_time stays None ON PURPOSE. It mirrors the
            # confirmed-failure path in apply_mode_with_outcome: after a send we
            # know failed, duplicate suppression must not swallow the next
            # attempt. The cost is one extra (harmless) inverter write if the
            # schedule repeats the same mode within half a slot.
            return

        # Record last-sent again, then re-check exactly ONCE so we learn
        # whether the resend actually took effect.
        with self._state_lock:
            self._last_mode_sent = mode_str
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()
        self._schedule_verification(mode_str, params, attempt=2)

    def get_diagnostics(self) -> dict:
        """Counters that make inverter-control health observable in HA.

        Interpretation:
          * mismatch_count high, resend_recovered_count ~= resend_count,
            persistent_mismatch_count == 0  -> the verification source merely
            LAGS. Raise verify_delay_seconds.
          * resend_recovered_count == 0 and persistent_mismatch_count ~=
            mismatch_count / 2 (every mismatch escalates, none ever recovers),
            while the battery physically does what was asked -> THE VERIFICATION
            SOURCE DOES NOT TRACK THE OVERRIDE. This is not a firmware fault and
            no amount of verify_delay_seconds will fix it. Observed 2026-09-02
            with verification_source == mode_sensor:*: 73/73 checks read
            'Passthrough' while SOC rose 9 -> 21 % on grid_charge and max_export
            moved 0.7 kWh in 4 minutes, because the integration had blacklisted
            the registers that sensor is computed from. Switch
            verify_source to "registers", which reads them directly. Each false
            mismatch costs a blocking ~10 s resend on the callback thread.
          * unverifiable_count climbing with verification_source == registers:*
            -> get_register_data is failing or timing out. Nothing is resent on
            that path (an unreadable source is never a mismatch), so this is a
            visibility problem, not a control problem: check the Modbus link.
          * persistent_mismatch_count growing while the battery does NOT follow
            the command -> the inverter genuinely drops the override (e.g. back
            to Passthrough). A configuration/firmware problem.
          * resend_failed_count growing -> the set_wit_mode service itself is
            failing; check the Modbus connection.
          * unconfirmed_count growing while sent_count stays flat -> every
            set_wit_mode call is hitting its client-side timeout. That is a hung
            growatt_modbus, even though apply_mode keeps returning True.
          * verification_source == None -> verification is off; mismatch and
            verified counters stay at 0 by construction.
          * unprobative_match_count only ever counts passthrough releases, whose
            expected status is also the source's no-override reading.
        """
        with self._state_lock:
            return self._diagnostics_locked()

    def _diagnostics_locked(self) -> dict:
        """``get_diagnostics`` body, with ``_state_lock`` held.

        Deliberately never touches ``_io_lock``: this runs from the app's
        health-sensor update under the APP lock, and must not wait out a 15 s
        inverter write.
        """
        counts = self._apply_outcome_counts
        return {
            "sent_count": counts.get(ApplyOutcome.SENT, 0),
            "unconfirmed_count": counts.get(ApplyOutcome.UNCONFIRMED_TIMEOUT, 0),
            "duplicate_skipped_count": counts.get(
                ApplyOutcome.SKIPPED_DUPLICATE, 0
            ),
            "dry_run_count": counts.get(ApplyOutcome.DRY_RUN, 0),
            "failed_count": counts.get(ApplyOutcome.FAILED, 0),
            "last_apply_outcome": (
                self.last_apply_outcome.value if self.last_apply_outcome else None
            ),
            "verification_enabled": self.verification_enabled,
            "verification_source": self.verification_source,
            "verify_generation": self._verify_generation,
            "mismatch_count": self._mismatch_count,
            "resend_count": self._resend_count,
            "resend_recovered_count": self._resend_recovered_count,
            "resend_failed_count": self._resend_failed_count,
            "persistent_mismatch_count": self._persistent_mismatch_count,
            "unverifiable_count": self._unverifiable_count,
            "verified_count": self._verified_count,
            "unprobative_match_count": self._unprobative_match_count,
            "last_mismatch": self._last_mismatch,
            "verify_delay_seconds": self._verify_delay,
            "verify_recheck_seconds": self._verify_recheck_delay,
            "set_wit_mode_timeout_seconds": self._set_mode_timeout,
        }

    def _ac_charge_mode_for_entry(self, entry: ScheduleEntry) -> str:
        """Determine AC charge mode based on entry and PV conditions."""
        if entry.ac_charge_mode is not None:
            return entry.ac_charge_mode

        if entry.mode == BatteryMode.CHARGE:
            pv_power = self._get_pv_power()
            if pv_power is not None and pv_power > self._get_pv_threshold():
                return "pv_priority"
            return "ac_priority"

        return "disabled"

    def _resolve_charge_mode(self, entry: ScheduleEntry) -> str:
        """Map CHARGE to service mode string."""
        return "grid_charge"

    def _resolve_discharge_mode(self, entry: ScheduleEntry) -> str:
        """Map DISCHARGE + export_rate to specific mode string."""
        export_rate = entry.export_rate

        if export_rate is not None and export_rate > 0:
            power = self.config.default_power_percent
            if export_rate >= 100 and power >= 100:
                return "max_export"
            return "discharge_to_grid"
        # Default: no accidental export
        return "discharge_to_load"

    def _get_pv_power(self) -> Optional[float]:
        """Read current PV power from HA sensor."""
        try:
            state = self.app.get_state(self.config.pv_power_sensor)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return None

    def _get_pv_threshold(self) -> float:
        """Get current PV threshold from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.pv_threshold_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self.config.default_pv_threshold

    def _get_min_soc(self) -> int:
        """Get current min SOC from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.min_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return int(float(state))
        except (ValueError, TypeError):
            pass
        return int(self.config.default_min_soc)

    def _get_max_soc(self) -> int:
        """Get current max SOC from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.max_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return int(float(state))
        except (ValueError, TypeError):
            pass
        return int(self.config.default_max_soc)

    def _is_duplicate(self, mode_str: str, params: dict) -> bool:
        """Check if this command is identical to the last one sent recently.

        Callers inside ``apply_mode_with_outcome`` already hold ``_io_lock``, so
        the answer cannot go stale between here and the send it guards.
        ``_state_lock`` only protects the read of the last-sent triple against a
        verification resend updating it.
        """
        with self._state_lock:
            if self._last_mode_sent != mode_str:
                return False
            if not self._last_mode_time:
                # Cleared on purpose after a confirmed failure or before a
                # verification resend: suppression is off until a send succeeds.
                return False

            elapsed = (
                datetime.datetime.now() - self._last_mode_time
            ).total_seconds()
            half_slot = self.config.slot_minutes * 60 / 2
            if elapsed > half_slot:
                return False  # Time to refresh even if same mode

            for key in ("mode", "export_rate", "ac_charge_mode",
                        "charge_cutoff_soc", "discharge_cutoff_soc"):
                if params.get(key) != self._last_params.get(key):
                    return False

            return True

    def _seconds_since_last(self) -> float:
        with self._state_lock:
            if self._last_mode_time:
                return (
                    datetime.datetime.now() - self._last_mode_time
                ).total_seconds()
            return float('inf')
