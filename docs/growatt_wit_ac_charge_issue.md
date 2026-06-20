# WIT 15000TL3 — Cannot enable AC (grid) charging via VPP Modbus registers

**Inverter:** Growatt WIT 15000TL3 (DTC: 5603)
**VPP Protocol Version:** 202 (V2.02)
**Firmware:** YE1.0
**Connection:** Modbus TCP (port 502)

## Problem

I use VPP registers to remotely control battery modes (charge, discharge, hold) for energy price optimization. All modes work correctly **except grid charging via remote power control (30407-30409)** — the inverter rejects the register value needed to enable AC charging.

## What works

| Operation | Registers | Result |
|-----------|-----------|--------|
| Discharge to load/grid | 30407=1, 30409=-100 | Works — battery discharges at full power |
| Hold (idle) | 30407=1, 30409=0 | Works — battery holds SOC |
| TOU schedule (positive power) | 30411=N, 30412-30414 periods | Works — charges from grid even without PV |
| AC charge enable = PV priority | 30410=1 (via FC 0x10) | Accepted |

## What does NOT work

**Register 30410 = 2 (AC priority)** — needed to enable grid charging via remote power control (30407/30409):

```
Write FC 0x10: address=30410, value=[2]
Response: ExceptionResponse(function_code=144, exception_code=1)  — Illegal Function
```

```
Write FC 0x06: address=30410, value=2
Response: ExceptionResponse(function_code=134, exception_code=1)  — Illegal Function
```

The register only accepts values **0** (disabled) and **1** (PV priority). Value **2** (AC priority) is rejected by the firmware.

With `30410=1` and remote power control (`30407=1, 30409=+100`), the inverter does **not charge from the grid**. The duration register (30408) does not count down, suggesting the remote charge command is not being executed.

I also tried:
- Legacy register 949 (`ac_charge_enable`) — write succeeds but value does not persist (reads back 0)
- Legacy register 905 (`ac_charge_power_rate`) — same behavior, writes accepted but not retained
- Legacy registers 201/202 (work mode + power rate) — register 201 does not persist

## Current workaround

I use TOU schedule periods (30411-30414) with positive power values (+100%) instead of remote power control (30407-30409) for charging. This works, but TOU scheduling is less flexible than remote power control — it requires rewriting period registers for each schedule change and is limited to 20 periods.

## Questions

1. Is there a different register or register sequence on the WIT platform that enables AC/grid charging via remote power control (30407-30409)?
2. Is `30410=2` support planned for a future firmware update?
3. Is there an alternative VPP command to achieve grid charging without TOU periods?
