"""Regression: the summary census must describe the schedule that will execute.

`DPOptimizerResult` counts the plan the DP produced. The orchestrator then
mutates it — the cloud-safe HOLD -> DISCHARGE(to load) conversion during PV
hours — but the "Schedule generated: N charge, M discharge(self), K hold" line
kept reporting the DP's pre-conversion counts. On any sunny run with
cloud_safe_count > 0 the log therefore printed two contradictory mode censuses
a few lines apart: the summary line, and the schedule log's own "Total:" line
which has always counted the final entries.

Both now go through `count_schedule_modes`.
"""

import datetime

from battery_optimizer_lib.models import (
    BatteryMode,
    ScheduleEntry,
    ScheduleModeCounts,
    count_schedule_modes,
)


def _schedule(*specs):
    """specs: (mode, export_rate) tuples -> {time: ScheduleEntry}."""
    base = datetime.datetime(2026, 7, 28, 10, 0)
    return {
        base + datetime.timedelta(minutes=15 * i): ScheduleEntry(
            time=base + datetime.timedelta(minutes=15 * i),
            mode=mode,
            reason="test",
            export_rate=export_rate,
        )
        for i, (mode, export_rate) in enumerate(specs)
    }


def test_counts_charge_hold_export_and_self_consume():
    schedule = _schedule(
        (BatteryMode.CHARGE, None),
        (BatteryMode.HOLD, None),
        (BatteryMode.DISCHARGE, None),   # to load
        (BatteryMode.DISCHARGE, 0),      # to load (explicit zero export)
        (BatteryMode.DISCHARGE, 100),    # export
    )

    counts = count_schedule_modes(schedule)

    assert counts == ScheduleModeCounts(
        charge=1, hold=1, export=1, self_consume=2
    )
    assert counts.discharge == 3


def test_counts_reflect_the_cloud_safe_conversion():
    """The whole point: recount AFTER HOLD -> DISCHARGE(to load)."""
    schedule = _schedule(
        (BatteryMode.CHARGE, None),
        (BatteryMode.HOLD, None),
        (BatteryMode.HOLD, None),
        (BatteryMode.HOLD, None),
    )
    before = count_schedule_modes(schedule)
    assert (before.hold, before.self_consume) == (3, 0)

    # What the orchestrator's cloud-safe loop does to the entries.
    for entry in list(schedule.values())[1:3]:
        entry.mode = BatteryMode.DISCHARGE
        entry.export_rate = 0
        entry.reason += " [cloud-safe]"

    after = count_schedule_modes(schedule)

    assert (after.hold, after.self_consume, after.export) == (1, 2, 0)
    assert after.charge == 1


def test_summary_parts_omit_empty_discharge_kinds():
    counts = count_schedule_modes(
        _schedule((BatteryMode.CHARGE, None), (BatteryMode.HOLD, None))
    )

    assert counts.summary_parts() == ["1 charge", "1 hold"]

    with_both = count_schedule_modes(
        _schedule((BatteryMode.DISCHARGE, 0), (BatteryMode.DISCHARGE, 100))
    )
    assert with_both.summary_parts() == [
        "0 charge", "1 discharge(self)", "1 discharge(export)", "0 hold"
    ]


def test_empty_schedule_counts_zero():
    counts = count_schedule_modes({})
    assert counts == ScheduleModeCounts()
    assert counts.summary_parts() == ["0 charge", "0 hold"]
