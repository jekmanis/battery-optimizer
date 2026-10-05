"""
Slot outcome tracking for prediction monitoring.

Records per-slot predictions vs actuals to evaluate optimizer accuracy.
Rolling window of recent outcomes with summary statistics.
"""

import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class SlotOutcome:
    """Outcome of a single scheduled time slot."""
    slot_time: str  # ISO format
    # Predictions (recorded at slot start)
    scheduled_mode: str = ""
    predicted_soc_end: float = 0.0
    predicted_load_kw: float = 0.0
    predicted_pv_kw: float = 0.0
    # Actuals (recorded at slot end)
    actual_soc_end: Optional[float] = None
    actual_pv_kw: Optional[float] = None
    actual_mode: Optional[str] = None
    # Derived errors
    soc_error: Optional[float] = None
    load_error_pct: Optional[float] = None
    pv_error_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class SlotOutcomeTracker:
    """
    Tracks per-slot predictions vs actuals for monitoring optimizer accuracy.

    Usage:
    - Call record_slot_start() at the beginning of each slot
    - Call record_slot_end() at the end (or beginning of next slot)
    - Call get_recent_outcomes() / get_accuracy_stats() for monitoring
    """

    def __init__(self, slot_minutes: int = 15, max_outcomes: int = 96, log_func=None):
        self._slot_minutes = slot_minutes
        self._max_outcomes = max_outcomes
        self._outcomes: List[SlotOutcome] = []
        self._pending: Optional[SlotOutcome] = None
        self._log = log_func or (lambda *a, **kw: None)

    def record_slot_start(
        self,
        slot_time: datetime.datetime,
        mode: str,
        predicted_soc_end: float,
        predicted_load_kw: float,
        predicted_pv_kw: float = 0.0,
    ):
        """Record predictions at the start of a slot."""
        # Finalize any pending outcome from previous slot
        if self._pending:
            self._finalize_pending()

        self._pending = SlotOutcome(
            slot_time=slot_time.isoformat(),
            scheduled_mode=mode,
            predicted_soc_end=predicted_soc_end,
            predicted_load_kw=predicted_load_kw,
            predicted_pv_kw=predicted_pv_kw,
        )

    def record_slot_end(
        self,
        actual_soc: Optional[float] = None,
        actual_pv_w: Optional[float] = None,
        actual_mode: Optional[str] = None,
    ):
        """Record actuals at the end of a slot (or start of next)."""
        if self._pending is None:
            return

        self._pending.actual_soc_end = actual_soc
        self._pending.actual_pv_kw = actual_pv_w / 1000.0 if actual_pv_w is not None else None
        self._pending.actual_mode = actual_mode

        # Compute errors
        if actual_soc is not None:
            self._pending.soc_error = actual_soc - self._pending.predicted_soc_end

        pred_pv = self._pending.predicted_pv_kw
        if self._pending.actual_pv_kw is not None and pred_pv > 0.01:
            self._pending.pv_error_pct = (
                (self._pending.actual_pv_kw - pred_pv) / pred_pv * 100
            )

        self._finalize_pending()

    def _finalize_pending(self):
        """Move pending outcome to completed list."""
        if self._pending is None:
            return
        self._outcomes.append(self._pending)
        if len(self._outcomes) > self._max_outcomes:
            self._outcomes = self._outcomes[-self._max_outcomes:]
        self._pending = None

    def get_recent_outcomes(self, n: int = 10) -> List[Dict]:
        """Get last N completed outcomes as dicts."""
        return [o.to_dict() for o in self._outcomes[-n:]]

    def get_accuracy_stats(self) -> Dict:
        """Compute summary accuracy metrics from recent outcomes."""
        if not self._outcomes:
            return {
                "soc_mae": None,
                "pv_mape": None,
                "mode_compliance_pct": None,
                "outcome_count": 0,
            }

        # SOC Mean Absolute Error
        soc_errors = [
            abs(o.soc_error) for o in self._outcomes
            if o.soc_error is not None
        ]
        soc_mae = sum(soc_errors) / len(soc_errors) if soc_errors else None

        # PV Mean Absolute Percentage Error
        pv_errors = [
            abs(o.pv_error_pct) for o in self._outcomes
            if o.pv_error_pct is not None
        ]
        pv_mape = sum(pv_errors) / len(pv_errors) if pv_errors else None

        # Mode compliance: scheduled mode matches actual inverter mode
        # Mapping from optimizer BatteryMode.name to expected inverter mode sensor strings
        _MODE_MATCH = {
            "HOLD": {"hold"},
            "CHARGE": {"grid charge", "gridcharge"},
            "DISCHARGE": {"discharge to load", "dischargetoload",
                          "discharge to grid", "dischargetogrid",
                          "max export", "maxexport"},
        }

        def _modes_match(scheduled: str, actual: str) -> bool:
            norm_actual = actual.lower().replace("_", "").replace("-", " ").strip()
            expected = _MODE_MATCH.get(scheduled, set())
            for variant in expected:
                if variant.replace(" ", "") == norm_actual.replace(" ", ""):
                    return True
            return False

        mode_checks = [
            o for o in self._outcomes
            if o.actual_mode is not None
        ]
        if mode_checks:
            compliant = sum(
                1 for o in mode_checks
                if _modes_match(o.scheduled_mode, o.actual_mode)
            )
            mode_compliance_pct = compliant / len(mode_checks) * 100
        else:
            mode_compliance_pct = None

        return {
            "soc_mae": round(soc_mae, 2) if soc_mae is not None else None,
            "pv_mape": round(pv_mape, 1) if pv_mape is not None else None,
            "mode_compliance_pct": round(mode_compliance_pct, 1) if mode_compliance_pct is not None else None,
            "outcome_count": len(self._outcomes),
        }
