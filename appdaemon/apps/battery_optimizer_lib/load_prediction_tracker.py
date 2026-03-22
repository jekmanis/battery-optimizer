"""
Load prediction accuracy tracker.

Records predicted vs actual load for each time slot, computes correction
factors, and provides risk metrics for discharge scheduling.
"""

import datetime
import json
import math
from typing import Dict, List, Optional

from .models import PredictionAccuracyStats, BatteryMode, ScheduleEntry


# Rolling window sizes
MAX_SAMPLES_PER_SLOT = 30   # ~1 month at 1 per day per slot
MAX_GLOBAL_SAMPLES = 200

# Ratio clamp bounds
MIN_RATIO = 0.1
MAX_RATIO = 10.0

# Minimum samples for per-slot correction
MIN_SLOT_SAMPLES = 5

# Minimum total comparisons for full confidence
FULL_CONFIDENCE_COUNT = 50


class LoadPredictionTracker:
    """Tracks prediction accuracy and provides correction factors."""

    def __init__(self, slot_minutes: int = 15, log_func=None):
        self.slot_minutes = max(1, int(slot_minutes))
        self.log = log_func or (lambda msg: None)
        self.stats = PredictionAccuracyStats()
        # Pending predictions keyed by slot string
        self._pending: Dict[str, float] = {}

    def _slot_key(self, dt: datetime.datetime) -> str:
        minutes = dt.hour * 60 + dt.minute
        return str(int(minutes // self.slot_minutes))

    def record_prediction(self, dt: datetime.datetime, predicted_kw: float):
        """Store a pending prediction for a future slot."""
        if predicted_kw <= 0:
            return
        key = self._slot_key(dt)
        self._pending[key] = predicted_kw

    def record_actual(self, dt: datetime.datetime, actual_kw: float):
        """Match actual load against pending prediction and store ratio."""
        key = self._slot_key(dt)
        predicted = self._pending.pop(key, None)
        if predicted is None or predicted <= 0:
            return
        if actual_kw <= 0:
            return

        ratio = actual_kw / predicted
        ratio = max(MIN_RATIO, min(MAX_RATIO, ratio))

        # Per-slot storage
        slot_ratios = self.stats.ratios_by_slot.get(key, [])
        slot_ratios.append(ratio)
        if len(slot_ratios) > MAX_SAMPLES_PER_SLOT:
            slot_ratios = slot_ratios[-MAX_SAMPLES_PER_SLOT:]
        self.stats.ratios_by_slot[key] = slot_ratios

        # Global storage
        self.stats.global_ratios.append(ratio)
        if len(self.stats.global_ratios) > MAX_GLOBAL_SAMPLES:
            self.stats.global_ratios = self.stats.global_ratios[-MAX_GLOBAL_SAMPLES:]

        self.stats.total_comparisons += 1
        self.stats.last_comparison = dt.isoformat()

    def get_correction_factor(self, dt: datetime.datetime) -> float:
        """
        Return correction factor for a slot time.

        Fallback chain: per-slot median -> adjacent slots -> global median -> 1.0
        """
        key = self._slot_key(dt)

        # Try per-slot
        slot_ratios = self.stats.ratios_by_slot.get(key, [])
        if len(slot_ratios) >= MIN_SLOT_SAMPLES:
            return _median(slot_ratios)

        # Try adjacent slots (+-1)
        slot_idx = int(key)
        slots_per_day = 1440 // self.slot_minutes
        adjacent_ratios = list(slot_ratios)  # include current slot's data too
        for offset in (-1, 1):
            adj_key = str((slot_idx + offset) % slots_per_day)
            adjacent_ratios.extend(self.stats.ratios_by_slot.get(adj_key, []))
        if len(adjacent_ratios) >= MIN_SLOT_SAMPLES:
            return _median(adjacent_ratios)

        # Try global
        if self.stats.global_ratios:
            return _median(self.stats.global_ratios)

        return 1.0

    def get_risk_metrics(self) -> dict:
        """Compute overall prediction accuracy risk metrics."""
        ratios = self.stats.global_ratios
        if not ratios:
            return {
                "overall_bias": 1.0,
                "underestimate_pct": 0.0,
                "p90_ratio": 1.0,
                "worst_slot": None,
                "worst_slot_ratio": 1.0,
                "confidence": 0.0,
            }

        sorted_ratios = sorted(ratios)
        overall_bias = _median(ratios)
        underestimate_pct = sum(1 for r in ratios if r > 1.0) / len(ratios) * 100.0
        p90_idx = min(len(sorted_ratios) - 1, int(math.ceil(len(sorted_ratios) * 0.9)) - 1)
        p90_ratio = sorted_ratios[p90_idx]

        # Find worst slot
        worst_slot = None
        worst_slot_ratio = 1.0
        for slot_key, slot_ratios in self.stats.ratios_by_slot.items():
            if slot_ratios:
                med = _median(slot_ratios)
                if med > worst_slot_ratio:
                    worst_slot_ratio = med
                    worst_slot = slot_key
        # Convert slot key to time string
        if worst_slot is not None:
            slot_idx = int(worst_slot)
            h = (slot_idx * self.slot_minutes) // 60
            m = (slot_idx * self.slot_minutes) % 60
            worst_slot = f"{h:02d}:{m:02d}"

        confidence = min(1.0, self.stats.total_comparisons / FULL_CONFIDENCE_COUNT)

        return {
            "overall_bias": round(overall_bias, 3),
            "underestimate_pct": round(underestimate_pct, 1),
            "p90_ratio": round(p90_ratio, 3),
            "worst_slot": worst_slot,
            "worst_slot_ratio": round(worst_slot_ratio, 3),
            "confidence": round(confidence, 3),
        }

    def get_schedule_risk_assessment(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        discharge_mode: BatteryMode = BatteryMode.DISCHARGE,
    ) -> dict:
        """Assess risk of discharge slots being under-predicted."""
        discharge_risks = {}
        risk_scores = []

        for dt, entry in schedule.items():
            if entry.mode != discharge_mode:
                continue
            key = self._slot_key(dt)
            slot_ratios = self.stats.ratios_by_slot.get(key, [])
            if not slot_ratios:
                continue
            # % of times actual exceeded predicted
            exceed_pct = sum(1 for r in slot_ratios if r > 1.0) / len(slot_ratios) * 100.0
            time_str = dt.strftime("%H:%M")
            discharge_risks[time_str] = round(exceed_pct, 1)
            risk_scores.append(exceed_pct)

        if not risk_scores:
            overall_risk = "low"
        else:
            avg_risk = sum(risk_scores) / len(risk_scores)
            if avg_risk > 60:
                overall_risk = "high"
            elif avg_risk > 30:
                overall_risk = "medium"
            else:
                overall_risk = "low"

        return {
            "discharge_slot_risks": discharge_risks,
            "overall_risk": overall_risk,
        }

    def to_json(self) -> str:
        """Serialize to JSON for persistence."""
        data = {
            "version": 1,
            "slot_minutes": self.slot_minutes,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        """Load from JSON. Returns True if successful."""
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                stored_slot_minutes = data.get("slot_minutes")
                if stored_slot_minutes != self.slot_minutes:
                    self.log(
                        f"Prediction tracker slot size mismatch "
                        f"({stored_slot_minutes}->{self.slot_minutes}), starting fresh"
                    )
                    return False
                if "stats" in data:
                    self.stats = PredictionAccuracyStats.from_dict(data["stats"])
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load prediction tracker: {e}")
        return False


def _median(values: List[float]) -> float:
    """Return the median of a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0
