"""
PV production profile module for statistical PV forecasting.

Tracks historical PV production by time-of-day slots and provides
quantile-based forecasts for self-consumption planning.
"""

import datetime
import json

from .models import PvProfileStats
from .load_profile import _quantile


class PvProfile:
    """
    Statistical PV production profile by time-of-day slots.
    Stores recent samples per slot and returns a quantile-based forecast.
    """

    def __init__(
        self,
        slot_minutes: int,
        default_pv_w: float = 0.0,
        max_samples: int = 60,
        min_samples: int = 6,
        log_func=None,
    ):
        self.slot_minutes = max(1, int(slot_minutes))
        self.slots_per_day = int(1440 / self.slot_minutes)
        self.default_pv_w = float(default_pv_w)
        self.max_samples = max(1, int(max_samples))
        self.min_samples = max(1, int(min_samples))
        self.log = log_func or print
        self.stats = PvProfileStats()

    def _slot_index(self, dt: datetime.datetime) -> int:
        minutes = dt.hour * 60 + dt.minute
        return int(minutes // self.slot_minutes)

    def record(self, dt: datetime.datetime, pv_w: float):
        """Record a PV production observation for the given time slot."""
        if pv_w < 0:
            return
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        samples.append(float(pv_w))
        if len(samples) > self.max_samples:
            samples = samples[-self.max_samples:]
        self.stats.samples_by_slot[slot] = samples
        self.stats.observation_count += 1
        self.stats.last_observation = dt.isoformat()

    def predict_kw(self, dt: datetime.datetime, quantile: float = 0.5) -> float:
        """
        Predict expected PV production (kW) for a slot using stored samples.

        Uses quantile-based forecast blended with default based on confidence.
        Default quantile=0.5 (median) since PV is symmetric (cloudy vs sunny).
        """
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        if not samples:
            return max(0.0, self.default_pv_w) / 1000.0
        q_value = _quantile(samples, quantile)
        confidence = min(1.0, len(samples) / self.min_samples)
        blended = (self.default_pv_w * (1 - confidence)) + (q_value * confidence)
        return max(0.0, blended) / 1000.0

    def to_json(self) -> str:
        """Serialize PV profile for persistence."""
        data = {
            "version": 1,
            "slot_minutes": self.slot_minutes,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        """Load PV profile from JSON. Returns True if successful."""
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                stored_slot_minutes = data.get("slot_minutes")
                if stored_slot_minutes != self.slot_minutes:
                    if (stored_slot_minutes and
                            stored_slot_minutes > self.slot_minutes and
                            stored_slot_minutes % self.slot_minutes == 0):
                        factor = stored_slot_minutes // self.slot_minutes
                        self.log(
                            f"Migrating PV profile from {stored_slot_minutes}min "
                            f"to {self.slot_minutes}min slots (factor={factor})"
                        )
                        old_stats = data.get("stats", {})
                        old_samples = old_stats.get("samples_by_slot", {})
                        new_samples = {}
                        for old_slot_str, samples in old_samples.items():
                            old_slot = int(old_slot_str)
                            mean_val = sum(samples) / len(samples)
                            for i in range(factor):
                                new_slot = old_slot * factor + i
                                new_samples[str(new_slot)] = [mean_val]
                        self.stats = PvProfileStats(
                            samples_by_slot=new_samples,
                            observation_count=old_stats.get("observation_count", 0),
                            last_observation=old_stats.get("last_observation"),
                        )
                        return True
                    else:
                        self.log(
                            f"PV profile slot size changed "
                            f"({stored_slot_minutes}\u2192{self.slot_minutes}), "
                            f"cannot migrate"
                        )
                        return False
                if "stats" in data:
                    self.stats = PvProfileStats.from_dict(data["stats"])
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load PV profile data: {e}")
        return False
