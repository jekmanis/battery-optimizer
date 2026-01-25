"""
Load profile module for statistical load forecasting.

Tracks historical household load by time-of-day slots and provides
quantile-based forecasts for battery discharge planning.
"""

import datetime
import json
import math
from typing import List

from .models import LoadProfileStats


def _quantile(values: List[float], q: float) -> float:
    """Return the q-quantile (0..1) with linear interpolation."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    frac = pos - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


class LoadProfile:
    """
    Simple statistical load profile by time-of-day slots.
    Stores recent samples per slot and returns a quantile-based forecast.
    """

    def __init__(
        self,
        slot_minutes: int,
        default_load_w: float,
        max_samples: int = 60,
        min_samples: int = 6,
        log_func=None,
    ):
        self.slot_minutes = max(1, int(slot_minutes))
        self.slots_per_day = int(1440 / self.slot_minutes)
        self.default_load_w = float(default_load_w)
        self.max_samples = max(1, int(max_samples))
        self.min_samples = max(1, int(min_samples))
        self.log = log_func or print
        self.stats = LoadProfileStats()

    def _slot_index(self, dt: datetime.datetime) -> int:
        minutes = dt.hour * 60 + dt.minute
        return int(minutes // self.slot_minutes)

    def record(self, dt: datetime.datetime, load_w: float):
        """Record a load observation for the given time slot."""
        if load_w <= 0:
            return
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        samples.append(float(load_w))
        if len(samples) > self.max_samples:
            samples = samples[-self.max_samples:]
        self.stats.samples_by_slot[slot] = samples
        self.stats.observation_count += 1
        self.stats.last_observation = dt.isoformat()

    def predict_kw(self, dt: datetime.datetime, quantile: float = 0.75) -> float:
        """
        Predict expected load (kW) for a slot using stored samples.

        Uses quantile-based forecast blended with default based on confidence.
        """
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        if not samples:
            return self.default_load_w / 1000.0
        q_value = _quantile(samples, quantile)
        confidence = min(1.0, len(samples) / self.min_samples)
        blended = (self.default_load_w * (1 - confidence)) + (q_value * confidence)
        return max(0.0, blended) / 1000.0

    def to_json(self) -> str:
        """Serialize load profile for persistence."""
        data = {
            "version": 1,
            "slot_minutes": self.slot_minutes,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        """Load load profile from JSON. Returns True if successful."""
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                if data.get("slot_minutes") != self.slot_minutes:
                    self.log("Load profile slot size changed, ignoring saved data")
                    return False
                if "stats" in data:
                    self.stats = LoadProfileStats.from_dict(data["stats"])
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load load profile data: {e}")
        return False
