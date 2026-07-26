"""Adaptive wake-word calibration from local acoustic observations."""

from __future__ import annotations

import json
import statistics
import threading
from collections import deque
from pathlib import Path


class WakeCalibration:
    """Learns a device noise floor and recommends conservative sensitivity changes."""

    def __init__(self, path: Path, sensitivity: float = 0.55):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.ambient_rms = deque(maxlen=500)
        self.wake_rms = deque(maxlen=50)
        self.false_wakes = 0
        self.sensitivity = float(sensitivity)
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.ambient_rms.extend(float(v) for v in data.get("ambient_rms", []))
            self.wake_rms.extend(float(v) for v in data.get("wake_rms", []))
            self.false_wakes = int(data.get("false_wakes", 0))
            self.sensitivity = float(data.get("sensitivity", self.sensitivity))
        except Exception:
            pass

    def observe_ambient(self, rms: float) -> None:
        if 0 <= rms < 12000:
            with self._lock:
                self.ambient_rms.append(round(float(rms), 2))

    def observe_wake(self, rms: float) -> None:
        if rms > 0:
            with self._lock:
                self.wake_rms.append(round(float(rms), 2))
                self._save_locked()

    def mark_false_wake(self) -> float:
        with self._lock:
            self.false_wakes += 1
            self.sensitivity = max(0.15, self.sensitivity - 0.05)
            self._save_locked()
            return self.sensitivity

    def recommended_sensitivity(self) -> float:
        with self._lock:
            if not self.ambient_rms or not self.wake_rms:
                return round(self.sensitivity, 2)
            ambient = statistics.median(self.ambient_rms)
            wake = statistics.median(self.wake_rms)
            ratio = wake / max(ambient, 1.0)
            if self.false_wakes >= 2 or ratio < 2.0:
                return round(max(0.2, self.sensitivity - 0.08), 2)
            if ratio > 6.0 and self.false_wakes == 0:
                return round(min(0.85, self.sensitivity + 0.04), 2)
            return round(self.sensitivity, 2)

    def summary(self) -> dict:
        with self._lock:
            ambient = statistics.median(self.ambient_rms) if self.ambient_rms else None
            wake = statistics.median(self.wake_rms) if self.wake_rms else None
            return {
                "ambient_samples": len(self.ambient_rms),
                "wake_samples": len(self.wake_rms),
                "ambient_rms_median": round(ambient, 2) if ambient is not None else None,
                "wake_rms_median": round(wake, 2) if wake is not None else None,
                "false_wakes": self.false_wakes,
                "current_sensitivity": round(self.sensitivity, 2),
                "recommended_sensitivity": self.recommended_sensitivity(),
            }

    def reset(self) -> None:
        with self._lock:
            self.ambient_rms.clear()
            self.wake_rms.clear()
            self.false_wakes = 0
            self.path.unlink(missing_ok=True)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ambient_rms": list(self.ambient_rms),
            "wake_rms": list(self.wake_rms),
            "false_wakes": self.false_wakes,
            "sensitivity": self.sensitivity,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
