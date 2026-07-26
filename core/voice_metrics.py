"""Persistent voice-interaction quality metrics for local tuning."""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import deque
from pathlib import Path


class VoiceMetrics:
    def __init__(self, path: Path, max_samples: int = 250):
        self.path = Path(path)
        self.max_samples = max_samples
        self._lock = threading.RLock()
        self._data = {
            "wake_attempts": 0,
            "wake_detections": 0,
            "false_wakes": 0,
            "interruptions": 0,
            "reconnects": 0,
            "vad_starts": 0,
            "vad_ends": 0,
            "wake_latency_ms": [],
            "barge_in_latency_ms": [],
            "speech_to_response_ms": [],
            "updated_at": None,
        }
        self._load()

    def _load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data.update(loaded)
        except Exception:
            pass
        for key in ("wake_latency_ms", "barge_in_latency_ms", "speech_to_response_ms"):
            self._data[key] = list(self._data.get(key, []))[-self.max_samples :]

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._data[name] = int(self._data.get(name, 0)) + amount
            self._save_locked()

    def observe(self, name: str, value_ms: float) -> None:
        if value_ms < 0 or value_ms > 120000:
            return
        with self._lock:
            samples = deque(self._data.get(name, []), maxlen=self.max_samples)
            samples.append(round(float(value_ms), 2))
            self._data[name] = list(samples)
            self._save_locked()

    def summary(self) -> dict:
        with self._lock:
            result = dict(self._data)
        for key in ("wake_latency_ms", "barge_in_latency_ms", "speech_to_response_ms"):
            vals = [float(v) for v in result.get(key, [])]
            result[f"{key}_avg"] = round(statistics.fmean(vals), 2) if vals else None
            result[f"{key}_p95"] = round(sorted(vals)[max(0, int(len(vals) * 0.95) - 1)], 2) if vals else None
        return result

    def _save_locked(self) -> None:
        self._data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
