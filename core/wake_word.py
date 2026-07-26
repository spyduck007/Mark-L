"""Local wake-word support for MARK L.

Porcupine runs entirely on-device after its keyword model has been created.
The first custom model creation may contact Picovoice; the resulting .ppn file
is cached under config/wake_words and reused on subsequent launches.
"""

from __future__ import annotations

import json
import os
import re
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_dir()
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
DEFAULT_MODEL_DIR = BASE_DIR / "config" / "wake_words"


@dataclass(frozen=True)
class WakeWordSettings:
    enabled: bool = True
    phrase: str = "Hey Jarvis"
    access_key: str = ""
    keyword_path: str = ""
    sensitivity: float = 0.55
    follow_up_timeout: float = 10.0
    pre_roll_ms: int = 750
    language: str = "en"
    voice_rms_threshold: int = 420

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "WakeWordSettings":
        data = data or {}

        def _float(key: str, default: float, low: float, high: float) -> float:
            try:
                value = float(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(low, min(high, value))

        def _int(key: str, default: int, low: int, high: int) -> int:
            try:
                value = int(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(low, min(high, value))

        phrase = str(data.get("wake_phrase", "Hey Jarvis") or "Hey Jarvis").strip()
        access_key = str(
            data.get("picovoice_access_key", "")
            or os.environ.get("PICOVOICE_ACCESS_KEY", "")
        ).strip()

        return cls(
            enabled=bool(data.get("wake_word_enabled", True)),
            phrase=phrase,
            access_key=access_key,
            keyword_path=str(data.get("wake_word_model_path", "") or "").strip(),
            sensitivity=_float("wake_word_sensitivity", 0.55, 0.0, 1.0),
            follow_up_timeout=_float("follow_up_timeout_seconds", 10.0, 2.0, 60.0),
            pre_roll_ms=_int("wake_word_pre_roll_ms", 750, 250, 2000),
            language=str(data.get("wake_word_language", "en") or "en").strip()[:2],
            voice_rms_threshold=_int("wake_word_voice_rms_threshold", 420, 50, 5000),
        )

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> "WakeWordSettings":
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return cls.from_mapping(data)

    @property
    def configured(self) -> bool:
        return bool(self.access_key)


class WakeWordConfigurationError(RuntimeError):
    """Raised when wake-word mode is enabled but cannot be configured."""


class WakeWordDetector:
    """Thin buffered adapter around ``pvporcupine.Porcupine``."""

    def __init__(self, settings: WakeWordSettings):
        if not settings.enabled:
            raise WakeWordConfigurationError("Wake-word mode is disabled.")
        if not settings.access_key:
            raise WakeWordConfigurationError(
                "A Picovoice AccessKey is required. Open Wake Word Settings to add one."
            )

        try:
            import pvporcupine
        except ImportError as exc:
            raise WakeWordConfigurationError(
                "pvporcupine is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.settings = settings
        self._pvporcupine = pvporcupine
        keyword_path = self._resolve_keyword_path(settings)
        self.keyword_path = keyword_path

        self._engine = pvporcupine.create(
            access_key=settings.access_key,
            keyword_paths=[str(keyword_path)],
            sensitivities=[settings.sensitivity],
        )
        self.sample_rate = int(self._engine.sample_rate)
        self.frame_length = int(self._engine.frame_length)
        self._byte_buffer = bytearray()
        self._frame_bytes = self.frame_length * 2

    def _resolve_keyword_path(self, settings: WakeWordSettings) -> Path:
        if settings.keyword_path:
            path = Path(settings.keyword_path).expanduser()
            if not path.is_absolute():
                path = BASE_DIR / path
            path = path.resolve()
            if not path.exists():
                raise WakeWordConfigurationError(f"Wake-word model not found: {path}")
            return path

        slug = re.sub(r"[^a-z0-9]+", "_", settings.phrase.lower()).strip("_")
        model_dir = DEFAULT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        path = (model_dir / f"{slug or 'wake_word'}.ppn").resolve()
        if not path.exists():
            self._pvporcupine.train_wake_word_from_phrase(
                settings.access_key,
                str(path),
                settings.language,
                settings.phrase,
            )
        return path

    def process_bytes(self, pcm_bytes: bytes) -> bool:
        """Process arbitrary 16-bit mono PCM chunks and report a detection."""
        if not pcm_bytes:
            return False
        self._byte_buffer.extend(pcm_bytes)

        detected = False
        while len(self._byte_buffer) >= self._frame_bytes:
            frame_bytes = bytes(self._byte_buffer[: self._frame_bytes])
            del self._byte_buffer[: self._frame_bytes]
            pcm = array("h")
            pcm.frombytes(frame_bytes)
            if sys.byteorder != "little":
                pcm.byteswap()
            if self._engine.process(pcm) >= 0:
                detected = True
                self._byte_buffer.clear()
                break
        return detected

    def reset(self) -> None:
        self._byte_buffer.clear()

    def close(self) -> None:
        engine = getattr(self, "_engine", None)
        self._engine = None
        if engine is not None:
            engine.delete()

    def __enter__(self) -> "WakeWordDetector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
