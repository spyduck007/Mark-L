"""Account-free, fully local wake-word support for MARK L.

The detector uses sherpa-onnx open-vocabulary keyword spotting. On first use it
downloads the official compact English/Chinese keyword model, extracts only the
low-latency INT8 files MARK L needs, and caches them under ``config/wake_words``.
After that, detection runs entirely on-device.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_dir()
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_NAME}.tar.bz2"
)
DEFAULT_MODEL_DIR = BASE_DIR / "config" / "wake_words" / MODEL_NAME

_ENCODER = "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
_DECODER = "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
_JOINER = "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
_REQUIRED_FILES = (_ENCODER, _DECODER, _JOINER, "tokens.txt", "en.phone")
_DOWNLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class WakeWordSettings:
    enabled: bool = True
    phrase: str = "Hey Jarvis"
    sensitivity: float = 0.55
    follow_up_timeout: float = 10.0
    pre_roll_ms: int = 750
    voice_rms_threshold: int = 420
    num_threads: int = 2

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

        return cls(
            enabled=bool(data.get("wake_word_enabled", True)),
            phrase=str(data.get("wake_phrase", "Hey Jarvis") or "Hey Jarvis").strip(),
            sensitivity=_float("wake_word_sensitivity", 0.55, 0.0, 1.0),
            follow_up_timeout=_float("follow_up_timeout_seconds", 10.0, 2.0, 60.0),
            pre_roll_ms=_int("wake_word_pre_roll_ms", 750, 250, 2000),
            voice_rms_threshold=_int("wake_word_voice_rms_threshold", 420, 50, 5000),
            num_threads=_int("wake_word_num_threads", 2, 1, 8),
        )

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG_PATH) -> "WakeWordSettings":
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return cls.from_mapping(data)

    @property
    def keyword_score(self) -> float:
        # Higher sensitivity gives the phrase more decoder bias.
        return 1.0 + 2.0 * self.sensitivity

    @property
    def keyword_threshold(self) -> float:
        # Lower acoustic threshold makes activation easier. At the default
        # sensitivity this is ~0.26, close to sherpa-onnx's documented default.
        return max(0.08, min(0.50, 0.48 - 0.40 * self.sensitivity))


class WakeWordConfigurationError(RuntimeError):
    """Raised when local keyword spotting cannot be initialized."""


def _model_is_ready(model_dir: Path) -> bool:
    return all(
        (model_dir / filename).is_file()
        and (model_dir / filename).stat().st_size > 0
        for filename in _REQUIRED_FILES
    )


def ensure_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Download and safely extract the compact sherpa keyword model once."""
    if _model_is_ready(model_dir):
        return model_dir

    with _DOWNLOAD_LOCK:
        if _model_is_ready(model_dir):
            return model_dir

        model_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WakeWord] downloading local model ({MODEL_NAME})...")

        try:
            with tempfile.TemporaryDirectory(
                prefix="markl-wake-", dir=str(model_dir.parent)
            ) as tmp_name:
                tmp = Path(tmp_name)
                archive_path = tmp / f"{MODEL_NAME}.tar.bz2"
                request = urllib.request.Request(
                    MODEL_URL,
                    headers={"User-Agent": "MARK-L wake-word installer"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    with archive_path.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)

                stage = tmp / MODEL_NAME
                stage.mkdir()
                with tarfile.open(archive_path, mode="r:bz2") as archive:
                    for filename in _REQUIRED_FILES:
                        member_name = f"{MODEL_NAME}/{filename}"
                        try:
                            member = archive.getmember(member_name)
                        except KeyError as exc:
                            raise WakeWordConfigurationError(
                                f"Downloaded wake model is missing {filename}."
                            ) from exc
                        source = archive.extractfile(member)
                        if source is None:
                            raise WakeWordConfigurationError(
                                f"Could not extract wake model file {filename}."
                            )
                        with source, (stage / filename).open("wb") as output:
                            shutil.copyfileobj(source, output)

                if not _model_is_ready(stage):
                    raise WakeWordConfigurationError("Wake model download was incomplete.")

                if model_dir.exists():
                    shutil.rmtree(model_dir)
                stage.rename(model_dir)
        except WakeWordConfigurationError:
            raise
        except Exception as exc:
            raise WakeWordConfigurationError(
                f"Could not download the local wake model: {exc}"
            ) from exc

        print(f"[WakeWord] local model ready: {model_dir}")
        return model_dir


class WakeWordDetector:
    """Streaming sherpa-onnx keyword spotter for arbitrary 16-bit PCM chunks."""

    sample_rate = 16000
    frame_length = 0  # Retained for compatibility; sherpa accepts arbitrary chunks.

    def __init__(
        self,
        settings: WakeWordSettings,
        model_dir: Path = DEFAULT_MODEL_DIR,
    ):
        if not settings.enabled:
            raise WakeWordConfigurationError("Wake-word mode is disabled.")

        try:
            import numpy as np
            import sherpa_onnx
        except ImportError as exc:
            raise WakeWordConfigurationError(
                "sherpa-onnx wake dependencies are missing. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.settings = settings
        self._np = np
        self.model_dir = ensure_model(Path(model_dir))
        self.keyword_path = self._write_keyword_file(sherpa_onnx)

        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(self.model_dir / "tokens.txt"),
                encoder=str(self.model_dir / _ENCODER),
                decoder=str(self.model_dir / _DECODER),
                joiner=str(self.model_dir / _JOINER),
                keywords_file=str(self.keyword_path),
                num_threads=settings.num_threads,
                sample_rate=self.sample_rate,
                provider="cpu",
            )
            self._stream = self._spotter.create_stream()
        except Exception as exc:
            raise WakeWordConfigurationError(
                f"Could not initialize sherpa-onnx keyword spotting: {exc}"
            ) from exc

    def _write_keyword_file(self, sherpa_onnx) -> Path:
        phrase = " ".join(self.settings.phrase.upper().split())
        if not phrase:
            raise WakeWordConfigurationError("Wake phrase cannot be empty.")

        try:
            encoded = sherpa_onnx.text2token(
                [phrase],
                tokens=str(self.model_dir / "tokens.txt"),
                tokens_type="phone+ppinyin",
                lexicon=str(self.model_dir / "en.phone"),
            )
        except Exception as exc:
            raise WakeWordConfigurationError(
                f"Could not encode wake phrase {self.settings.phrase!r}: {exc}"
            ) from exc

        if not encoded or not encoded[0]:
            raise WakeWordConfigurationError(
                f"Wake phrase contains unsupported words: {self.settings.phrase!r}"
            )

        slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "wake_word"
        original = re.sub(r"\s+", "_", phrase)
        keyword_path = self.model_dir / f"keyword-{slug}.txt"
        tokens = [str(token) for token in encoded[0]]
        tokens.extend(
            [
                f":{self.settings.keyword_score:.2f}",
                f"#{self.settings.keyword_threshold:.2f}",
                f"@{original}",
            ]
        )
        keyword_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
        return keyword_path

    def process_bytes(self, pcm_bytes: bytes) -> bool:
        if not pcm_bytes or len(pcm_bytes) < 2:
            return False

        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        samples = self._np.frombuffer(pcm_bytes[:usable], dtype="<i2")
        samples = samples.astype(self._np.float32) / 32768.0
        self._stream.accept_waveform(self.sample_rate, samples)

        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                self.reset()
                return True
        return False

    def reset(self) -> None:
        if getattr(self, "_spotter", None) is not None:
            self._stream = self._spotter.create_stream()

    def close(self) -> None:
        self._stream = None
        self._spotter = None

    def __enter__(self) -> "WakeWordDetector":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
