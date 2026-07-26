"""Local audio conditioning, VAD, echo reference, and barge-in signals."""

from __future__ import annotations

import math
import shutil
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.int8.onnx"
)


@dataclass(frozen=True)
class VoiceEvent:
    audio: bytes
    speech_started: bool = False
    speech_ended: bool = False
    is_speech: bool = False
    rms: float = 0.0


def _download_once(url: str, destination: Path) -> Path:
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MARK-L voice engine"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError("downloaded VAD model is empty")
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def _resample_int16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    if not data or source_rate == target_rate:
        return data
    samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2")
    if samples.size < 2:
        return b""
    output_count = max(1, round(samples.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, output_count, endpoint=False)
    result = np.interp(new_x, old_x, samples.astype(np.float32))
    return np.clip(result, -32768, 32767).astype("<i2").tobytes()


class LocalVoiceActivityDetector:
    def __init__(
        self,
        model_path: Path,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence: float = 0.45,
        min_speech: float = 0.12,
    ):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is required for local VAD") from exc

        model_path = _download_once(SILERO_VAD_URL, Path(model_path))
        silero = sherpa_onnx.SileroVadModelConfig(
            model=str(model_path),
            threshold=float(threshold),
            min_silence_duration=float(min_silence),
            min_speech_duration=float(min_speech),
            window_size=512,
            max_speech_duration=60.0,
        )
        config = sherpa_onnx.VadModelConfig(
            silero_vad=silero,
            sample_rate=sample_rate,
            num_threads=1,
            provider="cpu",
        )
        if not config.validate():
            raise RuntimeError("invalid sherpa-onnx VAD configuration")
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._was_speech = False
        self.sample_rate = sample_rate

    def process(self, pcm_bytes: bytes) -> tuple[bool, bool, bool]:
        samples = np.frombuffer(pcm_bytes[: len(pcm_bytes) - len(pcm_bytes) % 2], dtype="<i2")
        if samples.size:
            self._vad.accept_waveform(samples.astype(np.float32) / 32768.0)
        is_speech = bool(self._vad.is_speech_detected())
        started = is_speech and not self._was_speech
        ended = self._was_speech and not is_speech
        self._was_speech = is_speech
        while not self._vad.empty():
            self._vad.pop()
        return started, ended, is_speech

    def reset(self) -> None:
        self._vad.reset()
        self._was_speech = False


class WebRtcAudioConditioner:
    """Optional WebRTC AEC/NS/AGC wrapper operating on 10 ms frames."""

    def __init__(self, sample_rate: int = 16000, stream_delay_ms: int = 80):
        from aec_audio_processing import AudioProcessor

        self.sample_rate = sample_rate
        self.frame_samples = sample_rate // 100
        self.frame_bytes = self.frame_samples * 2
        self._processor = AudioProcessor(
            enable_aec=True,
            enable_ns=True,
            ns_level=2,
            enable_agc=True,
            agc_mode=1,
            enable_vad=False,
        )
        self._processor.set_stream_format(sample_rate, 1, sample_rate, 1)
        self._processor.set_reverse_stream_format(sample_rate, 1)
        self._processor.set_stream_delay(stream_delay_ms)
        self._capture_buffer = bytearray()
        self._capture_output = bytearray()
        self._reverse_buffer = bytearray()
        self._lock = threading.RLock()

    def feed_playback(self, pcm_bytes: bytes, source_rate: int) -> None:
        data = _resample_int16(pcm_bytes, source_rate, self.sample_rate)
        with self._lock:
            self._reverse_buffer.extend(data)
            while len(self._reverse_buffer) >= self.frame_bytes:
                frame = bytes(self._reverse_buffer[: self.frame_bytes])
                del self._reverse_buffer[: self.frame_bytes]
                self._processor.process_reverse_stream(frame)

    def process_capture(self, pcm_bytes: bytes) -> bytes:
        with self._lock:
            requested = len(pcm_bytes)
            self._capture_buffer.extend(pcm_bytes)
            while len(self._capture_buffer) >= self.frame_bytes:
                frame = bytes(self._capture_buffer[: self.frame_bytes])
                del self._capture_buffer[: self.frame_bytes]
                processed = self._processor.process_stream(frame)
                self._capture_output.extend(bytes(processed))
            if len(self._capture_output) >= requested:
                output = bytes(self._capture_output[:requested])
                del self._capture_output[:requested]
                return output
            # Preserve real-time cadence during the first partial frame.
            missing = requested - len(self._capture_output)
            output = bytes(self._capture_output) + pcm_bytes[-missing:]
            self._capture_output.clear()
            return output


class AdaptiveEchoSuppressor:
    """Dependency-free playback-reference suppressor used when WebRTC is unavailable."""

    def __init__(self, sample_rate: int = 16000, stream_delay_ms: int = 80):
        from collections import deque

        self.sample_rate = sample_rate
        self.delay_samples = max(0, int(sample_rate * stream_delay_ms / 1000))
        self._reference = deque(maxlen=sample_rate * 4)
        self._lock = threading.RLock()

    def feed_playback(self, pcm_bytes: bytes, source_rate: int) -> None:
        data = _resample_int16(pcm_bytes, source_rate, self.sample_rate)
        samples = np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2")
        with self._lock:
            self._reference.extend(int(v) for v in samples)

    def process_capture(self, pcm_bytes: bytes) -> bytes:
        capture = np.frombuffer(
            pcm_bytes[: len(pcm_bytes) - len(pcm_bytes) % 2], dtype="<i2"
        ).astype(np.float32)
        if capture.size == 0:
            return pcm_bytes
        with self._lock:
            available = len(self._reference) - self.delay_samples
            if available < capture.size:
                return pcm_bytes
            reference = np.asarray(list(self._reference), dtype=np.float32)
            end = reference.size - self.delay_samples if self.delay_samples else reference.size
            reference = reference[end - capture.size : end]
        energy = float(np.dot(reference, reference))
        if energy < 1.0:
            return pcm_bytes
        gain = float(np.dot(capture, reference) / energy)
        gain = max(0.0, min(1.2, gain))
        cleaned = capture - gain * reference
        return np.clip(cleaned, -32768, 32767).astype("<i2").tobytes()


class VoiceAudioEngine:
    def __init__(
        self,
        model_dir: Path,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        aec_enabled: bool = True,
        stream_delay_ms: int = 80,
    ):
        self.sample_rate = sample_rate
        self._process_lock = threading.RLock()
        self.vad = LocalVoiceActivityDetector(
            Path(model_dir) / "silero_vad.int8.onnx",
            sample_rate=sample_rate,
            threshold=vad_threshold,
        )
        self.conditioner = None
        self.conditioner_error = None
        self.conditioner_backend = "disabled"
        if aec_enabled:
            try:
                self.conditioner = WebRtcAudioConditioner(sample_rate, stream_delay_ms)
                self.conditioner_backend = "webrtc"
            except Exception as exc:
                self.conditioner_error = str(exc)
                self.conditioner = AdaptiveEchoSuppressor(sample_rate, stream_delay_ms)
                self.conditioner_backend = "adaptive-fallback"

    def feed_playback(self, pcm_bytes: bytes, source_rate: int = 24000) -> None:
        if self.conditioner:
            self.conditioner.feed_playback(pcm_bytes, source_rate)

    def process_microphone(self, pcm_bytes: bytes) -> VoiceEvent:
        with self._process_lock:
            clean = self.conditioner.process_capture(pcm_bytes) if self.conditioner else pcm_bytes
            samples = np.frombuffer(clean[: len(clean) - len(clean) % 2], dtype="<i2")
            rms = float(math.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
            started, ended, is_speech = self.vad.process(clean)
            return VoiceEvent(clean, started, ended, is_speech, rms)

    def reset(self) -> None:
        with self._process_lock:
            self.vad.reset()
