"""Optional, fully local owner-voice enrollment and verification."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np


MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"
)


def _download_model(path: Path) -> Path:
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "MARK-L speaker identity"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError("speaker model download was empty")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


class SpeakerIdentity:
    def __init__(
        self,
        model_path: Path,
        profile_path: Path,
        threshold: float = 0.65,
        sample_rate: int = 16000,
    ):
        self.model_path = Path(model_path)
        self.profile_path = Path(profile_path)
        self.threshold = max(0.1, min(0.95, float(threshold)))
        self.sample_rate = sample_rate
        self._extractor = None
        self._owner_embeddings: list[np.ndarray] = []
        self._load_profile()

    @property
    def enrolled(self) -> bool:
        return bool(self._owner_embeddings)

    def _load_profile(self) -> None:
        try:
            payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
            values = payload.get("owner_embeddings", [])
            self._owner_embeddings = [np.asarray(value, dtype=np.float32) for value in values]
            if "threshold" in payload:
                self.threshold = max(0.1, min(0.95, float(payload["threshold"])))
        except Exception:
            self._owner_embeddings = []

    def _ensure_extractor(self):
        if self._extractor is not None:
            return self._extractor
        import sherpa_onnx

        model = _download_model(self.model_path)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=2, provider="cpu", debug=False
        )
        if not config.validate():
            raise RuntimeError("invalid speaker embedding model configuration")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor

    def embedding(self, pcm_bytes: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm_bytes[: len(pcm_bytes) - len(pcm_bytes) % 2], dtype="<i2")
        if samples.size < self.sample_rate:
            raise ValueError("At least one second of clear speech is required.")
        extractor = self._ensure_extractor()
        stream = extractor.create_stream()
        stream.accept_waveform(self.sample_rate, samples.astype(np.float32) / 32768.0)
        stream.input_finished()
        if not extractor.is_ready(stream):
            raise ValueError("Not enough usable speech for speaker verification.")
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not norm:
            raise ValueError("Speaker embedding was empty.")
        return vector / norm

    def enroll(self, pcm_bytes: bytes) -> int:
        vector = self.embedding(pcm_bytes)
        self._owner_embeddings.append(vector)
        self._owner_embeddings = self._owner_embeddings[-5:]
        self._save_profile()
        return len(self._owner_embeddings)

    def verify(self, pcm_bytes: bytes) -> tuple[bool, float]:
        if not self.enrolled:
            return False, 0.0
        vector = self.embedding(pcm_bytes)
        score = max(float(np.dot(vector, owner)) for owner in self._owner_embeddings)
        return score >= self.threshold, score

    def clear(self) -> None:
        self._owner_embeddings = []
        self.profile_path.unlink(missing_ok=True)

    def _save_profile(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": self.threshold,
            "owner_embeddings": [embedding.tolist() for embedding in self._owner_embeddings],
        }
        tmp = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.profile_path)
