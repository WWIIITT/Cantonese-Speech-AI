from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from .metrics import char_error_rate, word_error_rate
from .text import normalize_transcript


@dataclass(frozen=True)
class WhisperApiConfig:
    api_key: str
    base_url: str = "https://api.vectorengine.cn/v1"
    model: str = "whisper-1"

    @classmethod
    def from_env(cls) -> "WhisperApiConfig":
        api_key = os.getenv("VECTORENGINE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set VECTORENGINE_API_KEY in .env before calling Whisper API.")
        return cls(
            api_key=api_key,
            base_url=os.getenv("VECTORENGINE_BASE_URL", cls.base_url).strip(),
            model=os.getenv("WHISPER_MODEL", cls.model).strip(),
        )


def transcribe_audio(client, audio_path: str | Path, model: str) -> str:
    with Path(audio_path).open("rb") as audio_file:
        result = client.audio.transcriptions.create(model=model, file=audio_file)
    return result.text


def build_prediction_row(row: Mapping[str, object], prediction: str) -> MutableMapping[str, object]:
    reference = str(row.get("normalized_sentence") or normalize_transcript(str(row.get("sentence", ""))))
    normalized_prediction = normalize_transcript(prediction)

    result = dict(row)
    result["prediction"] = prediction
    result["normalized_prediction"] = normalized_prediction
    result["cer"] = char_error_rate(reference, normalized_prediction)
    result["wer"] = word_error_rate(reference, normalized_prediction)
    return result
