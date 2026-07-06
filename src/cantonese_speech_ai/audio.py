from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    sample_rate: int
    num_frames: int

    @property
    def duration(self) -> float:
        return duration_seconds(self.num_frames, self.sample_rate)


def duration_seconds(num_frames: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return num_frames / sample_rate


def load_audio(path: str | Path, sample_rate: int = 16000) -> tuple[Any, int]:
    import librosa

    waveform, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, actual_rate


def inspect_audio(path: str | Path) -> AudioInfo:
    import soundfile as sf

    info = sf.info(path)
    return AudioInfo(Path(path), info.samplerate, info.frames)
