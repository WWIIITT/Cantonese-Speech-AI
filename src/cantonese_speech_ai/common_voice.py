import os
import csv
from dataclasses import dataclass
from pathlib import Path

from cantonese_speech_ai.text import normalize_transcript


DEFAULT_YUE_ROOT = Path("Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue")
SPLITS = {"train", "dev", "test", "validated", "other", "invalidated"}


def _project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    candidates = [
        Path(env_root) if env_root else None,
        Path.cwd(),
        *Path.cwd().parents,
        Path("D:/GitHub/Cantonese-Speech-AI"),
        Path("D:/GitHub/Cantonese-Speech-AI").resolve(),
        Path("/mnt/d/GitHub/Cantonese-Speech-AI"),
        Path("/content/Cantonese-Speech-AI"),
        Path("/content/drive/MyDrive/Cantonese-Speech-AI"),
        Path("/content/drive/My Drive/Cantonese-Speech-AI"),
    ]
    for path in candidates:
        if path is None:
            continue
        if (path / "src" / "cantonese_speech_ai").exists():
            return path
    return Path.cwd()


def _dotenv_value(root: Path, key: str) -> str | None:
    env_path = root / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("\"'")
    return None


@dataclass(frozen=True)
class CommonVoicePaths:
    root: Path

    @classmethod
    def from_env(cls) -> "CommonVoicePaths":
        project_root = _project_root()
        value = os.getenv("COMMON_VOICE_YUE_ROOT") or _dotenv_value(
            project_root, "COMMON_VOICE_YUE_ROOT"
        )
        root = Path(value) if value else DEFAULT_YUE_ROOT
        if not root.is_absolute():
            root = project_root / root
        return cls(root)

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def durations_path(self) -> Path:
        return self.root / "clip_durations.tsv"

    def split_path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return self.root / f"{split}.tsv"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_split(split: str, paths: CommonVoicePaths | None = None) -> list[dict[str, str]]:
    paths = paths or CommonVoicePaths.from_env()
    return _read_tsv(paths.split_path(split))


def read_clip_durations(paths: CommonVoicePaths | None = None) -> dict[str, int]:
    paths = paths or CommonVoicePaths.from_env()
    rows = _read_tsv(paths.durations_path)
    return {row["clip"]: int(row["duration[ms]"]) for row in rows}


def prepare_split(split: str, paths: CommonVoicePaths | None = None) -> list[dict[str, object]]:
    paths = paths or CommonVoicePaths.from_env()
    durations = read_clip_durations(paths)
    prepared = []
    for row in read_split(split, paths):
        duration_ms = durations.get(row["path"])
        item = dict(row)
        item["audio_path"] = paths.clips_dir / row["path"]
        item["duration_ms"] = duration_ms
        item["duration_sec"] = duration_ms / 1000 if duration_ms is not None else None
        item["normalized_sentence"] = normalize_transcript(row.get("sentence", ""))
        prepared.append(item)
    return prepared


def count_missing_audio(rows: list[dict[str, object]]) -> int:
    return sum(1 for row in rows if not Path(row["audio_path"]).exists())
