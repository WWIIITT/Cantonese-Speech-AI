import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    cache_root: Path
    model_cache_dir: Path
    common_voice_yue_root: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        return cls(
            data_root=Path(os.getenv("DATA_ROOT", "data/raw")),
            cache_root=Path(os.getenv("CACHE_ROOT", "data/cache")),
            model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", ".cache/models")),
            common_voice_yue_root=Path(
                os.getenv(
                    "COMMON_VOICE_YUE_ROOT",
                    "Mozilla-HK-Speech-datasets/cv-corpus-26.0-2026-06-12/yue",
                )
            ),
        )
