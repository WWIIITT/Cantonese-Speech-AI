import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class HelperTests(unittest.TestCase):
    def test_normalize_transcript_removes_punctuation_and_spaces(self):
        from cantonese_speech_ai.text import normalize_transcript

        self.assertEqual(normalize_transcript("  你好，WORLD！  "), "你好 world")

    def test_error_rates_use_reference_length(self):
        from cantonese_speech_ai.metrics import char_error_rate, word_error_rate

        self.assertEqual(char_error_rate("你好", "你"), 0.5)
        self.assertEqual(word_error_rate("nei hou", "nei ho"), 0.5)

    def test_project_paths_use_environment_defaults(self):
        from cantonese_speech_ai.config import ProjectPaths

        old_data = os.environ.pop("DATA_ROOT", None)
        old_cache = os.environ.pop("CACHE_ROOT", None)
        try:
            paths = ProjectPaths.from_env()
            self.assertEqual(paths.data_root, Path("data/raw"))
            self.assertEqual(paths.cache_root, Path("data/cache"))
        finally:
            if old_data is not None:
                os.environ["DATA_ROOT"] = old_data
            if old_cache is not None:
                os.environ["CACHE_ROOT"] = old_cache

    def test_dataset_registry_contains_cantonese_sources(self):
        from cantonese_speech_ai.datasets import DATASETS

        names = {dataset.name for dataset in DATASETS}
        self.assertIn("Mozilla Common Voice Yue", names)
        self.assertIn("MDCC", names)
        self.assertIn("WenetSpeech-Yue", names)

    def test_audio_duration_uses_frames_and_sample_rate(self):
        from cantonese_speech_ai.audio import duration_seconds

        self.assertEqual(duration_seconds(num_frames=32000, sample_rate=16000), 2.0)


if __name__ == "__main__":
    unittest.main()
