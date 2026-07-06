import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class CommonVoiceTests(unittest.TestCase):
    def test_paths_default_to_local_yue_dataset(self):
        from cantonese_speech_ai.common_voice import CommonVoicePaths

        old_root = os.environ.pop("COMMON_VOICE_YUE_ROOT", None)
        try:
            paths = CommonVoicePaths.from_env()
            self.assertEqual(
                paths.root.name,
                "yue",
            )
            self.assertEqual(paths.clips_dir, paths.root / "clips")
            self.assertEqual(paths.split_path("train"), paths.root / "train.tsv")
        finally:
            if old_root is not None:
                os.environ["COMMON_VOICE_YUE_ROOT"] = old_root

    def test_paths_resolve_from_notebooks_working_directory(self):
        from cantonese_speech_ai.common_voice import CommonVoicePaths

        old_cwd = Path.cwd()
        old_root = os.environ.pop("COMMON_VOICE_YUE_ROOT", None)
        try:
            os.chdir(ROOT / "notebooks")
            paths = CommonVoicePaths.from_env()
            exists_from_notebook = paths.split_path("train").exists()

            os.chdir(ROOT.parent)
            parent_paths = CommonVoicePaths.from_env()
            exists_from_parent = parent_paths.split_path("train").exists()
        finally:
            os.chdir(old_cwd)
            if old_root is not None:
                os.environ["COMMON_VOICE_YUE_ROOT"] = old_root

        self.assertTrue(exists_from_notebook)
        self.assertTrue(exists_from_parent)

    def test_read_split_loads_expected_columns(self):
        from cantonese_speech_ai.common_voice import read_split

        rows = read_split("train")
        first = rows[0]
        for column in ("path", "sentence", "client_id", "accents"):
            self.assertIn(column, first)
        self.assertGreater(len(rows), 0)

    def test_prepare_split_adds_paths_durations_and_normalized_text(self):
        from cantonese_speech_ai.common_voice import prepare_split

        rows = prepare_split("train")
        first = rows[0]
        for column in (
            "audio_path",
            "duration_ms",
            "duration_sec",
            "normalized_sentence",
        ):
            self.assertIn(column, first)
        self.assertTrue(str(first["audio_path"]).endswith(".mp3"))
        self.assertGreater(first["duration_sec"], 0)

    def test_missing_audio_count_is_reported_without_raising(self):
        from cantonese_speech_ai.common_voice import count_missing_audio, prepare_split

        rows = [row.copy() for row in prepare_split("train")[:5]]
        rows[0]["audio_path"] = Path("missing/audio.mp3")
        self.assertGreaterEqual(count_missing_audio(rows), 1)


if __name__ == "__main__":
    unittest.main()
