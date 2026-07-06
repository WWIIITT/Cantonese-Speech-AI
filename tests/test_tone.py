import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ToneTests(unittest.TestCase):
    def test_count_colloquial_markers_counts_phrases_and_chars(self):
        from cantonese_speech_ai.tone import count_colloquial_markers

        counts = count_colloquial_markers("我哋唔知乜嘢係啱嘅")

        self.assertEqual(counts["哋"], 1)
        self.assertEqual(counts["唔"], 1)
        self.assertEqual(counts["乜嘢"], 1)
        self.assertEqual(counts["嘅"], 1)

    def test_flag_written_replacements_detects_reference_to_prediction_shift(self):
        from cantonese_speech_ai.tone import flag_written_replacements

        flags = flag_written_replacements("我哋唔嚟呢度", "我們不來這裡")

        self.assertIn("哋->們", flags)
        self.assertIn("唔->不", flags)
        self.assertIn("嚟->來", flags)
        self.assertIn("呢->這", flags)

    def test_characters_to_jyutping_summary_returns_tone_numbers(self):
        from cantonese_speech_ai.tone import jyutping_summary

        summary = jyutping_summary("你鍾唔鍾意男仔")

        self.assertIn("nei5", summary["jyutping"])
        self.assertIn("zung1", summary["jyutping"])
        self.assertIn("5", summary["tones"])
        self.assertIn("1", summary["tones"])

    def test_tone_extraction_handles_unknown_text_safely(self):
        from cantonese_speech_ai.tone import extract_tone_numbers

        self.assertEqual(extract_tone_numbers("abc ?"), [])


if __name__ == "__main__":
    unittest.main()
