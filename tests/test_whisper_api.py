import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cantonese_speech_ai.whisper_api import (
    WhisperApiConfig,
    build_prediction_row,
    transcribe_audio,
)


class FakeTranscriptions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"text": "你好世界"})()


class FakeClient:
    def __init__(self):
        self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()


class WhisperApiTests(unittest.TestCase):
    def test_config_reads_vectorengine_environment(self):
        env = {
            "VECTORENGINE_API_KEY": "test-key",
            "VECTORENGINE_BASE_URL": "https://api.vectorengine.cn/v1",
            "WHISPER_MODEL": "whisper-1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = WhisperApiConfig.from_env()

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.vectorengine.cn/v1")
        self.assertEqual(config.model, "whisper-1")

    def test_transcribe_audio_calls_client_with_file_and_model(self):
        client = FakeClient()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"audio")
            audio_path = Path(tmp.name)

        try:
            text = transcribe_audio(client, audio_path, model="whisper-1")
        finally:
            audio_path.unlink(missing_ok=True)

        self.assertEqual(text, "你好世界")
        call = client.audio.transcriptions.calls[0]
        self.assertEqual(call["model"], "whisper-1")
        self.assertEqual(Path(call["file"].name), audio_path)

    def test_build_prediction_row_adds_normalized_text_and_metrics(self):
        row = {
            "path": "sample.mp3",
            "audio_path": "clips/sample.mp3",
            "sentence": "你好，世界！",
            "normalized_sentence": "你好世界",
        }

        result = build_prediction_row(row, "你好 世界")

        self.assertEqual(result["prediction"], "你好 世界")
        self.assertEqual(result["normalized_prediction"], "你好 世界")
        self.assertIn("cer", result)
        self.assertIn("wer", result)


if __name__ == "__main__":
    unittest.main()
