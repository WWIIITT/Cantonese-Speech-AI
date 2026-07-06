import re
import unicodedata


_PUNCT = re.compile(r"[^\w\s\u3400-\u9fff]+", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()
