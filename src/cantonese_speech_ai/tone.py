from __future__ import annotations

import re
from collections import Counter

import pycantonese


COLLOQUIAL_MARKERS = ("乜嘢", "嘅", "哋", "嗰", "唔", "咗", "畀", "俾", "嚟", "呢")

WRITTEN_REPLACEMENTS = {
    "嘅": ("的",),
    "哋": ("們", "们"),
    "嗰": ("那",),
    "唔": ("不",),
    "咗": ("了",),
    "畀": ("給", "给"),
    "俾": ("給", "给"),
    "嚟": ("來", "来"),
    "乜嘢": ("什麼", "什么"),
    "呢": ("這", "这", "那"),
}

_TONE = re.compile(r"[1-6]")


def count_colloquial_markers(text: str) -> dict[str, int]:
    return {marker: text.count(marker) for marker in COLLOQUIAL_MARKERS}


def flag_written_replacements(reference: str, prediction: str) -> list[str]:
    flags: list[str] = []
    for marker, replacements in WRITTEN_REPLACEMENTS.items():
        if marker not in reference:
            continue
        for replacement in replacements:
            if replacement in prediction:
                flags.append(f"{marker}->{replacement}")
    return flags


def characters_to_jyutping_text(text: str) -> str:
    pairs = pycantonese.characters_to_jyutping(text)
    readings = [reading for _, reading in pairs if reading]
    return " ".join(readings)


def extract_tone_numbers(jyutping_text: str) -> list[str]:
    return _TONE.findall(jyutping_text)


def jyutping_summary(text: str) -> dict[str, object]:
    jyutping = characters_to_jyutping_text(text)
    tones = extract_tone_numbers(jyutping)
    return {
        "jyutping": jyutping,
        "tones": tones,
        "tone_counts": dict(Counter(tones)),
    }
