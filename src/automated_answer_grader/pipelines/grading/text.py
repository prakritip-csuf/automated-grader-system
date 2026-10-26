import re

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

def normalize_text(text: str) -> str:
    return _WHITESPACE.sub("", text.strip())

def split_answer_units(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    units = [
        normalize_text(part)
        for part in _SENTENCE_BOUNDARY.split(cleaned)
        if part.strip()
    ]
    return units or [normalize_text(cleaned)]

