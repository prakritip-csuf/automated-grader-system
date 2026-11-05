import re


# Match repeated whitespace, including newlines.
_WHITESPACE = re.compile(r"\s+")

# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    # Normalize whitespace while preserving sentence meaning.

    if not text:
        return ""

    return _WHITESPACE.sub(" ", text.strip())


def split_answer_units(text: str) -> list[str]:
    # Split a student answer into sentence-like units.

    cleaned = normalize_text(text)

    if not cleaned:
        return []

    units = [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(cleaned) if sentence.strip()]

    return units or [cleaned]
