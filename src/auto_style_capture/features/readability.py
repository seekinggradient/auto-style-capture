from __future__ import annotations

import re


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    # Simple syllable counter
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(count, 1)


def extract_readability_features(text: str) -> dict[str, float]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)

    if not sentences or not words:
        return {}

    total_sentences = len(sentences)
    total_words = len(words)
    syllables = [_count_syllables(w) for w in words]
    total_syllables = sum(syllables)
    complex_words = sum(1 for s in syllables if s >= 3)

    features: dict[str, float] = {}

    # Average syllables per word
    features["read_syllables_per_word"] = total_syllables / total_words

    # Flesch-Kincaid Grade Level
    features["read_flesch_kincaid"] = (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )

    # Gunning Fog Index
    features["read_gunning_fog"] = 0.4 * (
        (total_words / total_sentences) + 100 * (complex_words / total_words)
    )

    # Coleman-Liau Index
    chars = sum(len(w) for w in words)
    L = (chars / total_words) * 100  # avg chars per 100 words
    S = (total_sentences / total_words) * 100  # avg sentences per 100 words
    features["read_coleman_liau"] = 0.0588 * L - 0.296 * S - 15.8

    # Complex word ratio
    features["read_complex_word_ratio"] = complex_words / total_words

    return features
