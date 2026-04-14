from __future__ import annotations

import re


def extract_punctuation_features(text: str) -> dict[str, float]:
    words = text.split()
    word_count = len(words) or 1

    features: dict[str, float] = {}

    # Per-character punctuation frequencies normalized by word count
    punct_chars = {
        "comma": ",",
        "semicolon": ";",
        "colon": ":",
        "period": ".",
        "exclamation": "!",
        "question": "?",
        "open_paren": "(",
        "close_paren": ")",
        "single_quote": "'",
        "double_quote": '"',
    }
    for name, char in punct_chars.items():
        features[f"punct_{name}_rate"] = text.count(char) / word_count

    # Special patterns
    features["punct_dash_rate"] = (text.count(" - ") + text.count(" -- ")) / word_count
    features["punct_emdash_rate"] = text.count("\u2014") / word_count
    features["punct_ellipsis_rate"] = (text.count("...") + text.count("\u2026")) / word_count

    # Contraction frequency
    contractions = re.findall(r"\b\w+'\w+\b", text)
    features["punct_contraction_rate"] = len(contractions) / word_count

    # Total punctuation density
    total_punct = sum(1 for c in text if c in '.,;:!?()"\'-')
    features["punct_density"] = total_punct / word_count

    return features
