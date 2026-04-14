from __future__ import annotations

import math
import re
from collections import Counter


FUNCTION_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
]


def extract_lexical_features(text: str) -> dict[str, float]:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {}

    total_words = len(words)
    unique_words = set(words)
    vocab_size = len(unique_words)
    word_counts = Counter(words)

    features: dict[str, float] = {}

    # Type-Token Ratio
    features["lex_ttr"] = vocab_size / total_words if total_words else 0
    features["lex_cttr"] = vocab_size / math.sqrt(2 * total_words) if total_words else 0

    # Hapax legomena / dislegomena
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_dis = sum(1 for w, c in word_counts.items() if c == 2)
    features["lex_hapax_ratio"] = hapax / vocab_size if vocab_size else 0
    features["lex_hapax_dis_ratio"] = hapax_dis / vocab_size if vocab_size else 0

    # Yule's K
    freq_spectrum = Counter(word_counts.values())
    m1 = total_words
    m2 = sum(i * i * freq for i, freq in freq_spectrum.items())
    features["lex_yules_k"] = (10000 * (m2 - m1)) / (m1 * m1) if m1 > 0 else 0

    # Word length stats
    word_lengths = [len(w) for w in words]
    features["lex_mean_word_length"] = sum(word_lengths) / len(word_lengths)
    if len(word_lengths) > 1:
        mean = features["lex_mean_word_length"]
        variance = sum((l - mean) ** 2 for l in word_lengths) / (len(word_lengths) - 1)
        features["lex_word_length_std"] = math.sqrt(variance)
    else:
        features["lex_word_length_std"] = 0

    # Function word frequencies
    for fw in FUNCTION_WORDS:
        features[f"lex_fw_{fw}"] = word_counts.get(fw, 0) / total_words

    # Short vs long words
    features["lex_short_word_ratio"] = sum(1 for w in words if len(w) <= 3) / total_words
    features["lex_long_word_ratio"] = sum(1 for w in words if len(w) >= 7) / total_words

    return features
