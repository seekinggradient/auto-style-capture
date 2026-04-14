from __future__ import annotations

import re
from collections import Counter


def extract_ngram_features(text: str, top_n: int = 20) -> dict[str, float]:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 3:
        return {}

    features: dict[str, float] = {}
    total_words = len(words)

    # Word bigrams
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    total_bigrams = len(bigrams) or 1
    for bigram, count in bigram_counts.most_common(top_n):
        features[f"ngram_bi_{bigram}"] = count / total_bigrams

    # Word trigrams
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    total_trigrams = len(trigrams) or 1
    for trigram, count in trigram_counts.most_common(top_n):
        features[f"ngram_tri_{trigram}"] = count / total_trigrams

    # Character trigrams (captures rhythm and common patterns)
    clean_text = text.lower()
    char_trigrams = [clean_text[i : i + 3] for i in range(len(clean_text) - 2)]
    char_tri_counts = Counter(char_trigrams)
    total_char_tri = len(char_trigrams) or 1
    for ct, count in char_tri_counts.most_common(top_n):
        safe_key = ct.replace(" ", "SPC").replace("\n", "NL")
        features[f"ngram_char3_{safe_key}"] = count / total_char_tri

    # Bigram diversity (unique bigrams / total bigrams)
    features["ngram_bigram_diversity"] = len(bigram_counts) / total_bigrams

    return features
