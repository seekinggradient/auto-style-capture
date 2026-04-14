from __future__ import annotations

import math
import re


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def extract_syntactic_features(text: str) -> dict[str, float]:
    sentences = _split_sentences(text)
    if not sentences:
        return {}

    features: dict[str, float] = {}

    # Sentence length distribution
    sent_lengths = [len(s.split()) for s in sentences]
    n = len(sent_lengths)
    mean = sum(sent_lengths) / n
    features["syn_sent_length_mean"] = mean
    features["syn_sent_length_median"] = sorted(sent_lengths)[n // 2]

    if n > 1:
        variance = sum((l - mean) ** 2 for l in sent_lengths) / (n - 1)
        std = math.sqrt(variance)
        features["syn_sent_length_std"] = std
        # Skewness
        if std > 0:
            features["syn_sent_length_skew"] = (
                sum((l - mean) ** 3 for l in sent_lengths) / (n * std**3)
            )
        else:
            features["syn_sent_length_skew"] = 0
    else:
        features["syn_sent_length_std"] = 0
        features["syn_sent_length_skew"] = 0

    # Percentiles
    sorted_lengths = sorted(sent_lengths)
    features["syn_sent_length_p10"] = sorted_lengths[max(0, int(n * 0.1))]
    features["syn_sent_length_p90"] = sorted_lengths[min(n - 1, int(n * 0.9))]

    # Sentence type ratios
    total_sents = len(sentences)
    features["syn_question_ratio"] = sum(1 for s in sentences if s.endswith("?")) / total_sents
    features["syn_exclamation_ratio"] = sum(1 for s in sentences if s.endswith("!")) / total_sents
    features["syn_declarative_ratio"] = sum(1 for s in sentences if s.endswith(".")) / total_sents

    # Short sentence ratio (fragments, punchy style)
    features["syn_short_sent_ratio"] = sum(1 for l in sent_lengths if l <= 5) / total_sents
    features["syn_long_sent_ratio"] = sum(1 for l in sent_lengths if l >= 25) / total_sents

    # Paragraph stats
    paragraphs = _split_paragraphs(text)
    if paragraphs:
        para_lengths = [len(p.split()) for p in paragraphs]
        features["syn_para_length_mean"] = sum(para_lengths) / len(para_lengths)
        para_sents = [len(_split_sentences(p)) for p in paragraphs]
        features["syn_sents_per_para_mean"] = sum(para_sents) / len(para_sents)
    else:
        features["syn_para_length_mean"] = 0
        features["syn_sents_per_para_mean"] = 0

    # Total sentence count (useful as context)
    features["syn_total_sentences"] = total_sents

    return features
