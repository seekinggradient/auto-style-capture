from __future__ import annotations

import logging

from .lexical import extract_lexical_features
from .syntactic import extract_syntactic_features
from .punctuation import extract_punctuation_features
from .ngrams import extract_ngram_features
from .readability import extract_readability_features
from .models import StyleProfile

logger = logging.getLogger(__name__)


def extract_features(text: str) -> StyleProfile:
    """Extract all stylometric features from a text."""
    features: dict[str, float] = {}

    extractors = [
        ("lexical", extract_lexical_features),
        ("syntactic", extract_syntactic_features),
        ("punctuation", extract_punctuation_features),
        ("readability", extract_readability_features),
    ]

    for name, extractor in extractors:
        try:
            result = extractor(text)
            features.update(result)
        except Exception:
            logger.warning("Feature extractor '%s' failed", name, exc_info=True)

    # N-grams extracted separately (they're corpus-specific, not universal features)
    # We don't include them in the ML classifier features to avoid high dimensionality
    # but they're available for the style profile summary

    return StyleProfile(features=features)


def extract_corpus_features(texts: list[str]) -> StyleProfile:
    """Extract averaged stylometric features across multiple texts."""
    if not texts:
        return StyleProfile()

    all_profiles = [extract_features(text) for text in texts]

    # Get union of all feature names
    all_names: set[str] = set()
    for profile in all_profiles:
        all_names.update(profile.features.keys())

    # Average across all texts
    averaged: dict[str, float] = {}
    for name in all_names:
        values = [p.features.get(name, 0.0) for p in all_profiles]
        averaged[name] = sum(values) / len(values)

    return StyleProfile(features=averaged)


def get_stable_feature_names() -> list[str]:
    """Return the list of stable feature names used for ML classification.

    These are features that are consistent across texts regardless of content,
    excluding n-gram features which are corpus-specific.
    """
    # Extract features from a dummy text to discover all feature names
    dummy = "The quick brown fox jumps over the lazy dog. She said hello to him, but he did not reply."
    profile = extract_features(dummy)
    return profile.feature_names
