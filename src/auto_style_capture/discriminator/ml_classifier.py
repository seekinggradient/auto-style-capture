from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..features.extractor import extract_features

logger = logging.getLogger(__name__)

# Human-readable descriptions for stylometric features
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Lexical
    "lex_ttr": "vocabulary diversity (unique words / total words)",
    "lex_cttr": "corrected vocabulary diversity (adjusted for text length)",
    "lex_hapax_ratio": "proportion of words used only once (higher = more varied vocabulary)",
    "lex_hapax_dis_ratio": "proportion of words used exactly twice",
    "lex_yules_k": "vocabulary constancy (lower = more repetitive word use)",
    "lex_mean_word_length": "average word length in characters",
    "lex_word_length_std": "word length variation (std dev)",
    "lex_short_word_ratio": "proportion of short words (1-3 letters: a, I, the, and, but...)",
    "lex_long_word_ratio": "proportion of long words (7+ letters: important, experience...)",
    # Syntax
    "syn_sent_length_mean": "average sentence length in words",
    "syn_sent_length_median": "median sentence length in words",
    "syn_sent_length_std": "sentence length variation",
    "syn_sent_length_skew": "sentence length skew (positive = mostly short with some long)",
    "syn_sent_length_p10": "10th percentile sentence length (shortest sentences)",
    "syn_sent_length_p90": "90th percentile sentence length (longest sentences)",
    "syn_question_ratio": "proportion of sentences that are questions",
    "syn_exclamation_ratio": "proportion of sentences with exclamation marks",
    "syn_declarative_ratio": "proportion of plain declarative sentences",
    "syn_short_sent_ratio": "proportion of short sentences (5 words or fewer)",
    "syn_long_sent_ratio": "proportion of long sentences (25+ words)",
    "syn_para_length_mean": "average paragraph length in words",
    "syn_sents_per_para_mean": "average sentences per paragraph",
    "syn_total_sentences": "total sentence count in the passage",
    # Punctuation
    "punct_comma_rate": "commas per word",
    "punct_semicolon_rate": "semicolons per word",
    "punct_colon_rate": "colons per word",
    "punct_period_rate": "periods per word",
    "punct_exclamation_rate": "exclamation marks per word",
    "punct_question_rate": "question marks per word",
    "punct_dash_rate": "hyphens/dashes per word",
    "punct_emdash_rate": "em dashes (\u2014) per word",
    "punct_ellipsis_rate": "ellipses (...) per word",
    "punct_contraction_rate": "contractions per word (don't, it's, can't...)",
    "punct_density": "total punctuation marks per word",
    "punct_open_paren_rate": "opening parentheses per word",
    "punct_close_paren_rate": "closing parentheses per word",
    "punct_double_quote_rate": 'double quote marks per word',
    "punct_single_quote_rate": "single quote/apostrophe marks per word",
    # Readability
    "read_flesch_kincaid": "Flesch-Kincaid grade level (8 = 8th grade, 12 = college)",
    "read_gunning_fog": "Gunning Fog index (years of education needed)",
    "read_coleman_liau": "Coleman-Liau readability index",
    "read_syllables_per_word": "average syllables per word",
    "read_complex_word_ratio": "proportion of words with 3+ syllables",
}


def _describe_feature(name: str) -> str:
    """Get human-readable description for a feature name."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    # Function word features
    if name.startswith("lex_fw_"):
        word = name.replace("lex_fw_", "")
        return f'frequency of the word "{word}"'
    return name


@dataclass
class MLResult:
    accuracy: float  # kept for interface compat: mapped from effect size (1.0 = large gap, 0.5 = chance)
    feature_importances: dict[str, float]
    feedback: str


class MLDiscriminator:
    def evaluate(self, real_texts: list[str], generated_texts: list[str]) -> MLResult:
        """Compare stylometric feature distributions between real and generated text.

        Uses statistical tests (Mann-Whitney U) and effect sizes (Cohen's d)
        instead of training a classifier. Works reliably with small sample sizes.
        """
        real_profiles = [extract_features(t) for t in real_texts]
        gen_profiles = [extract_features(t) for t in generated_texts]

        if not real_profiles or not gen_profiles:
            return MLResult(accuracy=0.5, feature_importances={}, feedback="Insufficient data.")

        # Get stable feature names
        all_names = set()
        for p in real_profiles + gen_profiles:
            all_names.update(p.feature_names)
        feature_names = sorted(n for n in all_names if not n.startswith("ngram_"))

        # Compare distributions per feature
        feature_gaps: list[tuple[str, float, float, float, float]] = []  # name, real_mean, gen_mean, effect_size, p_value
        all_effects: list[float] = []

        for name in feature_names:
            real_vals = np.array([p.features.get(name, 0.0) for p in real_profiles])
            gen_vals = np.array([p.features.get(name, 0.0) for p in gen_profiles])

            real_mean = float(np.mean(real_vals))
            gen_mean = float(np.mean(gen_vals))

            # Cohen's d effect size
            pooled_std = np.sqrt((np.std(real_vals) ** 2 + np.std(gen_vals) ** 2) / 2)
            if pooled_std > 0:
                cohens_d = abs(real_mean - gen_mean) / pooled_std
            else:
                cohens_d = 0.0

            # Mann-Whitney U test
            if len(set(real_vals)) > 1 or len(set(gen_vals)) > 1:
                try:
                    _, p_value = stats.mannwhitneyu(real_vals, gen_vals, alternative="two-sided")
                except Exception:
                    p_value = 1.0
            else:
                p_value = 1.0

            all_effects.append(cohens_d)
            if p_value < 0.05 and cohens_d > 0.5:
                feature_gaps.append((name, real_mean, gen_mean, cohens_d, p_value))

        # Sort by effect size (biggest gaps first)
        feature_gaps.sort(key=lambda x: x[3], reverse=True)

        # Overall score: mean effect size across all features
        mean_effect = float(np.mean(all_effects)) if all_effects else 0.0

        # Map effect size to a 0-1 "accuracy" for ensemble compatibility
        # 0.0 effect -> 0.5 accuracy (indistinguishable)
        # 1.0+ effect -> ~1.0 accuracy (easily distinguishable)
        accuracy = min(1.0, 0.5 + mean_effect * 0.4)

        # Build feedback showing top gaps
        n_significant = len(feature_gaps)
        n_total = len(feature_names)
        n_matched = n_total - n_significant

        feedback_lines = ["## Style Distance Analysis\n"]
        feedback_lines.append(f"Mean effect size: **{mean_effect:.3f}** (target: < 0.3)")
        feedback_lines.append(f"  0-0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large\n")
        feedback_lines.append(f"Matched features: {n_matched}/{n_total} ({n_matched/n_total:.0%})")
        feedback_lines.append(f"Significant gaps: {n_significant}/{n_total}\n")

        feedback_lines.append("### Top Feature Gaps (fix these first):\n")
        top_n = min(10, len(feature_gaps))
        importances = {}
        for name, real_mean, gen_mean, effect, p_val in feature_gaps[:top_n]:
            direction = "too high" if gen_mean > real_mean else "too low"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            desc = _describe_feature(name)
            feedback_lines.append(
                f"- **{name}** ({desc}): generated is {direction} "
                f"(real={real_mean:.4f}, gen={gen_mean:.4f}, effect={effect:.2f}) {sig}"
            )
            importances[name] = effect

        if n_matched > 0:
            feedback_lines.append(f"\n### Well-matched features ({n_matched}):\n")
            matched_names = [
                name for name in feature_names
                if name not in {g[0] for g in feature_gaps}
            ]
            # Show a sample of matched features
            for name in matched_names[:5]:
                real_vals = [p.features.get(name, 0.0) for p in real_profiles]
                gen_vals = [p.features.get(name, 0.0) for p in gen_profiles]
                feedback_lines.append(
                    f"- {name}: real={np.mean(real_vals):.4f}, gen={np.mean(gen_vals):.4f} (matched)"
                )
            if len(matched_names) > 5:
                feedback_lines.append(f"- ... and {len(matched_names) - 5} more")

        return MLResult(
            accuracy=accuracy,
            feature_importances=importances,
            feedback="\n".join(feedback_lines),
        )
