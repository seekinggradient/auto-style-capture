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
    if name.startswith("lex_fw_"):
        word = name.replace("lex_fw_", "")
        return f'frequency of the word "{word}"'
    return name


def _writing_advice(name: str, real_mean: float, gen_mean: float) -> str:
    """Translate a feature gap into concrete, qualitative writing advice."""
    too_high = gen_mean > real_mean

    # Specific advice for known features
    advice_map_high: dict[str, str] = {
        "lex_fw_and": "You're chaining too many clauses with 'and'. Break compound sentences into shorter ones using periods, or pivot with 'but', 'so', 'yet' instead.",
        "lex_cttr": "Your vocabulary is too varied. Real writers repeat key words freely — say 'the system' again instead of switching to 'the platform'. Don't reach for synonyms.",
        "lex_hapax_ratio": "Too many unique words. Reuse the same nouns and verbs across paragraphs instead of hunting for alternatives.",
        "lex_ttr": "Your vocabulary is too diverse. Repeat common words naturally rather than varying them.",
        "lex_mean_word_length": "Your words are too long on average. Use simpler, shorter words — 'use' not 'utilize', 'help' not 'facilitate', 'get' not 'obtain'.",
        "lex_long_word_ratio": "Too many long words. Mix in more short, common words like 'it', 'the', 'is', 'but', 'so'.",
        "lex_word_length_std": "Your word length is too varied. Settle into a more consistent mix of short and medium words.",
        "punct_emdash_rate": "Too many em dashes. Use commas or parentheses for asides instead, or start a new sentence.",
        "punct_semicolon_rate": "Too many semicolons. Use periods to start new sentences instead.",
        "punct_colon_rate": "Too many colons. Weave explanations into flowing prose rather than using colon introductions.",
        "punct_comma_rate": "Too many commas. Shorten sentences so they need fewer commas, or break into separate sentences.",
        "punct_density": "Too much punctuation overall. Write simpler sentences that need less punctuation.",
        "syn_sent_length_mean": "Your sentences are too long on average. Mix in more short and medium sentences.",
        "syn_long_sent_ratio": "Too many long sentences. Most sentences should be medium length (15-25 words) with only some long ones.",
        "syn_para_length_mean": "Your paragraphs are too long. Break them into shorter ones — each paragraph should carry one idea.",
        "syn_sents_per_para_mean": "Too many sentences per paragraph. Keep paragraphs to 2-4 sentences.",
        "syn_total_sentences": "You're writing too many sentences for this length. Make sentences slightly longer so you use fewer of them.",
        "syn_question_ratio": "Too many questions. Use them sparingly for emphasis — one or two per piece, not every paragraph.",
        "syn_exclamation_ratio": "Too many exclamation marks. Reserve them for genuine surprise or emphasis — very rarely.",
        "read_flesch_kincaid": "The reading level is too high. Use shorter words and simpler sentence structures.",
        "read_gunning_fog": "The text is too complex. Simplify vocabulary and shorten some sentences.",
        "read_coleman_liau": "The readability score is too high. Use shorter, more common words.",
        "read_complex_word_ratio": "Too many complex words (3+ syllables). Use simpler alternatives where possible.",
        "read_syllables_per_word": "Your words have too many syllables on average. Prefer one and two-syllable words.",
    }

    advice_map_low: dict[str, str] = {
        "lex_fw_and": "You're not using 'and' enough. Some natural clause-chaining with 'and' is fine.",
        "lex_cttr": "Your vocabulary is too repetitive. Add a bit more variety in word choice.",
        "lex_short_word_ratio": "Not enough short, common words. Use more tiny words like 'it', 'is', 'the', 'a', 'to', 'in', 'but', 'so'.",
        "lex_mean_word_length": "Your words are too short. Include some naturally longer words like 'important', 'experience', 'understand'.",
        "lex_long_word_ratio": "Not enough longer words. Don't avoid all multi-syllable words — some complexity is natural.",
        "punct_contraction_rate": "Not enough contractions. Use 'don't', 'it's', 'that's', 'can't', 'won't' naturally throughout.",
        "punct_emdash_rate": "Consider using an occasional em dash for dramatic asides or breaks in thought.",
        "punct_comma_rate": "Not enough commas. Longer sentences naturally need commas to separate clauses.",
        "syn_sent_length_mean": "Your sentences are too short on average. Include more medium and long sentences that develop ideas with subordinate clauses.",
        "syn_long_sent_ratio": "Not enough long sentences. About 30-40% of sentences should be 25+ words, with layered clauses that carry analytical weight.",
        "syn_short_sent_ratio": "Not enough short punchy sentences for emphasis. Sprinkle in a few 3-8 word sentences.",
        "syn_question_ratio": "Include more rhetorical questions to engage the reader. 'Why does this happen?' 'What's the alternative?'",
        "syn_exclamation_ratio": "Include an occasional exclamation for genuine emphasis or surprise.",
        "read_flesch_kincaid": "The reading level is too low. Include some complex ideas and longer sentences.",
        "read_gunning_fog": "The text reads too simply. Add some analytical depth with longer, more complex sentences.",
        "read_complex_word_ratio": "Not enough complex words. Natural writing includes some multi-syllable words like 'experience', 'organization', 'particularly'.",
        "syn_sent_length_skew": "Your sentence lengths are too uniform. Most sentences should be short-to-medium with a few long ones pulling the distribution right.",
        "syn_para_length_mean": "Your paragraphs are too short. Develop ideas more fully — aim for 3-5 sentences per paragraph.",
        "lex_yules_k": "Your vocabulary is too varied. Repeat common words more freely instead of hunting for synonyms.",
    }

    if too_high and name in advice_map_high:
        return advice_map_high[name]
    if not too_high and name in advice_map_low:
        return advice_map_low[name]

    # Function words get generic advice
    if name.startswith("lex_fw_"):
        word = name.replace("lex_fw_", "")
        if too_high:
            return f"You're overusing '{word}'. Try to use it less frequently."
        else:
            return f"Use '{word}' more naturally in your writing."

    # Fallback
    desc = _describe_feature(name)
    return f"{'Reduce' if too_high else 'Increase'} {desc}."


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

        feedback_lines.append("### Writing Advice (ranked by importance):\n")
        top_n = min(8, len(feature_gaps))
        importances = {}
        for name, real_mean, gen_mean, effect, p_val in feature_gaps[:top_n]:
            severity = "Major" if effect > 1.5 else "Moderate" if effect > 0.8 else "Minor"
            advice = _writing_advice(name, real_mean, gen_mean)
            feedback_lines.append(f"- **[{severity}]** {advice}")
            importances[name] = effect

        if n_matched > 0:
            feedback_lines.append(f"\n### What's working well ({n_matched} features matched):\n")
            matched_names = [
                name for name in feature_names
                if name not in {g[0] for g in feature_gaps}
            ]
            # Group by category for readability
            categories = {"lex_": "vocabulary", "syn_": "sentence structure", "punct_": "punctuation", "read_": "readability"}
            cat_counts: dict[str, int] = {}
            for mn in matched_names:
                for prefix, cat in categories.items():
                    if mn.startswith(prefix):
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        break
            for cat, count in sorted(cat_counts.items()):
                feedback_lines.append(f"- {cat}: {count} features matched")

        return MLResult(
            accuracy=accuracy,
            feature_importances=importances,
            feedback="\n".join(feedback_lines),
        )
