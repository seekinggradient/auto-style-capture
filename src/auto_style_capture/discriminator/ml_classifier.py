from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..features.extractor import extract_features

logger = logging.getLogger(__name__)


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
            feedback_lines.append(
                f"- **{name}**: generated is {direction} "
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
