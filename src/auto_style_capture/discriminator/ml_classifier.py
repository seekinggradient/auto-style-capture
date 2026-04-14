from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from ..features.extractor import extract_features

logger = logging.getLogger(__name__)


@dataclass
class MLResult:
    accuracy: float
    feature_importances: dict[str, float]
    feedback: str


class MLDiscriminator:
    def evaluate(self, real_texts: list[str], generated_texts: list[str]) -> MLResult:
        """Train a classifier to distinguish real from generated, return accuracy and feedback."""
        # Extract features
        real_profiles = [extract_features(t) for t in real_texts]
        gen_profiles = [extract_features(t) for t in generated_texts]

        if not real_profiles or not gen_profiles:
            return MLResult(accuracy=0.5, feature_importances={}, feedback="Insufficient data.")

        # Balance classes by subsampling the larger set
        # Imbalanced classes (e.g., 382 real vs 5 generated) inflate accuracy artificially
        n_gen = len(gen_profiles)
        if len(real_profiles) > n_gen * 3:
            # Subsample real to ~3x generated for a meaningful comparison
            sample_size = min(n_gen * 3, len(real_profiles))
            real_profiles_balanced = random.sample(real_profiles, sample_size)
            logger.info(
                "Balanced ML classifier: %d real (subsampled from %d) vs %d generated",
                sample_size, len(real_profiles), n_gen,
            )
        else:
            real_profiles_balanced = real_profiles

        # Use ALL profiles for feature importance analysis, but balanced set for accuracy
        all_profiles = real_profiles + gen_profiles

        # Get common feature names (stable features only, no n-grams)
        all_names = set()
        for p in all_profiles:
            all_names.update(p.feature_names)
        # Filter to stable features (exclude n-gram features)
        feature_names = sorted(n for n in all_names if not n.startswith("ngram_"))

        # Build balanced feature matrix for accuracy measurement
        X_real = np.array([p.to_vector(feature_names) for p in real_profiles_balanced])
        X_gen = np.array([p.to_vector(feature_names) for p in gen_profiles])
        X = np.vstack([X_real, X_gen])
        y = np.array([0] * len(X_real) + [1] * len(X_gen))  # 0=real, 1=generated

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Train and evaluate with cross-validation
        clf = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )

        n_samples = len(y)
        n_folds = min(5, n_samples // 2) if n_samples >= 4 else 2
        if n_samples < 4:
            # Too few samples for meaningful CV
            clf.fit(X, y)
            accuracy = clf.score(X, y)
        else:
            scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
            accuracy = float(scores.mean())
            clf.fit(X, y)  # Fit on all data for feature importances

        # Get feature importances
        importances = dict(zip(feature_names, clf.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]

        # Generate human-readable feedback
        feedback_lines = ["## ML Classifier Analysis\n"]
        feedback_lines.append(f"Classifier accuracy: {accuracy:.1%}")
        feedback_lines.append(
            f"({'easily distinguishable' if accuracy > 0.75 else 'somewhat distinguishable' if accuracy > 0.6 else 'hard to distinguish'})\n"
        )
        feedback_lines.append("### Most Distinguishing Features:\n")

        for fname, importance in top_features[:10]:
            if importance < 0.01:
                continue
            # Use ALL real profiles for mean comparison, not just the balanced subset
            real_vals = [p.features.get(fname, 0.0) for p in real_profiles_balanced]
            gen_vals = [p.features.get(fname, 0.0) for p in gen_profiles]
            real_mean = np.mean(real_vals)
            gen_mean = np.mean(gen_vals)
            direction = "higher" if gen_mean > real_mean else "lower"
            feedback_lines.append(
                f"- **{fname}**: generated is {direction} "
                f"(real={real_mean:.4f}, gen={gen_mean:.4f}, importance={importance:.3f})"
            )

        return MLResult(
            accuracy=accuracy,
            feature_importances=dict(top_features),
            feedback="\n".join(feedback_lines),
        )
