from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StyleProfile:
    features: dict[str, float] = field(default_factory=dict)

    def to_vector(self, feature_names: list[str] | None = None) -> np.ndarray:
        if feature_names is None:
            feature_names = sorted(self.features.keys())
        return np.array([self.features.get(name, 0.0) for name in feature_names])

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())

    def to_summary(self) -> str:
        lines = ["Stylometric Profile:"]
        categories: dict[str, list[tuple[str, float]]] = {}
        for name, value in sorted(self.features.items()):
            category = name.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append((name, value))

        for category, feats in sorted(categories.items()):
            lines.append(f"\n  {category.upper()}:")
            for name, value in feats:
                lines.append(f"    {name}: {value:.4f}")

        return "\n".join(lines)

    def compare(self, other: StyleProfile) -> str:
        """Compare this profile to another, highlighting significant differences."""
        lines = ["Feature Comparison (real vs generated):"]
        all_names = sorted(set(self.feature_names) | set(other.feature_names))
        diffs = []
        for name in all_names:
            real_val = self.features.get(name, 0.0)
            gen_val = other.features.get(name, 0.0)
            if real_val == 0 and gen_val == 0:
                continue
            denominator = max(abs(real_val), abs(gen_val), 0.001)
            pct_diff = abs(real_val - gen_val) / denominator
            diffs.append((name, real_val, gen_val, pct_diff))

        diffs.sort(key=lambda x: x[3], reverse=True)
        for name, real_val, gen_val, pct_diff in diffs[:20]:
            direction = "higher" if gen_val > real_val else "lower"
            lines.append(f"  {name}: real={real_val:.4f} gen={gen_val:.4f} ({direction}, {pct_diff:.0%} diff)")

        return "\n".join(lines)
