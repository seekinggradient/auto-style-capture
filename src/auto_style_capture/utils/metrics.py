from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IterationRecord:
    iteration: int
    ensemble_accuracy: float
    ml_accuracy: float
    llm_accuracy: float
    dimension_scores: dict[str, float]


@dataclass
class ConvergenceTracker:
    records: list[IterationRecord] = field(default_factory=list)

    def record(
        self,
        iteration: int,
        ensemble_accuracy: float,
        ml_accuracy: float,
        llm_accuracy: float,
        dimension_scores: dict[str, float] | None = None,
    ) -> None:
        self.records.append(
            IterationRecord(
                iteration=iteration,
                ensemble_accuracy=ensemble_accuracy,
                ml_accuracy=ml_accuracy,
                llm_accuracy=llm_accuracy,
                dimension_scores=dimension_scores or {},
            )
        )

    def is_converged(self, threshold: float = 0.55, min_iterations: int = 3) -> bool:
        if len(self.records) < min_iterations:
            return False
        return self.records[-1].ensemble_accuracy < threshold

    def is_plateaued(self, window: int = 3, min_delta: float = 0.02) -> bool:
        if len(self.records) < window + 1:
            return False
        recent = self.records[-window:]
        deltas = [
            abs(recent[i].ensemble_accuracy - recent[i - 1].ensemble_accuracy)
            for i in range(1, len(recent))
        ]
        return all(d < min_delta for d in deltas)

    @property
    def latest_accuracy(self) -> float:
        return self.records[-1].ensemble_accuracy if self.records else 1.0

    @property
    def best_accuracy(self) -> float:
        """Lowest accuracy = best (hardest to distinguish)."""
        return min(r.ensemble_accuracy for r in self.records) if self.records else 1.0

    def summary(self) -> str:
        if not self.records:
            return "No iterations recorded."
        lines = ["Iteration | Ensemble | ML      | LLM Judge"]
        lines.append("-" * 50)
        for r in self.records:
            lines.append(
                f"    {r.iteration:2d}    | {r.ensemble_accuracy:6.1%}  | {r.ml_accuracy:6.1%} | {r.llm_accuracy:6.1%}"
            )
        lines.append(f"\nBest (lowest) accuracy: {self.best_accuracy:.1%}")
        return "\n".join(lines)

    def save_plot(self, path: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            iterations = [r.iteration for r in self.records]
            ensemble = [r.ensemble_accuracy for r in self.records]
            ml = [r.ml_accuracy for r in self.records]
            llm = [r.llm_accuracy for r in self.records]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iterations, ensemble, "b-o", label="Ensemble", linewidth=2)
            ax.plot(iterations, ml, "g--s", label="ML Classifier", alpha=0.7)
            ax.plot(iterations, llm, "r--^", label="LLM Judge", alpha=0.7)
            ax.axhline(y=0.55, color="gray", linestyle=":", label="Convergence threshold")
            ax.axhline(y=0.5, color="lightgray", linestyle=":", label="Random chance")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Discriminator Accuracy")
            ax.set_title("Style Capture Convergence")
            ax.legend()
            ax.set_ylim(0.4, 1.05)
            ax.grid(True, alpha=0.3)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass
