from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..llm.provider import LLMProvider
from .ml_classifier import MLDiscriminator, MLResult
from .llm_judge import LLMJudge, LLMJudgeResult


@dataclass
class DiscriminatorResult:
    accuracy: float
    ml_result: MLResult
    llm_result: LLMJudgeResult
    feedback: str

    @property
    def is_converged(self) -> bool:
        return self.accuracy < 0.55


class EnsembleDiscriminator:
    def __init__(
        self,
        llm: LLMProvider,
        judge_model: str,
        judge_temperature: float = 0.2,
        ml_weight: float = 0.5,
        llm_weight: float = 0.5,
    ):
        self.ml_disc = MLDiscriminator()
        self.llm_judge = LLMJudge(llm, judge_model, judge_temperature)
        self.ml_weight = ml_weight
        self.llm_weight = llm_weight

    def evaluate(
        self,
        real_texts: list[str],
        generated_texts: list[str],
        author_name: str,
    ) -> DiscriminatorResult:
        # Run ML classifier and LLM judge in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ml_future = executor.submit(self.ml_disc.evaluate, real_texts, generated_texts)
            llm_future = executor.submit(
                self.llm_judge.evaluate, real_texts, generated_texts, author_name
            )
            ml_result = ml_future.result()
            llm_result = llm_future.result()

        accuracy = (
            self.ml_weight * ml_result.accuracy + self.llm_weight * llm_result.accuracy
        )

        feedback = (
            f"# Discriminator Report (Ensemble Accuracy: {accuracy:.1%})\n\n"
            f"{ml_result.feedback}\n\n---\n\n{llm_result.feedback}"
        )

        return DiscriminatorResult(
            accuracy=accuracy,
            ml_result=ml_result,
            llm_result=llm_result,
            feedback=feedback,
        )
