from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an expert literary critic and forensic stylometrist. You analyze writing \
style — sentence structure, vocabulary, rhythm, tone, punctuation habits, and \
rhetorical patterns. You do NOT try to identify authors by name or recognize \
specific texts. You evaluate purely based on stylistic similarity.

You must respond with valid JSON only."""

JUDGE_USER_PROMPT = """\
## Reference Style

Here are {n_ref} passages written in a particular style. Study the style carefully — \
sentence lengths, word choices, punctuation habits, tone, rhetorical devices, rhythm.

{references}

## Evaluation

Now here are {n_pairs} pairs of passages labeled A and B. For each pair, one passage \
is written in the SAME style as the references above, and one is an imitation. \
Based ONLY on stylistic similarity to the references, identify which is the original.

{pairs}

For each pair, respond with this JSON structure:
{{
  "pairs": [
    {{
      "pair_id": 1,
      "real_is": "A or B",
      "confidence": 0.0-1.0,
      "reasoning": "what stylistic features gave it away — reference specific patterns",
      "dimension_scores": {{
        "voice_and_tone": 1-5,
        "sentence_rhythm": 1-5,
        "vocabulary": 1-5,
        "punctuation": 1-5,
        "rhetorical_devices": 1-5,
        "overall_authenticity": 1-5
      }},
      "specific_feedback": "concrete advice for making the imitation closer to the reference style"
    }}
  ]
}}

Score 5 = the imitation is indistinguishable from the reference style for that dimension.
Score 1 = obvious stylistic difference.
Focus on STYLE, not content. Two passages can be about completely different topics but share the same style."""


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "integer"},
                    "real_is": {"type": "string", "enum": ["A", "B"]},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "dimension_scores": {
                        "type": "object",
                        "properties": {
                            "voice_and_tone": {"type": "integer"},
                            "sentence_rhythm": {"type": "integer"},
                            "vocabulary": {"type": "integer"},
                            "punctuation": {"type": "integer"},
                            "rhetorical_devices": {"type": "integer"},
                            "overall_authenticity": {"type": "integer"},
                        },
                        "required": ["voice_and_tone", "sentence_rhythm", "vocabulary",
                                     "punctuation", "rhetorical_devices", "overall_authenticity"],
                        "additionalProperties": False,
                    },
                    "specific_feedback": {"type": "string"},
                },
                "required": ["pair_id", "real_is", "confidence", "reasoning",
                             "dimension_scores", "specific_feedback"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pairs"],
    "additionalProperties": False,
}


@dataclass
class LLMJudgeResult:
    accuracy: float
    dimension_scores: dict[str, float]
    feedback: str


class LLMJudge:
    def __init__(self, llm: LLMProvider, model: str, temperature: float = 0.2):
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        real_texts: list[str],
        generated_texts: list[str],
        author_name: str,
    ) -> LLMJudgeResult:
        """Evaluate style match using reference-based comparison.

        Shows the judge reference examples of the target style, then asks it
        to identify which passage in each pair matches the reference style.
        The author_name is NOT given to the judge — it evaluates purely on
        stylistic similarity to the provided references.
        """
        n_pairs = min(len(real_texts), len(generated_texts), 5)
        if n_pairs == 0:
            return LLMJudgeResult(accuracy=0.5, dimension_scores={}, feedback="No data.")

        # Strictly partition real texts: references and pair samples must not overlap.
        # This prevents the judge from matching on content rather than style.
        available_real = list(real_texts)
        random.shuffle(available_real)

        n_refs = min(5, max(2, len(available_real) // 2))
        ref_texts = available_real[:n_refs]
        pair_real_pool = available_real[n_refs:]  # no overlap with refs

        real_sample = random.sample(pair_real_pool, min(n_pairs, len(pair_real_pool)))
        while len(real_sample) < n_pairs:
            # Only if corpus is very small — avoid but don't crash
            real_sample.append(random.choice(pair_real_pool))

        gen_sample = random.sample(generated_texts, min(n_pairs, len(generated_texts)))

        # Build reference section
        ref_parts = []
        for i, ref in enumerate(ref_texts):
            ref_parts.append(f"### Reference {i+1}\n\n{ref[:1500]}")
        references_str = "\n\n---\n\n".join(ref_parts)

        # Build randomized pairs
        pairs_text = []
        answer_key = []
        for i, (real, gen) in enumerate(zip(real_sample, gen_sample)):
            if random.random() < 0.5:
                a, b = real, gen
                answer_key.append("A")
            else:
                a, b = gen, real
                answer_key.append("B")

            pairs_text.append(
                f"### Pair {i+1}\n\n**Passage A:**\n{a[:1500]}\n\n**Passage B:**\n{b[:1500]}"
            )

        pairs_str = "\n\n---\n\n".join(pairs_text)

        try:
            result = self.llm.complete_json(
                model=self.model,
                system=JUDGE_SYSTEM_PROMPT,
                user=JUDGE_USER_PROMPT.format(
                    n_ref=n_refs,
                    references=references_str,
                    n_pairs=n_pairs,
                    pairs=pairs_str,
                ),
                temperature=self.temperature,
                max_tokens=16000,
                purpose="llm_judge",
                response_schema=JUDGE_RESPONSE_SCHEMA,
            )
        except Exception:
            logger.warning("LLM judge JSON parse failed, falling back", exc_info=True)
            return LLMJudgeResult(accuracy=0.5, dimension_scores={}, feedback="Judge evaluation failed.")

        # Score accuracy
        pairs_data = result.get("pairs", [])
        correct = 0
        all_dim_scores: dict[str, list[float]] = {}
        feedback_parts: list[str] = ["## LLM Judge Analysis\n"]

        for i, pair in enumerate(pairs_data):
            if i >= len(answer_key):
                break
            predicted_real = pair.get("real_is", "")
            if predicted_real.upper() == answer_key[i]:
                correct += 1

            dims = pair.get("dimension_scores", {})
            for dim, score in dims.items():
                if dim not in all_dim_scores:
                    all_dim_scores[dim] = []
                all_dim_scores[dim].append(float(score))

            reasoning = pair.get("reasoning", "")
            specific = pair.get("specific_feedback", "")
            if reasoning or specific:
                feedback_parts.append(f"**Pair {i+1}:** {reasoning}")
                if specific:
                    feedback_parts.append(f"  - Feedback: {specific}")

        accuracy = correct / len(answer_key) if answer_key else 0.5
        avg_dims = {dim: sum(scores) / len(scores) for dim, scores in all_dim_scores.items()}

        feedback_parts.insert(1, f"Judge accuracy: {accuracy:.1%} ({correct}/{len(answer_key)} correct)\n")

        if avg_dims:
            feedback_parts.append("\n### Dimension Scores (5=indistinguishable):\n")
            for dim, score in sorted(avg_dims.items()):
                bar = "=" * int(score) + "-" * (5 - int(score))
                feedback_parts.append(f"- {dim}: [{bar}] {score:.1f}/5")

        return LLMJudgeResult(
            accuracy=accuracy,
            dimension_scores=avg_dims,
            feedback="\n".join(feedback_parts),
        )
