from __future__ import annotations

import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..corpus.models import Corpus
from ..llm.provider import LLMProvider
from ..style_skill.skill import StyleSkill

logger = logging.getLogger(__name__)

TOPIC_EXTRACTION_PROMPT = """\
Analyze these writing samples and extract {n} distinct topics, themes, or subject \
matters that the author writes about. Return each topic as a one-sentence writing \
prompt that could be used to generate a new passage on that topic.

## Samples

{samples}

Return exactly {n} writing prompts, one per line, numbered 1-{n}. Each prompt should \
describe a specific topic or scenario for the author to write about, matching the \
types of subjects they cover. Keep prompts short (1-2 sentences).
"""


class Generator:
    def __init__(self, llm: LLMProvider, model: str, temperature: float = 0.8):
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def extract_topics(self, corpus: Corpus, n: int = 10) -> list[str]:
        """Extract topic prompts from the corpus for diverse generation."""
        sample_docs = corpus.sample(min(8, len(corpus)))
        samples_text = "\n\n---\n\n".join(doc.text[:1500] for doc in sample_docs)

        result = self.llm.complete(
            model=self.model,
            system="You are a literary analyst.",
            user=TOPIC_EXTRACTION_PROMPT.format(samples=samples_text, n=n),
            temperature=0.3,
            max_tokens=2048,
            purpose="topic_extraction",
        )

        topics = []
        for line in result.strip().split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned:
                topics.append(cleaned)

        return topics[:n] if topics else [
            "Write a short passage on any topic.",
        ]

    def _generate_one(
        self,
        style_skill: StyleSkill,
        topic: str,
        index: int,
        target_length: int = 200,
    ) -> str:
        """Generate a single sample with retries."""
        user_prompt = (
            f"{topic}\n\n"
            f"Write approximately {target_length} words. "
            f"Write ONLY the passage itself, no meta-commentary, no title, no preamble."
        )

        text = ""
        for attempt in range(3):
            text = self.llm.complete(
                model=self.model,
                system=style_skill.to_prompt(),
                user=user_prompt,
                temperature=self.temperature,
                max_tokens=16000,
                purpose="text_generation",
                    reasoning_effort="low",
            )
            text = text.strip() if text else ""
            if len(text.split()) >= 20:
                break
            logger.warning(
                "Sample %d attempt %d returned insufficient content (%d words), retrying",
                index + 1, attempt + 1, len(text.split()),
            )

        if not text or len(text.split()) < 20:
            logger.error("Sample %d: all retries returned empty/short content", index + 1)

        logger.debug("Generated sample %d (%d words)", index + 1, len(text.split()))
        return text

    def generate(
        self,
        style_skill: StyleSkill,
        topics: list[str],
        n: int = 5,
        target_length: int = 200,
        parallel: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Generate n text samples using the style skill.

        Returns (samples, selected_topics) so callers can see which prompts were used.
        """
        selected_topics = random.choices(topics, k=n) if topics else ["Write a passage."] * n

        if parallel and n > 1:
            # Generate all samples in parallel
            samples = [""] * n
            with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
                futures = {
                    executor.submit(
                        self._generate_one, style_skill, topic, i, target_length
                    ): i
                    for i, topic in enumerate(selected_topics)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    samples[idx] = future.result()
        else:
            samples = [
                self._generate_one(style_skill, topic, i, target_length)
                for i, topic in enumerate(selected_topics)
            ]

        return samples, selected_topics
