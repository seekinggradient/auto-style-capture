from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..discriminator.ensemble import DiscriminatorResult
from ..llm.provider import LLMProvider
from ..style_skill.skill import StyleSkill

logger = logging.getLogger(__name__)

UPDATE_PROMPT = """\
You are a style instruction engineer. Your job is to refine a "style skill" document \
-- a self-contained writing guide that tells an LLM how to write in a specific author's style.

Below you'll find the current style skill, a analysis of differences between the \
author's real writing and text generated using the current skill, and examples of both.

## Current Style Skill (v{version})

{current_skill}

## Style Analysis

{feedback}

## Example Real Text (by {author_name})

{real_samples}

## Example Generated Text (using current skill)

{generated_samples}

{momentum_section}

## Instructions

Rewrite the COMPLETE style skill to close the identified gaps. Key principles:
1. Be SPECIFIC and QUANTITATIVE -- instead of "use varied sentence lengths," say \
"alternate between short sentences (5-8 words) and long complex sentences (25-35 words), \
60/40 ratio."
2. Address EVERY issue identified in the style analysis.
3. Preserve what's already working well -- don't break things that aren't flagged.
4. Add CONCRETE anti-patterns based on what the generated text does wrong.
5. Include EXAMPLES of characteristic phrases, transitions, and sentence structures \
drawn from the real text samples.

CRITICAL: The output must be a SELF-CONTAINED style guide -- a writing manual that \
someone could follow without any knowledge of how it was created. Do NOT reference \
"the discriminator", "the classifier", "the analysis", "feature gaps", "iterations", \
or any part of this training process. Write it as a pure style guide: "Write like this. \
Avoid that. Use this structure."

Start the document with: # Style Skill: {author_name}
Return the complete updated Markdown document."""

RESET_PROMPT_ADDITION = """
## IMPORTANT: Strategy Reset Required

The following issues have persisted for {n} iterations without improvement:
{persistent_issues}

Incremental tweaks have not resolved these. Take a FUNDAMENTALLY DIFFERENT APPROACH \
to addressing them. Consider:
- Restructuring the relevant section entirely
- Adding worked examples showing the exact pattern to follow
- Adding explicit "DO this, NOT that" pairs with quoted examples from the real text
"""


@dataclass
class FeedbackTracker:
    """Tracks recurring feedback themes across iterations."""
    history: list[str] = field(default_factory=list)

    def add(self, feedback: str) -> None:
        self.history.append(feedback)

    def get_persistent_issues(self, window: int = 3) -> list[str]:
        """Find feedback themes that recur across recent iterations."""
        if len(self.history) < window:
            return []

        recent = self.history[-window:]
        # Simple keyword overlap detection
        from collections import Counter
        import re

        keyword_counts: Counter[str] = Counter()
        for fb in recent:
            # Extract feature names and key phrases
            keywords = set()
            for match in re.finditer(r"\*\*(\w+)\*\*", fb):
                keywords.add(match.group(1))
            for match in re.finditer(r"(sentence.length|word.length|punctuation|vocabulary|tone|voice|rhythm)", fb, re.I):
                keywords.add(match.group(0).lower())
            for kw in keywords:
                keyword_counts[kw] += 1

        return [kw for kw, count in keyword_counts.items() if count >= window]


class Updater:
    def __init__(self, llm: LLMProvider, model: str, temperature: float = 0.4):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.tracker = FeedbackTracker()

    def update(
        self,
        style_skill: StyleSkill,
        result: DiscriminatorResult,
        real_samples: list[str],
        generated_samples: list[str],
    ) -> StyleSkill:
        self.tracker.add(result.feedback)

        # Check for persistent issues
        persistent = self.tracker.get_persistent_issues()
        momentum_section = ""
        if persistent:
            momentum_section = RESET_PROMPT_ADDITION.format(
                n=len(self.tracker.history),
                persistent_issues="\n".join(f"- {issue}" for issue in persistent),
            )

        real_text = "\n\n---\n\n".join(s[:1500] for s in real_samples[:3])
        gen_text = "\n\n---\n\n".join(s[:1500] for s in generated_samples[:3])

        prompt = UPDATE_PROMPT.format(
            version=style_skill.version,
            current_skill=style_skill.content,
            feedback=result.feedback,
            author_name=style_skill.author_name,
            real_samples=real_text,
            generated_samples=gen_text,
            momentum_section=momentum_section,
        )

        new_content = self.llm.complete(
            model=self.model,
            system="You are an expert in computational stylistics and prompt engineering.",
            user=prompt,
            temperature=self.temperature,
            max_tokens=16000,
            purpose="skill_update",
        )

        new_skill = style_skill.update(new_content)
        logger.info(
            "Updated style skill v%d -> v%d",
            style_skill.version,
            new_skill.version,
        )
        return new_skill
