from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True
litellm.drop_params = True


@dataclass
class UsageStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    calls_by_purpose: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def record(self, prompt_tokens: int, completion_tokens: int, purpose: str) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        self.calls_by_purpose[purpose] = self.calls_by_purpose.get(purpose, 0) + 1

    def summary(self) -> str:
        lines = [
            f"Total LLM calls: {self.total_calls}",
            f"Total tokens: {self.total_tokens:,} (prompt: {self.total_prompt_tokens:,}, completion: {self.total_completion_tokens:,})",
            "Calls by purpose:",
        ]
        for purpose, count in sorted(self.calls_by_purpose.items()):
            lines.append(f"  {purpose}: {count}")
        return "\n".join(lines)


class LLMProvider:
    def __init__(self) -> None:
        self.usage = UsageStats()

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        purpose: str = "general",
        reasoning_effort: str | None = None,
    ) -> str:
        extra_kwargs = {}
        if reasoning_effort:
            extra_kwargs["reasoning_effort"] = reasoning_effort
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        content = response.choices[0].message.content
        usage = response.usage
        if usage:
            self.usage.record(usage.prompt_tokens, usage.completion_tokens, purpose)
        logger.debug("LLM call [%s] %s: %d tokens", purpose, model, usage.total_tokens if usage else 0)
        return content

    def complete_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 16000,
        purpose: str = "general",
        response_schema: dict | None = None,
    ) -> dict:
        # Use structured outputs (response_format) when a schema is provided
        extra_kwargs = {}
        if response_schema:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": purpose,
                    "schema": response_schema,
                    "strict": True,
                },
            }
        else:
            extra_kwargs["response_format"] = {"type": "json_object"}
            system = system + "\n\nRespond with valid JSON only. No markdown fences, no explanation."

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        raw = response.choices[0].message.content
        usage = response.usage
        if usage:
            self.usage.record(usage.prompt_tokens, usage.completion_tokens, purpose)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        # Try to extract JSON from mixed content (e.g., model "thinking" before JSON)
        if not raw or raw[0] not in "{[":
            # Find the first { or [ in the response
            brace = raw.find("{")
            bracket = raw.find("[")
            candidates = [i for i in (brace, bracket) if i >= 0]
            if candidates:
                start = min(candidates)
                raw = raw[start:]
            else:
                raise ValueError(f"No JSON found in response: {raw[:200]}")
        return json.loads(raw)
