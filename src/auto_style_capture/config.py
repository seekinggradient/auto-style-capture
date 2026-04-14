from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    generator_model: str = "openai/gpt-5-mini"
    judge_model: str = "openai/gpt-5-mini"
    updater_model: str = "openai/gpt-5-mini"
    temperature_generator: float = 0.8
    temperature_judge: float = 0.2
    temperature_updater: float = 0.4


class CorpusConfig(BaseModel):
    path: str = "./corpus"
    formats: list[str] = Field(default_factory=lambda: [".txt", ".md", ".json"])
    holdout_ratio: float = 0.2
    json_text_field: str = "text"


class LoopConfig(BaseModel):
    max_iterations: int = 10
    convergence_threshold: float = 0.55
    min_iterations: int = 3
    samples_per_iteration: int = 5
    plateau_window: int = 3
    plateau_min_delta: float = 0.02


class DiscriminatorConfig(BaseModel):
    ml_weight: float = 0.5
    llm_weight: float = 0.5


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    discriminator: DiscriminatorConfig = Field(default_factory=DiscriminatorConfig)
    output_dir: str = "./output"
    save_intermediates: bool = True
    author_name: str = "Unknown Author"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_defaults(cls, **overrides) -> Config:
        return cls(**overrides)
