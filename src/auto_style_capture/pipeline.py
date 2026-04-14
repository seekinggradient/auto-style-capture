from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .corpus.loader import load_corpus
from .corpus.splitter import split_corpus
from .discriminator.ensemble import DiscriminatorResult, EnsembleDiscriminator
from .features.extractor import extract_corpus_features
from .generator.generator import Generator
from .llm.provider import LLMProvider
from .style_skill.skill import StyleSkill
from .style_skill.templates import generate_initial_skill
from .updater.updater import Updater
from .utils.metrics import ConvergenceTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    style_skill: StyleSkill
    tracker: ConvergenceTracker
    final_result: DiscriminatorResult | None
    output_dir: str


class Pipeline:
    def __init__(self, config: Config, progress_callback=None):
        self.config = config
        self.llm = LLMProvider()
        self.progress = progress_callback or (lambda *args, **kwargs: None)

    def run(self) -> PipelineResult:
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Ingest & Analyze
        self.progress("phase", "Loading corpus...")
        corpus = load_corpus(cfg.corpus.path, cfg.corpus.json_text_field)
        train_corpus, holdout_corpus = split_corpus(corpus, cfg.corpus.holdout_ratio)
        logger.info(
            "Corpus: %d documents (%d train, %d holdout)",
            len(corpus),
            len(train_corpus),
            len(holdout_corpus),
        )

        train_chunks = train_corpus.chunks(target_length=200)
        if not train_chunks:
            train_chunks = [doc.text for doc in train_corpus.documents]
        logger.info("Train chunks: %d", len(train_chunks))

        # Extract style profile for reference
        self.progress("phase", "Extracting style profile...")
        real_profile = extract_corpus_features(train_chunks[:20])

        # Phase 2: Seed the style skill
        self.progress("phase", "Generating initial style skill...")
        style_skill = generate_initial_skill(
            train_corpus,
            cfg.author_name,
            self.llm,
            cfg.llm.updater_model,
            cfg.llm.temperature_updater,
        )
        if cfg.save_intermediates:
            style_skill.save(output_dir)
        logger.info("Initial style skill generated (v0)")

        # Phase 3: Build components
        generator = Generator(self.llm, cfg.llm.generator_model, cfg.llm.temperature_generator)
        discriminator = EnsembleDiscriminator(
            self.llm,
            cfg.llm.judge_model,
            cfg.llm.temperature_judge,
            cfg.discriminator.ml_weight,
            cfg.discriminator.llm_weight,
        )
        updater = Updater(self.llm, cfg.llm.updater_model, cfg.llm.temperature_updater)
        tracker = ConvergenceTracker()

        # Extract topics for diverse generation
        self.progress("phase", "Extracting topics...")
        topics = generator.extract_topics(train_corpus, n=10)

        # Phase 4: Adversarial loop
        for iteration in range(cfg.loop.max_iterations):
            self.progress("iteration", iteration=iteration, max_iterations=cfg.loop.max_iterations)
            logger.info("=== Iteration %d ===", iteration)

            # Generate samples
            self.progress("step", f"Iteration {iteration}: Generating samples...")
            samples, _ = generator.generate(
                style_skill, topics, n=cfg.loop.samples_per_iteration
            )

            # Discriminate
            self.progress("step", f"Iteration {iteration}: Evaluating...")
            result = discriminator.evaluate(train_chunks, samples, cfg.author_name)

            # Record
            tracker.record(
                iteration=iteration,
                ensemble_accuracy=result.accuracy,
                ml_accuracy=result.ml_result.accuracy,
                llm_accuracy=result.llm_result.accuracy,
                dimension_scores=result.llm_result.dimension_scores,
            )

            self.progress(
                "accuracy",
                iteration=iteration,
                accuracy=result.accuracy,
                ml_accuracy=result.ml_result.accuracy,
                llm_accuracy=result.llm_result.accuracy,
            )
            logger.info(
                "Accuracy: %.1f%% (ML: %.1f%%, LLM: %.1f%%)",
                result.accuracy * 100,
                result.ml_result.accuracy * 100,
                result.llm_result.accuracy * 100,
            )

            # Check convergence
            if tracker.is_converged(cfg.loop.convergence_threshold, cfg.loop.min_iterations):
                logger.info("Converged! Discriminator accuracy below threshold.")
                self.progress("converged", iteration=iteration)
                break

            # Check plateau
            if tracker.is_plateaued(cfg.loop.plateau_window, cfg.loop.plateau_min_delta):
                logger.warning("Plateau detected, updater will apply reset strategy.")

            # Update style skill
            self.progress("step", f"Iteration {iteration}: Updating style skill...")
            real_samples = [doc.text for doc in train_corpus.sample(3)]
            style_skill = updater.update(style_skill, result, real_samples, samples[:3])

            if cfg.save_intermediates:
                style_skill.save(output_dir)

        # Phase 5: Final evaluation on holdout
        self.progress("phase", "Running final evaluation on holdout set...")
        final_result = None
        holdout_chunks = holdout_corpus.chunks(target_length=200)
        if holdout_chunks and len(holdout_chunks) >= 2:
            holdout_topics = generator.extract_topics(holdout_corpus, n=5)
            final_samples, _ = generator.generate(
                style_skill, holdout_topics, n=cfg.loop.samples_per_iteration
            )
            final_result = discriminator.evaluate(holdout_chunks, final_samples, cfg.author_name)
            logger.info("Final holdout accuracy: %.1f%%", final_result.accuracy * 100)

        # Save final artifacts
        style_skill.save(output_dir)
        tracker.save_plot(str(output_dir / "convergence.png"))

        # Save convergence summary
        summary_path = output_dir / "convergence_summary.txt"
        summary_path.write_text(tracker.summary(), encoding="utf-8")

        # Save LLM usage
        usage_path = output_dir / "llm_usage.txt"
        usage_path.write_text(self.llm.usage.summary(), encoding="utf-8")


        self.progress("done")
        return PipelineResult(
            style_skill=style_skill,
            tracker=tracker,
            final_result=final_result,
            output_dir=str(output_dir),
        )
