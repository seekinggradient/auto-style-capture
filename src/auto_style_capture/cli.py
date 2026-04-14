from __future__ import annotations

import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Config

console = Console()

SKILLS_DIR = Path("skills")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def author_slug(author: str) -> str:
    """Convert author name to a directory-safe slug."""
    return author.lower().replace(" ", "_").replace(".", "")


def get_workspace(author: str) -> Path:
    """Get or create workspace directory for an author."""
    workspace = SKILLS_DIR / author_slug(author)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def get_latest_skill(workspace: Path) -> Path | None:
    """Find the highest-versioned skill file in a workspace."""
    skills = sorted(workspace.glob("skill_v*.md"))
    return skills[-1] if skills else None


def next_skill_version(workspace: Path) -> int:
    """Get the next version number for a skill file."""
    latest = get_latest_skill(workspace)
    if latest is None:
        return 0
    try:
        return int(latest.stem.split("_v")[-1]) + 1
    except ValueError:
        return 0


def make_progress_callback():
    """Create a rich-based progress callback."""
    def callback(event: str, *args, **kwargs):
        if event == "phase":
            msg = args[0] if args else kwargs.get("msg", list(kwargs.values())[0] if kwargs else "")
            console.print(f"\n[bold blue]>>> {msg}[/]")
        elif event == "step":
            msg = args[0] if args else kwargs.get("msg", list(kwargs.values())[0] if kwargs else "")
            console.print(f"  [dim]{msg}[/]")
        elif event == "iteration":
            it = kwargs.get("iteration", 0)
            mx = kwargs.get("max_iterations", 0)
            console.print(f"\n[bold yellow]--- Iteration {it + 1}/{mx} ---[/]")
        elif event == "accuracy":
            acc = kwargs.get("accuracy", 0)
            ml = kwargs.get("ml_accuracy", 0)
            llm = kwargs.get("llm_accuracy", 0)
            color = "green" if acc < 0.6 else "yellow" if acc < 0.75 else "red"
            console.print(f"  Ensemble: [{color}]{acc:.1%}[/] | ML: {ml:.1%} | LLM Judge: {llm:.1%}")
        elif event == "converged":
            console.print(Panel("[bold green]Converged! Discriminator can no longer distinguish real from generated.[/]"))
        elif event == "done":
            console.print("\n[bold green]Pipeline complete.[/]")

    return callback


@click.group()
def main():
    """Auto Style Capture - GAN-inspired adversarial style emulation."""
    pass


@main.command()
@click.option("--corpus", type=click.Path(exists=True), required=False, multiple=True, help="Path to corpus directory (can specify multiple)")
@click.option("--author", required=True, help="Author name")
@click.option("--from-skill", "from_skill", type=click.Path(exists=True), default=None, help="Start from an existing skill file instead of generating one")
@click.option("--model", default="openai/gpt-5-mini", help="LLM model")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def seed(corpus, author, from_skill, model, verbose):
    """Generate an initial style skill (v0) for an author.

    Creates the author workspace under skills/{author}/ and generates
    skill_v0.md from the corpus. Use --from-skill to continue from an
    existing skill file instead of generating a new one.
    """
    setup_logging(verbose)

    workspace = get_workspace(author)

    if from_skill:
        # Copy existing skill as v0 in the new workspace
        import shutil
        skill_path = workspace / "skill_v0.md"
        shutil.copy2(from_skill, skill_path)
        console.print(f"Workspace: {workspace}/")
        console.print(f"[bold green]Copied {from_skill} -> {skill_path}[/]")
    else:
        if not corpus:
            console.print("[red]Either --corpus or --from-skill is required.[/]")
            raise SystemExit(1)

        from .corpus.loader import load_corpus, load_corpora
        from .llm.provider import LLMProvider
        from .style_skill.templates import generate_initial_skill

        llm = LLMProvider()
        if len(corpus) == 1:
            corp = load_corpus(corpus[0])
        else:
            corp = load_corpora(list(corpus))
        groups = set(doc.group for doc in corp.documents)
        console.print(f"Loaded {len(corp)} documents from {len(groups)} source(s): {', '.join(sorted(groups))}")
        console.print(f"Workspace: {workspace}/")
        console.print("Generating initial style skill...\n")

        skill = generate_initial_skill(corp, author, llm, model)
        skill_path = workspace / "skill_v0.md"
        skill_path.write_text(skill.content, encoding="utf-8")

        console.print(f"[bold green]Skill written to:[/] {skill_path}")

    console.print(f"\nNext step: edit the skill or run evaluation:")
    console.print(f"  auto-style-capture evaluate --author \"{author}\" --corpus {corpus}")


@main.command()
@click.option("--corpus", type=click.Path(exists=True), required=True, multiple=True, help="Path to corpus directory (can specify multiple)")
@click.option("--author", required=True, help="Author name")
@click.option("--skill", type=click.Path(exists=True), default=None, help="Skill file to evaluate (default: latest version)")
@click.option("--model", default="openai/gpt-5-mini", help="LLM model")
@click.option("--samples", default=5, type=int, help="Number of samples to generate per run")
@click.option("--runs", default=1, type=int, help="Number of evaluation runs to average (reduces variance)")
@click.option("--hypothesis", default=None, help="What you changed and why you expect it to help")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def evaluate(corpus, author, skill, model, samples, runs, hypothesis, verbose):
    """Evaluate a style skill against a corpus.

    Generates samples using the skill, runs the ensemble discriminator
    (ML classifier + LLM judge), and writes feedback.md and results.tsv
    to the author workspace. Use --runs to average multiple evaluations
    for more stable scores.
    """
    setup_logging(verbose)

    from .corpus.loader import load_corpus, load_corpora
    from .discriminator.ensemble import EnsembleDiscriminator
    from .features.extractor import extract_corpus_features
    from .generator.generator import Generator
    from .llm.provider import LLMProvider
    from .style_skill.skill import StyleSkill

    workspace = get_workspace(author)

    if skill:
        skill_path = Path(skill)
    else:
        skill_path = get_latest_skill(workspace)
        if skill_path is None:
            console.print("[red]No skill found. Run 'seed' first.[/]")
            raise SystemExit(1)

    llm = LLMProvider()
    style_skill = StyleSkill.load(skill_path)
    if len(corpus) == 1:
        corp = load_corpus(corpus[0])
    else:
        corp = load_corpora(list(corpus))
    chunks = corp.chunks(target_length=200)

    groups = set(doc.group for doc in corp.documents)
    console.print(f"Author: {author}")
    console.print(f"Skill: {skill_path}")
    console.print(f"Sources: {', '.join(sorted(groups))}")
    console.print(f"Corpus: {len(corp)} documents, {len(chunks)} chunks")
    if runs > 1:
        console.print(f"Runs: {runs} (scores will be averaged)")
    console.print()

    gen = Generator(llm, model)
    disc = EnsembleDiscriminator(llm, model)

    # Load cached topics or extract new ones
    import json
    topics_cache = workspace / "topics.json"
    if topics_cache.exists():
        topics = json.loads(topics_cache.read_text())
        console.print(f"Loaded {len(topics)} cached topics")
    else:
        console.print("Extracting topics (will be cached for future runs)...")
        topics = gen.extract_topics(corp, n=10)
        topics_cache.write_text(json.dumps(topics, indent=2))
        console.print(f"Cached {len(topics)} topics to {topics_cache}")

    all_results = []
    all_generated = []
    all_used_topics = []

    for run_idx in range(runs):
        if runs > 1:
            console.print(f"\n[bold yellow]--- Run {run_idx + 1}/{runs} ---[/]")

        console.print(f"Generating {samples} samples (parallel)...")
        generated, used_topics = gen.generate(style_skill, topics, n=samples)

        # Check for empty samples
        empty_count = sum(1 for s in generated if len(s.split()) < 20)
        if empty_count > 0:
            console.print(f"[red]Warning: {empty_count}/{samples} samples are empty/too short[/]")

        console.print("Running discriminator (ML + LLM judge in parallel)...")
        result = disc.evaluate(chunks, generated, author)

        all_results.append(result)
        all_generated.extend(generated)
        all_used_topics.extend(used_topics)

        color = "green" if result.accuracy < 0.6 else "yellow" if result.accuracy < 0.75 else "red"
        console.print(
            f"  Ensemble: [{color}]{result.accuracy:.1%}[/] | "
            f"ML: {result.ml_result.accuracy:.1%} | "
            f"LLM Judge: {result.llm_result.accuracy:.1%}"
        )

    # Average scores across runs
    avg_ensemble = sum(r.accuracy for r in all_results) / len(all_results)
    avg_ml = sum(r.ml_result.accuracy for r in all_results) / len(all_results)
    avg_llm = sum(r.llm_result.accuracy for r in all_results) / len(all_results)

    # Use the last run's detailed feedback (most recent)
    last_result = all_results[-1]

    # Feature comparison using all generated samples
    real_profile = extract_corpus_features(chunks[:20])
    gen_profile = extract_corpus_features(all_generated)
    feature_comparison = real_profile.compare(gen_profile)

    if runs > 1:
        console.print(f"\n[bold]Averaged over {runs} runs:[/]")
        color = "green" if avg_ensemble < 0.6 else "yellow" if avg_ensemble < 0.75 else "red"
        console.print(
            f"  Ensemble: [{color}]{avg_ensemble:.1%}[/] | "
            f"ML: {avg_ml:.1%} | LLM Judge: {avg_llm:.1%}"
        )

    console.print(Panel(feature_comparison, title="Feature Comparison"))

    # Write feedback.md
    feedback_path = workspace / "feedback.md"
    feedback_content = (
        f"# Evaluation Feedback\n\n"
        f"- **Author**: {author}\n"
        f"- **Skill**: {skill_path.name}\n"
        f"- **Ensemble accuracy**: {avg_ensemble:.1%}"
        f"{f' (averaged over {runs} runs)' if runs > 1 else ''}\n"
        f"- **ML classifier accuracy**: {avg_ml:.1%}\n"
        f"- **LLM judge accuracy**: {avg_llm:.1%}\n"
        f"- **Timestamp**: {datetime.now().isoformat()}\n"
    )
    if hypothesis:
        feedback_content += f"- **Hypothesis**: {hypothesis}\n"
    feedback_content += (
        f"\n## Topic Prompts\n\n"
        f"Available topics extracted from corpus:\n\n"
    )
    for i, t in enumerate(topics):
        feedback_content += f"{i+1}. {t}\n"
    feedback_content += (
        f"\n## Discriminator Feedback\n\n{last_result.feedback}\n\n"
        f"## Feature Comparison\n\n{feature_comparison}\n\n"
        f"## Generated Samples\n\n"
    )
    for i, s in enumerate(all_generated):
        prompt = all_used_topics[i] if i < len(all_used_topics) else "unknown"
        feedback_content += f"### Sample {i+1}\n\n**Prompt:** {prompt}\n\n{s}\n\n---\n\n"
    feedback_path.write_text(feedback_content, encoding="utf-8")

    # Append to results.tsv
    results_path = workspace / "results.tsv"
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["timestamp", "skill_version", "ensemble_acc", "ml_acc", "llm_acc", "runs", "hypothesis"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            skill_path.name,
            f"{avg_ensemble:.4f}",
            f"{avg_ml:.4f}",
            f"{avg_llm:.4f}",
            runs,
            hypothesis or "",
        ])

    console.print(f"\n[bold]Feedback:[/] {feedback_path}")
    console.print(f"[bold]Results:[/] {results_path}")
    console.print(f"\n[bold]Score: {avg_ensemble:.1%}[/] (lower is better, <55% = converged)")


@main.command()
@click.option("--author", required=True, help="Author name")
def snapshot(author):
    """Save the current skill as a new version.

    Copies the latest skill_v{N}.md to skill_v{N+1}.md so you can
    edit the new version while preserving the previous one.
    """
    workspace = get_workspace(author)
    latest = get_latest_skill(workspace)
    if latest is None:
        console.print("[red]No skill found. Run 'seed' first.[/]")
        raise SystemExit(1)

    next_v = next_skill_version(workspace)
    new_path = workspace / f"skill_v{next_v}.md"
    shutil.copy2(latest, new_path)
    console.print(f"[bold green]Snapshot:[/] {latest.name} -> {new_path.name}")
    console.print(f"Edit {new_path} and run evaluate to test your changes.")


@main.command()
@click.option("--author", required=True, help="Author name")
def status(author):
    """Show the current state of an author workspace."""
    workspace = get_workspace(author)

    console.print(f"\n[bold]Workspace:[/] {workspace}/")

    # List skill versions
    skills = sorted(workspace.glob("skill_v*.md"))
    if skills:
        console.print(f"\n[bold]Skill versions:[/] {len(skills)}")
        for s in skills:
            size = s.stat().st_size
            console.print(f"  {s.name} ({size:,} bytes)")
    else:
        console.print("\n[yellow]No skills yet. Run 'seed' to create one.[/]")

    # Show results history
    results_path = workspace / "results.tsv"
    if results_path.exists():
        console.print(f"\n[bold]Evaluation history:[/]")
        table = Table()
        table.add_column("Timestamp")
        table.add_column("Skill")
        table.add_column("Ensemble", justify="right")
        table.add_column("ML", justify="right")
        table.add_column("LLM Judge", justify="right")

        with open(results_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 5:
                    table.add_row(row[0], row[1], row[2], row[3], row[4])
        console.print(table)
    else:
        console.print("\n[dim]No evaluation results yet.[/]")


@main.command()
@click.option("--corpus", type=click.Path(exists=True), required=True, help="Path to corpus directory or file")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def analyze(corpus, verbose):
    """Analyze a corpus and display its stylometric profile."""
    setup_logging(verbose)

    from .corpus.loader import load_corpus
    from .features.extractor import extract_corpus_features

    corp = load_corpus(corpus)
    console.print(f"Loaded {len(corp)} documents from {corpus}\n")

    chunks = corp.chunks(target_length=200)
    profile = extract_corpus_features(chunks[:20])

    console.print(Panel(profile.to_summary(), title="Stylometric Profile"))


@main.command()
@click.option("--author", required=True, help="Author name")
@click.option("--corpus", type=click.Path(exists=True), default=None, multiple=True, help="Corpus directory for topic extraction (can specify multiple)")
@click.option("--skill", type=click.Path(exists=True), default=None, help="Skill file (default: latest)")
@click.option("--prompt", default=None, help="Specific topic prompt (omit to auto-extract from corpus)")
@click.option("--model", default="openai/gpt-5-mini", help="LLM model")
@click.option("--n", default=3, type=int, help="Number of samples to generate")
@click.option("--save", "save_path", default=None, help="Save samples to a file")
def generate(author, corpus, skill, prompt, model, n, save_path):
    """Generate text samples using a style skill (quick preview, no evaluation).

    Use this to quickly check how a skill performs before running a full
    evaluation. Provide --prompt for a specific topic or --corpus to
    auto-extract topics from the author's writing.
    """
    from .corpus.loader import load_corpus, load_corpora
    from .generator.generator import Generator
    from .llm.provider import LLMProvider
    from .style_skill.skill import StyleSkill

    workspace = get_workspace(author)
    if skill:
        skill_path = Path(skill)
    else:
        skill_path = get_latest_skill(workspace)
        if skill_path is None:
            console.print("[red]No skill found. Run 'seed' first.[/]")
            raise SystemExit(1)

    llm = LLMProvider()
    style_skill = StyleSkill.load(skill_path)
    gen = Generator(llm, model)

    if prompt:
        topics = [prompt]
    elif corpus:
        if len(corpus) == 1:
            corp = load_corpus(corpus[0])
        else:
            corp = load_corpora(list(corpus))
        console.print(f"Extracting topics from {len(corp)} documents...")
        topics = gen.extract_topics(corp, n=10)
        console.print(f"Topics: {', '.join(t[:50] for t in topics[:5])}...\n")
    else:
        topics = ["Write a short essay on a topic you find interesting."]

    console.print(f"Generating {n} samples with {skill_path.name} (parallel)...\n")
    samples, used_topics = gen.generate(style_skill, topics, n=n)

    output_lines = []
    for i, sample in enumerate(samples):
        word_count = len(sample.split())
        header = f"Sample {i + 1} ({word_count} words) — {skill_path.name}"
        console.print(f"[dim]Prompt: {used_topics[i][:80]}[/]")
        console.print(Panel(sample, title=header))
        output_lines.append(f"## Sample {i+1}\n\n**Prompt:** {used_topics[i]}\n\n{sample}\n")

    if save_path:
        Path(save_path).write_text("\n---\n\n".join(output_lines), encoding="utf-8")
        console.print(f"\n[bold]Saved to:[/] {save_path}")


@main.command()
@click.option("--config", "config_path", type=click.Path(), default=None, help="Path to config YAML file")
@click.option("--corpus", type=click.Path(exists=True), required=True, help="Path to corpus directory or file")
@click.option("--author", required=True, help="Name of the author to emulate")
@click.option("--model", default=None, help="LLM model")
@click.option("--max-iterations", default=None, type=int, help="Max adversarial loop iterations")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def run(config_path, corpus, author, model, max_iterations, verbose):
    """Run the full automated adversarial pipeline."""
    setup_logging(verbose)

    if config_path:
        cfg = Config.from_yaml(config_path)
    else:
        cfg = Config()

    workspace = get_workspace(author)
    cfg.corpus.path = corpus
    cfg.author_name = author
    cfg.output_dir = str(workspace)
    if model:
        cfg.llm.generator_model = model
        cfg.llm.judge_model = model
        cfg.llm.updater_model = model
    if max_iterations is not None:
        cfg.loop.max_iterations = max_iterations

    console.print(Panel(
        f"[bold]Auto Style Capture[/]\n"
        f"Author: {cfg.author_name}\n"
        f"Workspace: {workspace}/\n"
        f"Corpus: {cfg.corpus.path}\n"
        f"Model: {cfg.llm.generator_model}\n"
        f"Max iterations: {cfg.loop.max_iterations}",
        title="Configuration",
    ))

    from .pipeline import Pipeline

    callback = make_progress_callback()
    pipeline = Pipeline(cfg, progress_callback=callback)
    result = pipeline.run()

    console.print("\n")
    console.print(Panel(result.tracker.summary(), title="Convergence History"))

    if result.final_result:
        console.print(f"\n[bold]Final holdout accuracy:[/] {result.final_result.accuracy:.1%}")

    console.print(f"\n[bold]Workspace:[/] {workspace}/")
    console.print(f"[bold]Final skill:[/] skill_v{result.style_skill.version}.md")


if __name__ == "__main__":
    main()
