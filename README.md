# Auto Style Capture

A GAN-inspired adversarial system for capturing and emulating writing styles. An ensemble discriminator (statistical style analysis + LLM judge) evaluates how well generated text matches a target style, and a style skill document gets iteratively refined until the discriminator can't tell the difference.

No model fine-tuning required -- the output is a Markdown style guide that any LLM can follow.

## Use Cases

**Write like yourself.** Export your blog posts, journal entries, emails, or any writing you've done. Feed them in as a corpus and the system learns your voice -- sentence rhythms, vocabulary habits, punctuation patterns, the way you structure arguments. Then use the resulting style skill whenever you want an LLM to draft something that sounds like you, not like an AI.

**Write like your favorite authors.** Collect a writer's public writing into a corpus directory and let the system distill their style into a concrete, actionable writing guide.

**Blend multiple voices.** Pass multiple corpus directories and the system balances them equally. Combine corpora from different writers and the discriminator optimizes for the blend. Each source is automatically weighted so no single voice dominates.

**Beat the AI detector.** The system identifies and corrects the statistical fingerprints that make AI-generated text detectable -- overuse of "and," artificially diverse vocabulary, missing contractions, uniform sentence lengths, em-dash habits. The resulting style skill produces text that's quantitatively indistinguishable from human writing.

## How It Works

The system has two loops:

**Inner loop (automated):** Generate text using a style skill, score it with an ensemble discriminator (statistical style analysis + reference-based LLM judge), produce qualitative writing feedback.

**Outer loop (agent-driven):** An AI agent reads the feedback, edits the style skill to close identified gaps, previews samples, runs evaluation, and repeats. The agent follows instructions in `program.md` -- a lightweight skill document in the style of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

```
                    ┌──────────────┐
                    │  Style Skill │ ◄─── Agent edits this
                    │   (Markdown) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Generator   │  LLM + skill = generated text
                    │  (parallel)  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼───────┐
       │   Style     │          │  LLM Judge   │
       │  Distance   │          │ (reference-  │
       │ (statistical│          │  based, no   │
       │  analysis)  │          │  author name)│
       └──────┬──────┘          └──────┬───────┘
              │                         │
              └────────────┬────────────┘
                    ┌──────▼───────┐
                    │   Feedback   │ ──► Agent reads this
                    │ (writing     │
                    │  advice)     │
                    └──────────────┘
```

The artifact being optimized is a **Markdown style skill** -- a self-contained writing guide that an LLM follows to produce text in a target style.

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Set your API key

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

Any LLM provider supported by [litellm](https://github.com/BerriAI/litellm) works. Set the appropriate environment variable (`ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, etc.) and pass `--model provider/model-name`.

### 3. Prepare a corpus

Create a directory with `.txt`, `.md`, or `.json` files containing the target writing. The more text, the better the style capture.

```bash
mkdir -p corpus/my_writing

# Examples of what to put in there:
# - Blog posts exported as text files
# - Journal entries
# - Emails you've written (one per file)
# - Essays, articles, or any long-form writing
# - Slack/Discord messages exported as text
```

For `.json` files, the system expects either a list of strings or a list of objects with a `text` field.

### 4. Capture the style

```bash
# Generate initial style skill from your corpus
auto-style-capture seed --corpus ./corpus/my_writing/ --author "My Style"

# Preview what the skill produces (fast, no scoring)
auto-style-capture generate --author "My Style" --corpus ./corpus/my_writing/ --n 3

# Run a scored evaluation
auto-style-capture evaluate --author "My Style" --corpus ./corpus/my_writing/ \
  --hypothesis "Baseline seed skill"

# Check your workspace
auto-style-capture status --author "My Style"
```

### 5. Iterate

The `evaluate` command writes qualitative writing advice to `skills/my_style/feedback.md` -- things like "You're chaining too many clauses with 'and'" or "Not enough contractions." Edit the skill, snapshot, and re-evaluate:

```bash
# Snapshot before editing (preserves the previous version)
auto-style-capture snapshot --author "My Style"

# Edit the skill file
# (or let an AI agent do it -- see "Agent-Driven Refinement" below)

# Re-evaluate
auto-style-capture evaluate --author "My Style" --corpus ./corpus/my_writing/ \
  --hypothesis "Added contraction guidance, broke up run-on sentences"
```

### 6. Select the best

When you're done iterating, pick the best version as your final skill:

```bash
# Auto-select best from score history
auto-style-capture select --author "My Style"

# Or manually pick a specific version
auto-style-capture select --author "My Style" --version 3
```

The final skill is saved as `skills/my_style/skill.md` -- ready to use with any LLM.

## Blending Multiple Voices

Pass multiple `--corpus` flags to blend styles. Each source is automatically balanced so no single voice dominates regardless of how much text each has:

```bash
# Seed from a blend
auto-style-capture seed \
  --corpus ./corpus/my_blog_posts/ \
  --corpus ./corpus/favorite_author/ \
  --corpus ./corpus/technical_writing/ \
  --author "My Blend"

# Evaluate against the same blend
auto-style-capture evaluate \
  --corpus ./corpus/my_blog_posts/ \
  --corpus ./corpus/favorite_author/ \
  --corpus ./corpus/technical_writing/ \
  --author "My Blend" --runs 2
```

## Continuing Previous Runs

Start from an existing skill instead of generating from scratch:

```bash
# Continue from a previous run's best skill
auto-style-capture seed --author "My Style v2" \
  --from-skill ./skills/my_style/skill.md
```

## Agent-Driven Refinement

The system is designed to be driven by an AI agent following `program.md`:

1. Agent reads the current style skill
2. Previews samples with `generate` (fast, ~10s)
3. Runs `evaluate` with a `--hypothesis` describing what changed
4. Reads `feedback.md` for qualitative writing advice
5. Edits the skill to close identified gaps
6. Repeats until satisfied, then runs `select` to pick the best version

Each workspace under `skills/` tracks versioned skills, score history, and feedback:

```
skills/my_style/
├── skill.md          # Final selected skill (the output)
├── skill_v0.md       # Seed (auto-generated)
├── skill_v1.md       # Agent's first revision
├── skill_v2.md       # Agent's second revision
├── feedback.md       # Latest evaluation feedback
├── results.tsv       # Full score history with hypotheses
└── topics.json       # Cached topic prompts
```

## CLI Reference

| Command | Purpose | Speed |
|---------|---------|-------|
| `seed` | Generate initial skill from corpus | ~15s |
| `seed --from-skill` | Continue from an existing skill | Instant |
| `generate` | Preview samples, no scoring | ~10s |
| `evaluate` | Full scoring with feedback | ~45s |
| `evaluate --runs 3` | Averaged scoring, more stable | ~2min |
| `evaluate --judge-model` | Use a stronger model for the judge | ~1min |
| `snapshot` | Version the skill before editing | Instant |
| `select` | Pick best version as final `skill.md` | Instant |
| `status` | See version history and scores | Instant |
| `analyze` | Show corpus stylometric profile | Instant |

## How the Discriminator Works

**Style Distance Analysis:** Extracts ~80 stylometric features (sentence length distributions, punctuation rates, vocabulary richness, function word frequencies, readability scores) from real corpus chunks and generated text. Compares distributions using statistical tests (Mann-Whitney U) and effect sizes (Cohen's d). Reports qualitative writing advice ranked by importance -- e.g., "You're chaining too many clauses with 'and'. Break compound sentences into shorter ones." Works reliably with small sample sizes, unlike a trained classifier.

**LLM Judge:** Given anonymous reference passages from the corpus (no author name, no identity hints), evaluates blinded A/B pairs of real vs generated text purely on stylistic similarity. References and evaluation pairs are strictly separated so the judge can't match on content. Scores 6 dimensions: voice/tone, sentence rhythm, vocabulary, punctuation, rhetorical devices, overall authenticity. Use `--judge-model` to specify a stronger model for more rigorous evaluation.

**Ensemble:** Weighted average of both (default 50/50). Lower is better -- 50% means the discriminator can't tell real from generated.

## What the System Learns

From experiments across diverse corpora, the system consistently identifies these differences between human and AI writing:

- **Sentence rhythm:** Humans write with more variance and positive skew in sentence length. AI defaults to uniform medium-length sentences or overuses short punchy fragments.
- **Function word density:** Human writing has higher rates of common words like "the," "of," "it," "this." AI reaches for varied vocabulary, creating artificially high lexical richness.
- **Punctuation fingerprint:** AI overuses em dashes and semicolons. Human writers rely more on commas and parentheses.
- **Contractions:** Human writing uses contractions naturally throughout. AI tends formal unless explicitly instructed.
- **Word repetition:** Humans repeat common words freely. AI avoids repetition, searching for synonyms -- a strong statistical signal.
- **"And" overuse:** LLMs chain clauses with "and" at roughly 2x the human rate. This is one of the strongest and hardest-to-fix AI tells.

## Configuration

Default model is `openai/gpt-5-mini` via [litellm](https://github.com/BerriAI/litellm). Override per command or in a config file:

```bash
# Use a different model for generation
auto-style-capture evaluate --model anthropic/claude-sonnet-4-20250514 ...

# Use a stronger model just for the judge
auto-style-capture evaluate --judge-model openai/gpt-5.1 ...

# Via config file
cp config.example.yaml config.yaml
```

See `config.example.yaml` for all options.
