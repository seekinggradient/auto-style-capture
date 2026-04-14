# Auto Style Capture

A GAN-inspired adversarial system for capturing and emulating writing styles. An ensemble discriminator (ML classifier + LLM judge) evaluates how well generated text matches a target style, and a style skill document gets iteratively refined until the discriminator can't tell the difference.

No model fine-tuning required -- the output is a Markdown style guide that any LLM can follow.

## Use Cases

**Write like yourself.** Export your blog posts, journal entries, emails, or any writing you've done. Feed them in as a corpus and the system learns your voice -- sentence rhythms, vocabulary habits, punctuation patterns, the way you structure arguments. Then use the resulting style skill whenever you want an LLM to draft something that sounds like you, not like an AI.

**Write like your favorite authors.** Have a writer whose style you admire? Collect their public writing into a corpus directory and let the system distill their style into a concrete, actionable writing guide. Useful for learning what makes a particular voice work.

**Blend multiple voices.** Pass multiple corpus directories and the system balances them equally. Want Paul Graham's directness mixed with the enthusiasm of a technical explainer? Combine corpora and the discriminator optimizes for the blend. Each source is automatically weighted so no single voice dominates.

**Beat the AI detector.** The system specifically identifies and corrects the statistical fingerprints that make AI-generated text detectable -- overuse of "and," artificially diverse vocabulary, missing contractions, uniform sentence lengths, em-dash habits. The resulting style skill produces text that's quantitatively indistinguishable from human writing.

## How It Works

The system has two loops:

**Inner loop (automated):** Generate text using a style skill, score it with an ensemble discriminator (stylometric ML classifier + reference-based LLM judge), produce detailed feedback.

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
       │ ML Classifier│          │  LLM Judge   │
       │ (stylometric │          │ (reference-  │
       │  features)   │          │  based, no   │
       └──────┬──────┘          │  author name)│
              │                  └──────┬───────┘
              └────────────┬────────────┘
                    ┌──────▼───────┐
                    │   Feedback   │ ──► Agent reads this
                    │  (feedback.md│
                    │  results.tsv)│
                    └──────────────┘
```

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

The `evaluate` command writes detailed feedback to `skills/my_style/feedback.md` showing exactly what's different between your writing and the generated text. Edit the skill, snapshot, and re-evaluate:

```bash
# Snapshot before editing (preserves the previous version)
auto-style-capture snapshot --author "My Style"

# Edit the skill file
# (or let an AI agent do it -- see "Agent-Driven Refinement" below)

# Re-evaluate
auto-style-capture evaluate --author "My Style" --corpus ./corpus/my_writing/ \
  --hypothesis "Increased contraction rate, shortened sentences"
```

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

## Agent-Driven Refinement

The system is designed to be driven by an AI agent following `program.md`:

1. Agent reads the current style skill
2. Previews samples with `generate` (fast, ~10s)
3. Runs `evaluate` with a `--hypothesis` describing what changed
4. Reads `feedback.md` for discriminator analysis
5. Edits the skill to close identified gaps
6. Repeats until the discriminator can't distinguish real from generated (< 55% accuracy)

Each workspace under `skills/` tracks versioned skills, score history, and feedback:

```
skills/my_style/
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
| `generate` | Preview samples, no scoring | ~10s |
| `evaluate` | Full scoring with feedback | ~45s |
| `evaluate --runs 3` | Averaged scoring, more stable | ~2min |
| `snapshot` | Version the skill before editing | Instant |
| `status` | See version history and scores | Instant |
| `analyze` | Show corpus stylometric profile | Instant |

## How the Discriminator Works

**ML Classifier:** Extracts ~80 stylometric features (sentence length distributions, punctuation rates, vocabulary richness, function word frequencies, readability scores) from real corpus chunks and generated text. Trains a GradientBoosting classifier to distinguish them. Classes are balanced so large corpora don't inflate accuracy. Reports which features are most distinguishing -- this is the actionable feedback that drives skill improvements.

**LLM Judge:** Given anonymous reference passages from the corpus (no author name, no identity hints), evaluates blinded pairs of real vs generated text purely on stylistic similarity. Scores 6 dimensions: voice/tone, sentence rhythm, vocabulary, punctuation, rhetorical devices, overall authenticity. Uses structured outputs for reliable JSON scoring.

**Ensemble:** Weighted average of both (default 50/50). Target is < 55% accuracy -- meaning the discriminator is barely better than a coin flip at telling real from generated.

## What the System Learns

From experiments across diverse corpora, the system consistently identifies these differences between human and AI writing:

- **Sentence rhythm:** Humans write with more variance and positive skew in sentence length. AI defaults to uniform medium-length sentences or overuses short punchy fragments.
- **Function word density:** Human writing has higher rates of common words like "the," "of," "it," "this." AI reaches for varied vocabulary, creating artificially high lexical richness.
- **Punctuation fingerprint:** AI overuses em dashes and colons. Human writers rely more on commas and parentheses.
- **Contractions:** Human writing uses ~2-3% contractions naturally. AI tends formal unless explicitly instructed.
- **Word repetition:** Humans repeat common words freely. AI avoids repetition, searching for synonyms -- a strong classifier signal.
- **"And" overuse:** LLMs chain clauses with "and" at roughly 2x the human rate. This is one of the strongest and hardest-to-fix AI tells.

## Configuration

Default model is `openai/gpt-5-mini` via [litellm](https://github.com/BerriAI/litellm). Override per command or in a config file:

```bash
# Per command
auto-style-capture evaluate --model anthropic/claude-sonnet-4-20250514 ...

# Via config file
cp config.example.yaml config.yaml
# Edit config.yaml with your preferred models and settings
```

See `config.example.yaml` for all options including separate models for generation, judging, and skill updates.
