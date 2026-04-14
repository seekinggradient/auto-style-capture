# Auto Style Capture

A GAN-inspired adversarial system for capturing and emulating writing styles. An ensemble discriminator (ML classifier + LLM judge) evaluates how well generated text matches a target style, and a style skill document gets iteratively refined until the discriminator can't tell the difference.

## How It Works

The system has two loops:

**Inner loop (automated):** Generate text using a style skill, score it with an ensemble discriminator (stylometric ML classifier + reference-based LLM judge), produce feedback.

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

The artifact being optimized is a **Markdown style skill** -- a self-contained writing guide that an LLM follows to produce text in a target style. No model fine-tuning required.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# Scrape a corpus (Paul Graham as an example)
python scripts/fetch_pg_essays.py

# Generate initial style skill
auto-style-capture seed --corpus ./corpus/paul_graham/ --author "Paul Graham"

# Preview generated samples (fast, no scoring)
auto-style-capture generate --author "Paul Graham" --corpus ./corpus/paul_graham/ --n 3

# Run evaluation
auto-style-capture evaluate --author "Paul Graham" --corpus ./corpus/paul_graham/ \
  --hypothesis "Baseline seed skill"

# Check workspace status
auto-style-capture status --author "Paul Graham"
```

## Blended Styles

Capture a blend of multiple authors by passing multiple `--corpus` flags. Each source is automatically balanced so no single author dominates:

```bash
auto-style-capture seed \
  --corpus ./corpus/paul_graham/ \
  --corpus ./corpus/julia_evans/ \
  --corpus ./corpus/morgan_housel/ \
  --author "Human Writer"

auto-style-capture evaluate \
  --corpus ./corpus/paul_graham/ \
  --corpus ./corpus/julia_evans/ \
  --corpus ./corpus/morgan_housel/ \
  --author "Human Writer" --runs 2
```

## Agent-Driven Refinement

The system is designed to be driven by an AI agent following `program.md`:

1. Agent reads the style skill
2. Previews samples with `generate`
3. Runs `evaluate` with a `--hypothesis`
4. Reads `feedback.md` for discriminator analysis
5. Edits the skill to close gaps
6. Repeats until the discriminator can't distinguish real from generated (< 55% accuracy)

Each author workspace under `skills/` tracks versioned skills, score history, and feedback:

```
skills/paul_graham/
├── skill_v0.md       # Seed
├── skill_v1.md       # Iteration 1
├── skill_v2.md       # Iteration 2
├── feedback.md       # Latest eval feedback
├── results.tsv       # Score history with hypotheses
└── topics.json       # Cached topic prompts
```

## CLI Commands

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

**ML Classifier:** Extracts ~80 stylometric features (sentence lengths, punctuation rates, vocabulary richness, function word frequencies, readability scores) from real corpus chunks and generated text. Trains a GradientBoosting classifier to distinguish them. Reports accuracy and which features are most distinguishing.

**LLM Judge:** Given reference passages from the corpus (anonymized, no author name), evaluates blinded pairs of real vs generated text on style similarity. Scores 6 dimensions: voice/tone, sentence rhythm, vocabulary, punctuation, rhetorical devices, overall authenticity. Uses structured outputs for reliable scoring.

**Ensemble:** Weighted average of both (default 50/50). Target is < 55% accuracy (barely better than random chance).

## Corpus Scrapers

Scripts to build corpora from public blogs (under `scripts/`):

| Script | Author | Source |
|--------|--------|--------|
| `fetch_pg_essays.py` | Paul Graham | paulgraham.com |
| `scrape_julia_evans.py` | Julia Evans | jvns.ca |
| `scrape_danluu.py` | Dan Luu | danluu.com |
| `scrape_patio11.py` | Patrick McKenzie | kalzumeus.com |
| `scrape_willison.py` | Simon Willison | simonwillison.net |
| `scrape_seth_godin.py` | Seth Godin | seths.blog |
| `scrape_scott_alexander.py` | Scott Alexander | astralcodexten.substack.com |
| `scrape_wbw.py` | Tim Urban | waitbutwhy.com |
| `scrape_morgan_housel.py` | Morgan Housel | collabfund.com/blog |
| `scrape_gwern.py` | Gwern Branwen | gwern.net |

## Configuration

Default model is `openai/gpt-5-mini` via [litellm](https://github.com/BerriAI/litellm). Any provider supported by litellm works:

```bash
# Use a different model
auto-style-capture evaluate --model anthropic/claude-sonnet-4-20250514 ...

# Or configure in config.yaml
cp config.example.yaml config.yaml
```

See `config.example.yaml` for all options.
