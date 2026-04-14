# Auto Style Capture

A GAN-inspired adversarial system for emulating writing styles. An ensemble discriminator (ML classifier + LLM judge) evaluates how well generated text matches a target author, and a style skill document gets iteratively refined until the discriminator can't tell the difference.

## Project Structure

```
auto-style-capture/
├── program.md              # Agent program — read this to run the style capture loop
├── corpus/                 # Target author corpora
│   └── paul_graham/        #   e.g., Paul Graham essays
├── skills/                 # Author workspaces (one per author)
│   └── paul_graham/
│       ├── skill_v0.md     #   Versioned skill files
│       ├── skill_v1.md
│       ├── feedback.md     #   Latest evaluation feedback
│       └── results.tsv     #   Score history
├── src/auto_style_capture/ # Pipeline code — do not modify
├── config.example.yaml     # Configuration reference
└── .env                    # API keys (not committed)
```

## CLI Commands

```bash
# Generate initial style skill from a corpus
auto-style-capture seed --corpus ./corpus/paul_graham/ --author "Paul Graham"

# Evaluate the latest skill version
auto-style-capture evaluate --author "Paul Graham" --corpus ./corpus/paul_graham/

# Evaluate a specific version
auto-style-capture evaluate --author "Paul Graham" --corpus ./corpus/paul_graham/ --skill skills/paul_graham/skill_v2.md

# Snapshot current skill to a new version (before editing)
auto-style-capture snapshot --author "Paul Graham"

# Show workspace status and score history
auto-style-capture status --author "Paul Graham"

# Analyze a corpus (show stylometric profile)
auto-style-capture analyze --corpus ./corpus/paul_graham/

# Generate text using a skill
auto-style-capture generate --author "Paul Graham" --prompt "Write about startups"

# Run the full automated pipeline (non-interactive)
auto-style-capture run --corpus ./corpus/paul_graham/ --author "Paul Graham" --max-iterations 10
```

## Environment

- Python 3.11+, dependencies in pyproject.toml
- Virtualenv at `.venv/` — activate with `source .venv/bin/activate`
- API keys in `.env` (not committed). Currently using OpenAI (`OPENAI_API_KEY`).
- Default model: `openai/gpt-5-mini` (configurable via `--model` or `config.yaml`)
- All models specified in litellm format: `provider/model-name`

## IMPORTANT RULE: Do NOT Read Corpus Files

Do NOT read any files under `corpus/`. You must learn about the author's style ONLY through the seed skill and the discriminator feedback in `skills/{author}/feedback.md`. Reading the corpus directly would let you copy verbatim phrases rather than learning genuine style patterns -- that is cheating. The evaluation harness reads the corpus; you do not. If you have already read corpus files, do not use any specific phrases or sentences you saw there in your skill edits. Focus on structural and stylistic patterns from the feedback instead.

## Key Design Decisions

- One workspace per author under `skills/` — all skill versions, feedback, and results scoped together.
- Skill versions are immutable snapshots — `snapshot` copies the latest to a new version for editing.
- `feedback.md` is overwritten each evaluation run. `results.tsv` is append-only.
- Style skills are self-contained Markdown writing guides — no references to the training process.
- ML classifier uses stable stylometric features (lexical, syntactic, punctuation, readability).
