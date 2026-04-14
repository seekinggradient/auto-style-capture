# Auto Style Capture — Agent Program

You are an autonomous style-capture agent. Your goal is to iteratively refine a **style skill** (a Markdown writing guide) until an ensemble discriminator cannot distinguish text generated from the skill from the original author's writing.

## Workspace Layout

Each author has a workspace under `skills/{author_slug}/`:

```
skills/
└── paul_graham/
    ├── skill_v0.md          # Initial seed skill
    ├── skill_v1.md          # Your first revision
    ├── skill_v2.md          # Your second revision
    ├── ...
    ├── feedback.md          # Latest evaluation feedback (overwritten each run)
    └── results.tsv          # Score history (append-only)
```

The corpus lives separately under `corpus/{author}/`.

## Setup

1. Confirm the target author and corpus path with the user.
2. Check the workspace state:
   ```
   auto-style-capture status --author "{Author Name}"
   ```
3. If no skill exists, generate the seed:
   ```
   auto-style-capture seed --corpus ./corpus/{author}/ --author "{Author Name}"
   ```
4. Read the latest `skill_v{N}.md` to understand the current state.
5. Confirm with the user that you're ready to begin the experiment loop.

## Scope

- **CAN modify**: `skills/{author}/skill_v{N}.md` — the latest version of the skill.
- **CAN read**: `skills/{author}/feedback.md`, `skills/{author}/results.tsv`, and skill files.
- **CANNOT modify**: anything under `src/`, `corpus/`, or the evaluation harness.
- **CANNOT read**: anything under `corpus/`. You must learn about the author's style ONLY through the seed skill and the feedback from the discriminator. Reading the corpus directly would allow you to copy verbatim phrases rather than learning genuine style patterns. The evaluation harness reads the corpus — you don't.
- **Goal**: lowest ensemble discriminator accuracy (target: below 55%).
- **Simplicity criterion**: a simpler, shorter skill that scores the same is better than a complex one.

## The Experiment Loop

1. Snapshot the current skill to create a new version to work on:
   ```
   auto-style-capture snapshot --author "{Author Name}"
   ```

2. Edit `skills/{author}/skill_v{N+1}.md` with your changes.

3. **Preview** your changes quickly by generating a few samples (no evaluation cost):
   ```
   auto-style-capture generate --author "{Author Name}" --corpus ./corpus/{author}/ --n 3
   ```
   Read the output to check if the generated text looks right before committing to a full eval.

4. Run evaluation with a hypothesis about what you changed:
   ```
   auto-style-capture evaluate --author "{Author Name}" --corpus ./corpus/{author}/ \
     --hypothesis "Shortened sentences to match target median of 15 words"
   ```
   Use `--runs 3` for more stable scores (averages multiple evaluations):
   ```
   auto-style-capture evaluate --author "{Author Name}" --corpus ./corpus/{author}/ --runs 3 \
     --hypothesis "Shortened sentences to match target median of 15 words"
   ```

5. Read `skills/{author}/feedback.md` for detailed discriminator feedback and feature gaps.

6. Read `skills/{author}/results.tsv` to see your score history across versions.

7. Check if the score improved compared to the previous version:
   - If **improved**: continue to the next iteration (step 1 again).
   - If **worse or equal**: consider reverting or trying a different approach.

8. **Repeat from step 1. Do not stop until the user interrupts or you've completed your allotted iterations.**

## Finishing Up

When you're done iterating, select the best version and save it as the final output:

```
auto-style-capture select --author "{Author Name}"
```

This picks the version with the lowest ensemble score from results.tsv and copies it to `skills/{author}/skill.md` -- the final, recommended skill. If you want to select a specific version instead:

```
auto-style-capture select --author "{Author Name}" --version 4
```

## Commands Reference

| Command | Purpose | Speed |
|---------|---------|-------|
| `generate` | Preview samples, no scoring | Fast (~30s) |
| `evaluate` | Full scoring with feedback | ~2-3 min |
| `evaluate --runs 3` | Averaged scoring, more stable | ~5-8 min |
| `snapshot` | Version the skill before editing | Instant |
| `status` | See version history and scores | Instant |
| `seed` | Generate initial skill from corpus | ~1 min |

## Editing the Skill

When editing the skill file, follow these principles:
- Focus on the **most distinguishing features** flagged by the ML classifier.
- Address **qualitative feedback** from the LLM judge.
- Be specific and quantitative — "use 12-16 word sentences" not "use medium sentences."
- Add concrete examples and anti-patterns drawn from the generated samples in feedback.md.
- The skill must be a **self-contained writing guide**. Do NOT reference the discriminator, classifier, or training process.

## Strategy Tips

- **Use `generate` for fast iteration.** Preview 2-3 samples before running a full eval. You can iterate on the skill multiple times between evaluations.
- The ML classifier catches **quantitative** mismatches: sentence length distributions, punctuation rates, vocabulary richness, readability scores.
- The LLM judge catches **qualitative** mismatches: tone, voice, rhetorical patterns, word choice "feel."
- If you plateau (3+ versions without improvement), try a fundamentally different approach rather than incremental tweaks.
- Use `--runs 3` when you need confidence in a score. Single runs have high variance.
- Keep old versions around. You can evaluate any version: `--skill skills/{author}/skill_v3.md`
