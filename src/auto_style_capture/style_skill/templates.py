from __future__ import annotations

from ..corpus.models import Corpus
from ..features.extractor import extract_corpus_features
from ..llm.provider import LLMProvider
from .skill import StyleSkill

SEED_GENERATION_PROMPT = """\
You are an expert stylometrist and writing coach. Analyze the following writing samples \
from {author_name} and create a detailed style guide that would allow someone to write \
in this exact style.

## Writing Samples

{samples}

## Stylometric Profile

{profile}

## Instructions

Create a comprehensive style skill document in Markdown format. The document must have \
these sections:

1. **Voice & Tone** - Overall attitude, emotional register, level of formality
2. **Sentence Structure** - Preferred sentence lengths (be SPECIFIC with word counts), \
use of fragments, compound vs simple sentences, ratio of short to long sentences
3. **Vocabulary & Diction** - Word choice patterns, formality level, characteristic \
words/phrases, preference for simple vs complex words
4. **Punctuation & Formatting** - Distinctive punctuation habits (quantify: e.g., \
"uses semicolons approximately once per 100 words"), paragraph structure
5. **Rhythm & Pacing** - Cadence, repetition patterns, how paragraphs transition
6. **Rhetorical Devices** - Metaphor style, use of irony, dialogue patterns, \
characteristic literary devices
7. **Characteristic Patterns** - Specific tics, recurring phrases, opening/closing \
patterns, structural signatures
8. **Anti-patterns (What to Avoid)** - Common LLM habits that diverge from this \
author's style (e.g., "Never use bullet points", "Avoid hedging language like \
'it's worth noting'")

Be CONCRETE and QUANTITATIVE wherever possible. Instead of "uses varied sentence \
lengths," say "alternates between short declarative sentences (5-8 words) and long \
complex sentences with 2-3 subordinate clauses (25-35 words), with a 60/40 ratio."

Start the document with: # Style Skill: {author_name}
"""


def generate_initial_skill(
    corpus: Corpus,
    author_name: str,
    llm: LLMProvider,
    model: str,
    temperature: float = 0.4,
) -> StyleSkill:
    # Sample representative texts
    sample_docs = corpus.sample(min(5, len(corpus)))
    samples_text = "\n\n---\n\n".join(
        f"### Sample {i+1}\n\n{doc.text[:2000]}"
        for i, doc in enumerate(sample_docs)
    )

    # Extract stylometric profile
    chunks = corpus.chunks(target_length=200)
    profile = extract_corpus_features(chunks[:20])

    prompt = SEED_GENERATION_PROMPT.format(
        author_name=author_name,
        samples=samples_text,
        profile=profile.to_summary(),
    )

    content = llm.complete(
        model=model,
        system="You are an expert in computational stylistics and authorship analysis.",
        user=prompt,
        temperature=temperature,
        max_tokens=16000,
        purpose="initial_skill_generation",
    )

    return StyleSkill(author_name=author_name, content=content, version=0)
