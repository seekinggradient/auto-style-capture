#!/usr/bin/env python3
"""Scrape essays from gwern.net using requests + html.parser."""

import os
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

BASE_URL = "https://gwern.net"
OUTPUT_DIR = "corpus/gwern"
MAX_POSTS = 25
MIN_WORDS = 1000
DELAY = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-scraper; polite; +akshay@example.com)"
}

# Curated list of Gwern's major essays (long-form, substantive content).
# These are known to be real essays rather than index/doc pages.
ESSAY_PATHS = [
    "/spaced-repetition",
    "/scaling-hypothesis",
    "/bitcoin-is-worse-is-better",
    "/silk-road",
    "/death-note-anonymity",
    "/dnm-archive",
    "/complement",
    "/embryo-selection",
    "/longevity",
    "/tool-ai",
    "/everything",
    "/replication",
    "/prediction-market",
    "/modafinil",
    "/melatonin",
    "/dnb-faq",
    "/dnb-meta-analysis",
    "/iodine",
    "/creatine",
    "/narrowing-circle",
    "/archiving",
    "/copyright",
    "/terrorism-is-not-effective",
    "/littlewood",
    "/conscientiousness",
    "/sunk-cost",
    "/forking-path",
    "/choosing-software",
    "/fiction/missing-cities",
    "/improvement",
    "/correlation",
    "/causality",
    "/google-shutdown",
    "/ab-test",
    "/order-statistic",
    "/selection",
    "/tank",
    "/gpt-3",
    "/timing",
    "/design",
]


# ---------------------------------------------------------------------------
# HTML parser for Gwern essays
# ---------------------------------------------------------------------------

class GwernArticleExtractor(HTMLParser):
    """Extract main text from a gwern.net essay page.

    Gwern's pages wrap the essay body in <div id="markdownBody">.
    We skip: footnotes, sidenotes, navigation, metadata, TOC,
    link-bibliography sections, and script/style blocks.

    Strategy: use a simple regex pre-pass to extract the markdownBody div,
    then parse that fragment. This avoids depth-tracking issues with the
    full page HTML.
    """

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    # Tokens in class or id that trigger skipping the element and its children
    SKIP_TOKENS = {"footnote", "footnotes", "sidenote", "sidebar",
                   "backlink", "backlinks", "link-bibliography",
                   "aux-links", "page-metadata", "page-description",
                   "link-tag", "directory-indexes", "TOC",
                   "collapse", "abstract-collapse"}

    BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "blockquote", "tr", "hr", "br", "ul", "ol", "section",
                  "figcaption", "dt", "dd"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0          # >0 means we're inside a skipped region
        self._skip_stack: list[str] = []  # tags opened inside skip regions
        self._skip_script_tag = None
        self._skip_script_depth = 0
        self._text_parts: list[str] = []

    def _should_skip(self, attrs: list[tuple[str, str | None]]) -> bool:
        """Check if element's class/id contains a skip token."""
        d = dict(attrs)
        cls = (d.get("class", "") or "")
        id_val = (d.get("id", "") or "")
        combined = cls + " " + id_val
        for token in self.SKIP_TOKENS:
            if token in combined:
                return True
        return False

    def handle_starttag(self, tag, attrs):
        # Handle script/style skip
        if self._skip_script_tag:
            if tag == self._skip_script_tag:
                self._skip_script_depth += 1
            return

        if tag in ("script", "style"):
            self._skip_script_tag = tag
            self._skip_script_depth = 1
            return

        # Skip nav elements entirely
        if tag == "nav":
            self._skip_depth += 1
            self._skip_stack.append(tag)
            return

        # Check if this element should be skipped
        if tag not in self.VOID and self._should_skip(attrs):
            self._skip_depth += 1
            self._skip_stack.append(tag)
            return

        # Track nested tags inside skip regions
        if self._skip_depth > 0:
            if tag not in self.VOID:
                self._skip_stack.append(tag)
            return

        # Normal content
        if tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip_script_tag:
            if tag == self._skip_script_tag:
                self._skip_script_depth -= 1
                if self._skip_script_depth <= 0:
                    self._skip_script_tag = None
            return

        if self._skip_depth > 0:
            # Pop from stack if it matches
            if self._skip_stack and self._skip_stack[-1] == tag:
                self._skip_stack.pop()
                self._skip_depth -= 1
            elif self._skip_stack:
                # Mismatched close tag — try to find it in the stack
                for i in range(len(self._skip_stack) - 1, -1, -1):
                    if self._skip_stack[i] == tag:
                        self._skip_stack.pop(i)
                        self._skip_depth -= 1
                        break
            return

        if tag in self.BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and not self._skip_script_tag:
            self._text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        lines = raw.splitlines()
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        text = "\n".join(cleaned)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        # Strip trailing JS/navigation boilerplate that leaks through
        for marker in [
            "[Error: JavaScript disabled.]",
            "[Backlinks, similar links",
            "Similar Links",
            "[Similar links by topic]",
            "Backlinks",
        ]:
            idx = text.find(marker)
            if idx > 0 and idx > len(text) * 0.8:
                text = text[:idx].rstrip()
        return text


def extract_markdown_body(html: str) -> str:
    """Extract the content of <div id="markdownBody" ...> from the full page HTML.

    This avoids parsing the entire page and dealing with nav/header/metadata
    that sits outside the essay body.
    """
    # Find the opening tag
    marker = 'id="markdownBody"'
    idx = html.find(marker)
    if idx < 0:
        return html  # fallback: parse entire page

    # Walk backward to find the opening <div
    start = html.rfind("<div", 0, idx)
    if start < 0:
        return html

    # Find the closing angle bracket of the opening tag
    tag_end = html.find(">", idx)
    if tag_end < 0:
        return html

    # Now find the matching closing </div> by counting depth
    depth = 1
    pos = tag_end + 1
    while depth > 0 and pos < len(html):
        next_open = html.find("<div", pos)
        next_close = html.find("</div>", pos)

        if next_close < 0:
            break

        if next_open >= 0 and next_open < next_close:
            depth += 1
            pos = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                return html[tag_end + 1:next_close]
            pos = next_close + 6

    # Fallback: return everything after the opening tag
    return html[tag_end + 1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slug_from_path(path: str) -> str:
    """Turn /some-slug into 'some-slug', /fiction/foo into 'fiction_foo'."""
    return path.strip("/").replace("/", "_") or "index"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    post_paths = ESSAY_PATHS[:MAX_POSTS]
    print(f"Fetching up to {MAX_POSTS} essays from gwern.net...")

    saved = 0
    total_words = 0

    for i, path in enumerate(post_paths):
        slug = slug_from_path(path)
        url = urljoin(BASE_URL, path)
        print(f"  [{i+1}/{len(post_paths)}] {url} -> {slug}.txt", end=" ", flush=True)

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"FAILED: {e}")
            time.sleep(DELAY)
            continue

        body_html = extract_markdown_body(r.text)
        extractor = GwernArticleExtractor()
        extractor.feed(body_html)
        text = extractor.get_text()

        wc = len(text.split())
        if wc < MIN_WORDS:
            print(f"(only {wc} words — skipped, need {MIN_WORDS}+)")
            time.sleep(DELAY)
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        total_words += wc
        saved += 1
        print(f"({wc:,} words)")

        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} essays, {total_words:,} total words.")


if __name__ == "__main__":
    main()
