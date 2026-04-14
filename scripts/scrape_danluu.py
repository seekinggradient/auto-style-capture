#!/usr/bin/env python3
"""Scrape blog posts from danluu.com using requests + html.parser."""

import os
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

BASE_URL = "https://danluu.com"
OUTPUT_DIR = "corpus/dan_luu"
MAX_POSTS = 30
DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-scraper; polite; +akshay@example.com)"
}


# ---------------------------------------------------------------------------
# HTML parsers
# ---------------------------------------------------------------------------

class LinkExtractor(HTMLParser):
    """Extract post links from the danluu.com index page."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            # Links may be relative (/slug/) or absolute (https://danluu.com/slug/)
            # Normalise to relative paths
            if href.startswith(BASE_URL):
                href = href[len(BASE_URL):]
            if href.startswith("/") and not href.startswith("//"):
                skip = {"/#", "/atom.xml", "/atom/", "/rss/", "/rss.xml", "/"}
                if href not in skip:
                    self.links.append(href)


class ArticleExtractor(HTMLParser):
    """Extract the main text content from a danluu.com post page.

    danluu.com wraps post content in <main>...</main>.
    We only capture text inside <main>, skipping <script>/<style> blocks.
    """

    # Void elements that never have a closing tag
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self._in_main = False
        self._skip_tag = None       # currently skipping this tag's content
        self._skip_depth = 0
        self._text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        # Track skip regions (script/style) — only when inside main
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return

        if tag == "main":
            self._in_main = True
            return

        if not self._in_main:
            return

        if tag in ("script", "style"):
            self._skip_tag = tag
            self._skip_depth = 1
            return

        if tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4",
                    "h5", "h6", "blockquote", "tr", "hr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return

        if tag == "main":
            self._in_main = False
            return

        if not self._in_main:
            return

        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4",
                    "h5", "h6", "blockquote", "tr"):
            self._text_parts.append("\n")

    def handle_data(self, data):
        if self._in_main and not self._skip_tag:
            self._text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        # Collapse whitespace within lines, keep paragraph breaks
        lines = raw.splitlines()
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        text = "\n".join(cleaned)
        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slug_from_path(path: str) -> str:
    """Turn /some-slug/ into 'some-slug'."""
    return path.strip("/").replace("/", "_") or "index"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Fetch index page
    print("Fetching index page...")
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    parser = LinkExtractor()
    parser.feed(resp.text)

    # Deduplicate while preserving order
    seen = set()
    unique_links = []
    for link in parser.links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    post_links = unique_links[:MAX_POSTS]
    print(f"Found {len(parser.links)} links, fetching up to {MAX_POSTS} unique posts...")

    # 2. Fetch each post
    saved = 0
    total_words = 0

    for i, path in enumerate(post_links):
        slug = slug_from_path(path)
        url = urljoin(BASE_URL, path)
        print(f"  [{i+1}/{len(post_links)}] {url} -> {slug}.txt", end=" ")

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"FAILED: {e}")
            time.sleep(DELAY)
            continue

        extractor = ArticleExtractor()
        extractor.feed(r.text)
        text = extractor.get_text()

        if not text or len(text) < 100:
            print(f"(too short, {len(text)} chars — skipped)")
            time.sleep(DELAY)
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        wc = len(text.split())
        total_words += wc
        saved += 1
        print(f"({wc} words)")

        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} posts, {total_words:,} total words.")


if __name__ == "__main__":
    main()
