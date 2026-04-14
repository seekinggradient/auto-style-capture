"""
Scrape up to 30 blog posts from Astral Codex Ten (Scott Alexander) on Substack.
Uses requests + html.parser only. Saves posts with 500+ words as .txt files.
"""

import json
import os
import re
import time
from html.parser import HTMLParser

import requests

BASE = "https://www.astralcodexten.com"
ARCHIVE_API = f"{BASE}/api/v1/archive?sort=new&search=&offset={{offset}}&limit={{limit}}"
TARGET_POSTS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus", "scott_alexander")
DELAY = 0.5
MIN_WORDS = 500

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-scraper; polite; +rate-limited)",
    "Accept": "text/html,application/xhtml+xml,application/json",
}


class ArticleTextExtractor(HTMLParser):
    """Extract readable text from Substack article HTML, skipping images/buttons."""

    SKIP_TAGS = {"script", "style", "svg", "button", "nav", "footer", "figcaption"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        # Add line breaks for block elements
        if tag in ("p", "br", "h1", "h2", "h3", "h4", "li", "blockquote", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in ("p", "h1", "h2", "h3", "h4", "li", "blockquote"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.parts)
        # Collapse whitespace within lines, preserve paragraph breaks
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            if stripped:
                cleaned.append(stripped)
        return "\n\n".join(cleaned)


def fetch_post_list() -> list[dict]:
    """Fetch posts from the Substack archive API, paginating until we have enough."""
    all_posts: list[dict] = []
    offset = 0
    batch_size = 30

    while len(all_posts) < TARGET_POSTS * 2:  # fetch extra to account for skips
        url = ARCHIVE_API.format(offset=offset, limit=batch_size)
        print(f"Fetching archive list (offset={offset})...")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_posts.extend(batch)
        offset += len(batch)
        print(f"  Got {len(batch)} posts (total so far: {len(all_posts)})")
        time.sleep(DELAY)

    print(f"Total posts fetched from API: {len(all_posts)}")
    return all_posts


def fetch_post_html(slug: str) -> str | None:
    """Fetch a single post page and return the body HTML from embedded JSON."""
    url = f"{BASE}/p/{slug}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  SKIP {slug}: HTTP {resp.status_code}")
        return None
    html = resp.text

    # Strategy 1: Extract body_html from the JSON blob in window._preloads
    # Look for "body_html":" pattern and extract the JSON string value
    marker = '"body_html":"'
    idx = html.find(marker)
    if idx == -1:
        # Strategy 2: Parse the visible <div class="body markup"> region
        return extract_body_from_page(html)

    start = idx + len(marker)
    # Find the end of this JSON string value — scan for unescaped quote
    body_html = _extract_json_string(html, start)
    if body_html:
        return body_html

    return extract_body_from_page(html)


def _extract_json_string(text: str, start: int) -> str | None:
    """Extract a JSON-encoded string value starting at `start` (after opening quote)."""
    i = start
    chars = []
    while i < len(text):
        c = text[i]
        if c == '"':
            break
        if c == '\\':
            i += 1
            if i >= len(text):
                break
            nc = text[i]
            if nc == 'n':
                chars.append('\n')
            elif nc == 't':
                chars.append('\t')
            elif nc == '"':
                chars.append('"')
            elif nc == '\\':
                chars.append('\\')
            elif nc == '/':
                chars.append('/')
            elif nc == 'u':
                hex_str = text[i + 1:i + 5]
                try:
                    chars.append(chr(int(hex_str, 16)))
                except (ValueError, IndexError):
                    chars.append(nc)
                i += 4
            else:
                chars.append(nc)
        else:
            chars.append(c)
        i += 1
    return "".join(chars) if chars else None


class BodyMarkupExtractor(HTMLParser):
    """Fallback: extract text from <div class="body markup"> on the rendered page."""

    def __init__(self):
        super().__init__()
        self.in_body = False
        self._depth = 0
        self._inner = ArticleTextExtractor()

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and "body" in attrs_dict.get("class", "") and "markup" in attrs_dict.get("class", ""):
            self.in_body = True
            self._depth = 1
            return
        if self.in_body:
            if tag == "div":
                self._depth += 1
            self._inner.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self.in_body:
            if tag == "div":
                self._depth -= 1
                if self._depth <= 0:
                    self.in_body = False
                    return
            self._inner.handle_endtag(tag)

    def handle_data(self, data):
        if self.in_body:
            self._inner.handle_data(data)

    def get_text(self) -> str:
        return self._inner.get_text()


def extract_body_from_page(html: str) -> str | None:
    parser = BodyMarkupExtractor()
    parser.feed(html)
    text = parser.get_text()
    return text if text.strip() else None


def html_to_text(body_html: str) -> str:
    parser = ArticleTextExtractor()
    parser.feed(body_html)
    return parser.get_text()


def word_count(text: str) -> int:
    return len(text.split())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    posts = fetch_post_list()

    saved = 0
    total_words = 0
    skipped_short = 0
    skipped_error = 0

    for i, post in enumerate(posts):
        slug = post.get("slug", "")
        title = post.get("title", "(no title)")
        audience = post.get("audience", "everyone")

        # Skip paywalled posts
        if audience != "everyone":
            print(f"[{i+1}/{len(posts)}] SKIP (paywalled): {title}")
            skipped_error += 1
            continue

        if saved >= TARGET_POSTS:
            break

        print(f"[{i+1}/{len(posts)}] Fetching: {title} ({slug})")

        body_html = fetch_post_html(slug)
        if not body_html:
            print(f"  Could not extract body HTML")
            skipped_error += 1
            time.sleep(DELAY)
            continue

        text = html_to_text(body_html)
        wc = word_count(text)

        if wc < MIN_WORDS:
            print(f"  SKIP: only {wc} words (minimum {MIN_WORDS})")
            skipped_short += 1
            time.sleep(DELAY)
            continue

        # Sanitize slug for filename
        safe_slug = re.sub(r'[^\w\-]', '_', slug)
        filepath = os.path.join(OUT_DIR, f"{safe_slug}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  Saved: {wc} words -> {safe_slug}.txt")
        saved += 1
        total_words += wc
        time.sleep(DELAY)

    print("\n" + "=" * 60)
    print(f"Done. Saved {saved} posts ({total_words:,} total words)")
    print(f"Skipped: {skipped_short} too short, {skipped_error} errors/paywalled")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
