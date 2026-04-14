"""
Scrape blog posts from jvns.ca.
Uses requests + html.parser (no BeautifulSoup).
"""

import os
import re
import time
import requests
from html.parser import HTMLParser

OUTPUT_DIR = "corpus/julia_evans"
MAX_POSTS = 30
DELAY = 0.5
BASE_URL = "https://jvns.ca"
HEADERS = {"User-Agent": "style-corpus-scraper/1.0 (research; polite)"}


# ---------------------------------------------------------------------------
# 1. Parse the homepage to extract post URLs
# ---------------------------------------------------------------------------

class LinkExtractor(HTMLParser):
    """Pull all <a href="/blog/..."> links from the homepage."""

    def __init__(self):
        super().__init__()
        self.links = []  # list of (url, title_text)
        self._in_a = False
        self._href = None
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_d = dict(attrs)
            href = attrs_d.get("href", "")
            # Match blog post URLs like /blog/2026/03/10/slug/
            if re.match(r"^/blog/\d{4}/\d{2}/\d{2}/", href):
                self._in_a = True
                self._href = href
                self._text_parts = []

    def handle_data(self, data):
        if self._in_a:
            self._text_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._in_a:
            title = "".join(self._text_parts).strip()
            self.links.append((self._href, title))
            self._in_a = False
            self._href = None
            self._text_parts = []


# ---------------------------------------------------------------------------
# 2. Parse a single blog post page to extract the article body text
# ---------------------------------------------------------------------------

class PostContentExtractor(HTMLParser):
    """
    Extract article text from a jvns.ca blog post.

    Strategy: collect text from <p>, <li>, <h1>-<h6>, <blockquote>,
    <pre>/<code> blocks, skipping <nav>, <header>, <footer>, <script>,
    <style>, and the site navigation.
    """

    SKIP_TAGS = {"nav", "footer", "script", "style", "noscript", "svg"}
    BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "blockquote", "pre", "code", "td", "th", "figcaption", "dt", "dd"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0       # depth inside a SKIP_TAGS element
        self._paragraphs = []      # collected text blocks
        self._current_block = []   # accumulator for current block
        self._in_block = False
        self._block_depth = 0
        self._found_article = False
        self._article_depth = 0
        # Track if we are inside the post content area
        self._in_content = False
        self._content_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")

        # Detect the article/content wrapper
        # jvns.ca uses <div class="container"> or <article> etc.
        # We look for common content markers
        if tag == "article" or (tag == "div" and "post" in cls.lower()):
            self._in_content = True
            self._content_depth = 1

        if self._in_content and tag == "div":
            self._content_depth += 1

        # Track skip regions
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        # Track block elements
        if tag in self.BLOCK_TAGS:
            if not self._in_block:
                self._in_block = True
                self._block_depth = 1
                self._current_block = []
            else:
                self._block_depth += 1

        # Add line break for <br>
        if tag == "br" and self._in_block:
            self._current_block.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if self._skip_depth > 0:
            return

        if tag in self.BLOCK_TAGS and self._in_block:
            self._block_depth -= 1
            if self._block_depth <= 0:
                text = "".join(self._current_block).strip()
                if text:
                    self._paragraphs.append(text)
                self._in_block = False
                self._current_block = []

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_block:
            self._current_block.append(data)

    def get_text(self):
        return "\n\n".join(self._paragraphs)


# ---------------------------------------------------------------------------
# 3. Fallback: simple tag-stripping extractor
# ---------------------------------------------------------------------------

class SimpleTextExtractor(HTMLParser):
    """Fallback: strip all tags, collect text between markers."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self):
        raw = " ".join(self._chunks)
        # Collapse whitespace
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def extract_post_text(html: str) -> str:
    """Extract article body text from a blog post page.

    jvns.ca structure: <article class="hentry"> ... <main> CONTENT </main> ... </article>
    We first try to isolate the <main> block inside the article, then parse it.
    """
    # Try to extract just the <main>...</main> region (the post body)
    main_match = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)
    if main_match:
        region = main_match.group(1)
    else:
        # Fallback: try <article ...>...</article>
        article_match = re.search(r'<article[^>]*>(.*?)</article>', html, re.DOTALL)
        region = article_match.group(1) if article_match else html

    # Primary approach: structured extraction on the isolated region
    parser = PostContentExtractor()
    parser.feed(region)
    text = parser.get_text()

    # If structured extraction got very little, fall back to simple tag stripping
    if len(text.split()) < 50:
        parser2 = SimpleTextExtractor()
        parser2.feed(region)
        fallback_text = parser2.get_text()
        if len(fallback_text.split()) > len(text.split()):
            text = fallback_text

    return text


def slug_from_url(url: str) -> str:
    """Extract a filename-safe slug from a post URL."""
    # /blog/2026/03/10/some-slug/ -> some-slug
    parts = url.strip("/").split("/")
    # The slug is everything after the date components
    if len(parts) >= 5:
        slug = "-".join(parts[4:])  # handles multi-segment slugs
    else:
        slug = parts[-1]
    return slug or "untitled"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Fetch homepage to get post links
    print("Fetching homepage...")
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    extractor = LinkExtractor()
    extractor.feed(resp.text)

    # Deduplicate while preserving order
    seen = set()
    post_links = []
    for href, title in extractor.links:
        if href not in seen:
            seen.add(href)
            post_links.append((href, title))

    post_links = post_links[:MAX_POSTS]
    print(f"Found {len(post_links)} post links (capped at {MAX_POSTS})")

    # Step 2: Fetch each post
    saved = 0
    total_words = 0

    for i, (href, title) in enumerate(post_links):
        url = BASE_URL + href if href.startswith("/") else href
        slug = slug_from_url(href)
        filename = f"{slug}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"  [{i+1}/{len(post_links)}] {title[:60]}...", end=" ", flush=True)

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"FAILED ({e})")
            time.sleep(DELAY)
            continue

        text = extract_post_text(r.text)
        word_count = len(text.split())

        if word_count < 20:
            print(f"SKIPPED (only {word_count} words extracted)")
            time.sleep(DELAY)
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        saved += 1
        total_words += word_count
        print(f"OK ({word_count} words)")

        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} posts, {total_words} total words.")
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
