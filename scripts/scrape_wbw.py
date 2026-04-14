"""Scrape up to 25 blog posts from Wait But Why's archive."""

import os
import re
import time
from html.parser import HTMLParser
import requests

OUTPUT_DIR = "corpus/tim_urban"
MAX_POSTS = 25
MIN_WORDS = 500
DELAY = 1.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


class ArticleExtractor(HTMLParser):
    """Extract text from the entry-content div, skipping scripts/styles."""

    SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "figure"}

    def __init__(self):
        super().__init__()
        self.in_entry = False
        self.depth = 0  # nesting depth inside entry-content
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and "entry-content" in attrs_dict.get("class", ""):
            self.in_entry = True
            self.depth = 1
            return
        if self.in_entry:
            if tag in ("div", "section", "article", "aside", "header", "footer",
                       "nav", "main", "blockquote", "ul", "ol", "table",
                       "form", "fieldset", "details"):
                self.depth += 1
            if tag in self.SKIP_TAGS:
                self.skip_depth += 1
            # Block-level tags produce line breaks
            if tag in ("p", "br", "h1", "h2", "h3", "h4", "h5", "h6",
                       "li", "div", "blockquote", "tr", "hr"):
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.in_entry:
            if tag in self.SKIP_TAGS and self.skip_depth > 0:
                self.skip_depth -= 1
            if tag in ("div", "section", "article", "aside", "header", "footer",
                       "nav", "main", "blockquote", "ul", "ol", "table",
                       "form", "fieldset", "details"):
                self.depth -= 1
                if self.depth <= 0:
                    self.in_entry = False
            if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6",
                       "li", "blockquote", "tr"):
                self.parts.append("\n")

    def handle_data(self, data):
        if self.in_entry and self.skip_depth == 0:
            self.parts.append(data)

    def handle_entityref(self, name):
        if self.in_entry and self.skip_depth == 0:
            char = {"amp": "&", "lt": "<", "gt": ">", "quot": '"',
                    "apos": "'", "nbsp": " ", "mdash": "—", "ndash": "–",
                    "lsquo": "\u2018", "rsquo": "\u2019",
                    "ldquo": "\u201c", "rdquo": "\u201d"}.get(name, "")
            self.parts.append(char)

    def handle_charref(self, name):
        if self.in_entry and self.skip_depth == 0:
            try:
                if name.startswith("x"):
                    self.parts.append(chr(int(name[1:], 16)))
                else:
                    self.parts.append(chr(int(name)))
            except (ValueError, OverflowError):
                pass

    def get_text(self):
        raw = "".join(self.parts)
        # Collapse whitespace per line, then collapse blank lines
        lines = []
        for line in raw.split("\n"):
            stripped = " ".join(line.split())
            lines.append(stripped)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text


def get_archive_links():
    """Fetch post URLs from the WBW archive page."""
    r = requests.get("https://waitbutwhy.com/archive", headers=HEADERS, timeout=30)
    r.raise_for_status()
    # Extract unique post links (format: /YYYY/MM/slug.html)
    raw_links = re.findall(
        r'href="(https?://waitbutwhy\.com/\d{4}/\d{2}/[^"]+)"', r.text
    )
    seen = set()
    unique = []
    for url in raw_links:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def slug_from_url(url):
    """Extract a filename-safe slug from a WBW post URL."""
    # e.g. https://waitbutwhy.com/2015/12/the-tail-end.html -> the-tail-end
    path = url.rstrip("/").split("/")[-1]
    slug = path.replace(".html", "").replace(".htm", "")
    # sanitize
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)
    return slug


def scrape_post(url):
    """Download a post and return its extracted text."""
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    parser = ArticleExtractor()
    parser.feed(r.text)
    return parser.get_text()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    links = get_archive_links()
    print(f"Found {len(links)} posts in archive, will fetch up to {MAX_POSTS}")

    saved = 0
    total_words = 0
    for i, url in enumerate(links):
        if saved >= MAX_POSTS:
            break
        slug = slug_from_url(url)
        out_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")

        print(f"[{i+1}/{min(len(links), MAX_POSTS + 10)}] Fetching {slug}...", end=" ", flush=True)
        try:
            text = scrape_post(url)
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(DELAY)
            continue

        wc = len(text.split())
        if wc < MIN_WORDS:
            print(f"skipped ({wc} words < {MIN_WORDS})")
            time.sleep(DELAY)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved += 1
        total_words += wc
        print(f"saved ({wc} words)")
        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} posts, {total_words} total words.")
    # List saved files
    files = sorted(os.listdir(OUTPUT_DIR))
    for fn in files:
        path = os.path.join(OUTPUT_DIR, fn)
        wc = len(open(path).read().split())
        print(f"  {fn}: {wc} words")


if __name__ == "__main__":
    main()
