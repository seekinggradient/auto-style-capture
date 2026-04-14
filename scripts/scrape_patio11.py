"""Scrape blog posts from kalzumeus.com (patio11's blog)."""

import os
import re
import time
import requests
from html.parser import HTMLParser
from urllib.parse import urljoin

BASE_URL = "https://www.kalzumeus.com"
ARCHIVE_URL = f"{BASE_URL}/archive/"
OUTPUT_DIR = "corpus/patrick_mckenzie"
MAX_POSTS = 30
DELAY = 0.5


class ArchiveLinkParser(HTMLParser):
    """Extract post links from the archive page."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            d = dict(attrs)
            if d.get("class") == "post-title-archive":
                href = d.get("href", "")
                if href:
                    self.links.append(href)


class ArticleContentParser(HTMLParser):
    """Extract text content from the post-content div inside <article>."""

    def __init__(self):
        super().__init__()
        self.in_article = False
        self.in_post_content = False
        self.depth = 0
        self.content_parts = []
        self.current_tag = None
        self.skip_tags = {"script", "style", "nav", "header", "footer"}
        self.in_skip = 0
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)

        if tag == "article":
            self.in_article = True

        # Capture post title
        if tag == "h1" and "post-title-main" in d.get("class", ""):
            self.in_title = True

        if self.in_article and tag == "div" and "post-content" in d.get("class", ""):
            self.in_post_content = True
            self.depth = 1
            return

        if self.in_post_content:
            if tag == "div":
                self.depth += 1
            if tag in self.skip_tags:
                self.in_skip += 1
            # Add spacing for block elements
            if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "blockquote", "div", "tr"):
                self.content_parts.append("\n\n")
            self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == "h1" and self.in_title:
            self.in_title = False

        if self.in_post_content:
            if tag in self.skip_tags:
                self.in_skip = max(0, self.in_skip - 1)
            if tag == "div":
                self.depth -= 1
                if self.depth <= 0:
                    self.in_post_content = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.in_post_content and self.in_skip == 0:
            self.content_parts.append(data)

    def get_text(self):
        raw = "".join(self.content_parts)
        # Collapse whitespace within lines but keep paragraph breaks
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        text = "\n".join(cleaned)
        # Collapse multiple blank lines into max two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def slug_from_path(path):
    """Derive a filename slug from a URL path."""
    # Remove leading/trailing slashes
    path = path.strip("/")
    # Use the last segment, or the whole path joined with hyphens
    parts = [p for p in path.split("/") if p]
    if not parts:
        return "index"
    # For date-based URLs like /2012/01/23/salary-negotiation/, use the slug part
    # For non-date URLs like /essays/dropping-hashes/, join all parts
    slug = "-".join(parts)
    # Clean up
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug


def fetch_archive_links():
    """Get all post links from the archive page."""
    print("Fetching archive page...")
    resp = requests.get(ARCHIVE_URL, timeout=30)
    resp.raise_for_status()
    parser = ArchiveLinkParser()
    parser.feed(resp.text)
    print(f"Found {len(parser.links)} posts in archive")
    return parser.links


def fetch_post(url):
    """Fetch a single post and extract its text content."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    parser = ArticleContentParser()
    parser.feed(resp.text)
    title = parser.title.strip()
    text = parser.get_text()
    if title:
        text = f"{title}\n\n{text}"
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    links = fetch_archive_links()

    # Filter out podcast episodes and non-article pages, but keep up to MAX_POSTS
    # We'll skip external links (ones not starting with /)
    internal_links = [l for l in links if l.startswith("/")]

    # Skip podcast episodes to get meatier written content
    filtered = []
    for link in internal_links:
        lower = link.lower()
        if "podcast" in lower:
            continue
        filtered.append(link)
        if len(filtered) >= MAX_POSTS:
            break

    print(f"Will fetch {len(filtered)} posts (skipping podcasts)")

    saved = 0
    total_words = 0
    failures = []

    for i, path in enumerate(filtered):
        url = urljoin(BASE_URL, path)
        slug = slug_from_path(path)
        outfile = os.path.join(OUTPUT_DIR, f"{slug}.txt")

        print(f"[{i+1}/{len(filtered)}] {url}")

        try:
            text = fetch_post(url)
            if len(text.split()) < 50:
                print(f"  WARNING: Very short content ({len(text.split())} words), may not be a real article")

            with open(outfile, "w", encoding="utf-8") as f:
                f.write(text)

            wc = len(text.split())
            total_words += wc
            saved += 1
            print(f"  Saved {outfile} ({wc} words)")

        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append((url, str(e)))

        if i < len(filtered) - 1:
            time.sleep(DELAY)

    print(f"\n{'='*60}")
    print(f"Done! Saved {saved} posts, {total_words} total words")
    if failures:
        print(f"Failures ({len(failures)}):")
        for url, err in failures:
            print(f"  {url}: {err}")


if __name__ == "__main__":
    main()
