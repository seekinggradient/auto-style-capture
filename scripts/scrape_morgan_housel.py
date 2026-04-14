"""
Scraper for Morgan Housel's blog posts on collabfund.com/blog/
Uses requests + html.parser (no BeautifulSoup).
"""

import os
import re
import time
import requests
from html.parser import HTMLParser

OUTPUT_DIR = "corpus/morgan_housel"
BASE_URL = "https://collabfund.com"
AUTHOR_URL = f"{BASE_URL}/blog/authors/morgan/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
MAX_POSTS = 30
MIN_WORDS = 500
DELAY = 0.5


class TextExtractor(HTMLParser):
    """Extract text from HTML, skipping script/style tags."""

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        # Add newline after block elements
        if tag in ("p", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div"):
            self.result.append("\n\n")

    def handle_data(self, data):
        if not self._skip:
            self.result.append(data)

    def get_text(self):
        text = "".join(self.result)
        # Collapse whitespace within lines, preserve paragraph breaks
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = " ".join(line.split())
            cleaned.append(stripped)
        text = "\n".join(cleaned)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def fetch(url):
    """Fetch a URL and return response text."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def get_morgan_post_urls():
    """Get post URLs from Morgan Housel's author page."""
    html = fetch(AUTHOR_URL)
    # Extract unique blog post links (excluding /blog/authors/ and /blog/ itself and pagination)
    raw_links = re.findall(r'href="(/blog/[^/\"]+/)"', html)
    seen = set()
    urls = []
    for link in raw_links:
        if link not in seen and not link.startswith("/blog/authors/"):
            seen.add(link)
            urls.append(BASE_URL + link)
    return urls


def extract_article_content(html):
    """Extract the main article text from a blog post page."""
    # Get content between <article> and the newsletter section
    article_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
    if not article_match:
        return ""

    article_html = article_match.group(1)

    # Find where the actual prose starts (first <p> tag after header area)
    first_p = article_html.find("<p>")
    if first_p < 0:
        first_p = article_html.find("<p ")
    if first_p > 0:
        article_html = article_html[first_p:]

    # Cut off at newsletter section if present
    newsletter_idx = article_html.find('<section class="newsletter')
    if newsletter_idx > 0:
        article_html = article_html[:newsletter_idx]

    # Also cut off at "related posts" or share sections that appear after content
    for cutoff in ['class="post__share', 'class="post__related', 'class="post__tags']:
        idx = article_html.find(cutoff)
        if idx > 0:
            tag_start = article_html.rfind("<", 0, idx)
            if tag_start > 0:
                article_html = article_html[:tag_start]

    extractor = TextExtractor()
    extractor.feed(article_html)
    return extractor.get_text()


def slug_from_url(url):
    """Extract slug from URL like https://collabfund.com/blog/some-post/"""
    parts = url.rstrip("/").split("/")
    return parts[-1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching Morgan Housel's post list...")
    post_urls = get_morgan_post_urls()
    print(f"Found {len(post_urls)} posts on author page")

    # Limit to MAX_POSTS
    post_urls = post_urls[:MAX_POSTS]

    saved = 0
    total_words = 0
    skipped_short = 0

    for i, url in enumerate(post_urls):
        slug = slug_from_url(url)
        print(f"[{i+1}/{len(post_urls)}] Fetching: {slug} ...", end=" ", flush=True)

        try:
            html = fetch(url)
            text = extract_article_content(html)
            word_count = len(text.split())

            if word_count < MIN_WORDS:
                print(f"SKIP ({word_count} words < {MIN_WORDS})")
                skipped_short += 1
            else:
                filepath = os.path.join(OUTPUT_DIR, f"{slug}.txt")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                saved += 1
                total_words += word_count
                print(f"OK ({word_count} words)")

        except Exception as e:
            print(f"ERROR: {e}")

        if i < len(post_urls) - 1:
            time.sleep(DELAY)

    print(f"\n--- Done ---")
    print(f"Posts saved: {saved}")
    print(f"Posts skipped (< {MIN_WORDS} words): {skipped_short}")
    print(f"Total word count: {total_words:,}")


if __name__ == "__main__":
    main()
