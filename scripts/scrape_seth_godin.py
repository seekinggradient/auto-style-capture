"""Scrape up to 50 blog posts from seths.blog and save as .txt files."""

import os
import re
import time
import html
import html.parser
import requests

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus", "seth_godin")
BASE_URL = "https://seths.blog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; style-corpus-bot/1.0)"}
MAX_POSTS = 50
DELAY = 0.5


class TextExtractor(html.parser.HTMLParser):
    """Extract visible text from HTML, stripping tags."""

    def __init__(self):
        super().__init__()
        self.pieces = []
        self._skip = False
        self._skip_tags = {"script", "style", "svg", "noscript"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "blockquote", "li"):
            self.pieces.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        if tag == "p":
            self.pieces.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.pieces.append(data)

    def get_text(self):
        raw = "".join(self.pieces)
        # Collapse multiple blank lines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def extract_text(html_fragment: str) -> str:
    """Convert an HTML fragment to clean plain text."""
    # Decode HTML entities first
    decoded = html.unescape(html_fragment)
    parser = TextExtractor()
    parser.feed(decoded)
    return parser.get_text()


def get_post_links(page_html: str) -> list[tuple[str, str]]:
    """Return list of (url, title) from a listing page."""
    # Pattern: <h2><a href="URL">Title</a></h2> inside post divs
    return re.findall(r'<h2><a href="([^"]+)">(.+?)</a></h2>', page_html)


def get_post_content(post_html: str) -> str:
    """Extract the main post body text from a single post page."""
    # Content sits between <div class="post single"> or <div class="post">
    # and <div class="post-footer">
    m_start = re.search(r'<div class="post[^"]*">\s*<h2>', post_html)
    # End before byline/date or post-footer, whichever comes first
    m_end = re.search(r'<p class="byline">|<div class="sharedaddy|<div class="post-footer">', post_html)
    if not m_start or not m_end:
        return ""
    # Get content after the closing </h2> tag (skip the title)
    fragment = post_html[m_start.start():m_end.start()]
    # Remove the <h2>...</h2> title line
    fragment = re.sub(r"<h2>.*?</h2>", "", fragment, count=1, flags=re.DOTALL)
    return extract_text(fragment)


def slugify(title: str) -> str:
    """Turn a post title into a filename-safe slug."""
    text = html.unescape(title).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text).strip("_")
    return text[:80] or "untitled"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: collect post URLs from listing pages
    post_links: list[tuple[str, str]] = []
    page = 1
    while len(post_links) < MAX_POSTS:
        url = BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"
        print(f"Fetching listing page {page}: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  Got status {resp.status_code}, stopping.")
            break
        links = get_post_links(resp.text)
        if not links:
            print("  No posts found, stopping.")
            break
        post_links.extend(links)
        page += 1
        time.sleep(DELAY)

    post_links = post_links[:MAX_POSTS]
    print(f"\nCollected {len(post_links)} post URLs. Fetching content...\n")

    # Phase 2: fetch each post and save
    saved = 0
    total_words = 0
    for i, (post_url, title) in enumerate(post_links, 1):
        clean_title = html.unescape(re.sub(r"<[^>]+>", "", title))
        print(f"  [{i}/{len(post_links)}] {clean_title}")
        try:
            resp = requests.get(post_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"    Skipped (status {resp.status_code})")
                continue
            body = get_post_content(resp.text)
            if not body:
                print("    Skipped (no content extracted)")
                continue

            slug = slugify(title)
            filename = f"{slug}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Prepend the title
            full_text = f"{clean_title}\n\n{body}\n"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_text)

            wc = len(full_text.split())
            total_words += wc
            saved += 1
            print(f"    Saved ({wc} words)")
        except requests.RequestException as e:
            print(f"    Error: {e}")
        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} posts, {total_words} total words.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
