"""
Scrape actual blog posts (entries) from simonwillison.net.
Uses requests + html.parser only.
Filters out blogmarks, quotes, notes -- only gets long-form entries.
Saves each post as a .txt file in corpus/simon_willison/.
"""

import os
import re
import time
import requests
from html.parser import HTMLParser

OUTPUT_DIR = "corpus/simon_willison"
MIN_WORDS = 500
TARGET_POSTS = 25
DELAY = 0.5
HEADERS = {"User-Agent": "Mozilla/5.0 (style-research scraper; polite; one-time)"}
BASE = "https://simonwillison.net"


# ---------------------------------------------------------------------------
# 1) Collect entry URLs from yearly archive pages
#    Archive pages at /2025/, /2024/ etc list entries with direct links
# ---------------------------------------------------------------------------

class ArchiveEntryExtractor(HTMLParser):
    """Extract blog entry links from a yearly archive page.

    The archive pages have structure like:
        <div class="entry">
            <ul>
                <li>...: <a href="/2025/Jan/2/ending-a-year-long-posting-streak/">Title</a></li>
            </ul>
        </div>

    Entry links match pattern: /YYYY/Mon/DD/slug/
    """

    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self._seen: set[str] = set()
        self._in_entry_div = False
        self._entry_div_depth = 0
        self._link_re = re.compile(r"^/\d{4}/[A-Z][a-z]{2}/\d{1,2}/[\w-]+/$")

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")

        if tag == "div" and "entry" in cls.split():
            self._in_entry_div = True
            self._entry_div_depth = 1
            return

        if self._in_entry_div and tag == "div":
            self._entry_div_depth += 1

        if self._in_entry_div and tag == "a":
            href = attrs_d.get("href", "")
            if self._link_re.match(href) and href not in self._seen:
                self._seen.add(href)
                self.links.append(href)

    def handle_endtag(self, tag):
        if self._in_entry_div and tag == "div":
            self._entry_div_depth -= 1
            if self._entry_div_depth <= 0:
                self._in_entry_div = False


# ---------------------------------------------------------------------------
# 2) Extract title from a post page
# ---------------------------------------------------------------------------

class TitleExtractor(HTMLParser):
    """Extract the h2 title from an entry page."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self._in_entry = False
        self._in_h2 = False
        self._found = False

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "")
        if tag == "div" and "entryPage" in cls:
            self._in_entry = True
        if self._in_entry and tag == "h2" and not self._found:
            self._in_h2 = True

    def handle_endtag(self, tag):
        if self._in_h2 and tag == "h2":
            self._in_h2 = False
            self._found = True

    def handle_data(self, data):
        if self._in_h2:
            self.title += data


# ---------------------------------------------------------------------------
# 3) Extract text content from a single post page
# ---------------------------------------------------------------------------

class PostContentExtractor(HTMLParser):
    """Extract main text from a blog post page.

    Captures text inside <div class="entry entryPage">,
    stops at <div class="entryFooter">.
    """

    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []
        self._capturing = False
        self._entry_depth = 0
        self._stop = False
        self._skip_tags = {"script", "style", "svg", "code", "pre"}
        self._skip_depth = 0
        self._in_h2_title = False  # skip the title h2, we add it separately

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")

        if tag == "div" and "entryPage" in cls:
            self._capturing = True
            self._entry_depth = 1
            return

        if not self._capturing or self._stop:
            return

        if tag == "div" and ("entryFooter" in cls or "recent-articles" in cls):
            self._stop = True
            return

        if tag == "div":
            self._entry_depth += 1

        if tag in self._skip_tags:
            self._skip_depth += 1

        # Skip the date paragraph
        if tag == "p" and "mobile-date" in cls:
            self._skip_depth += 1

        if tag in ("p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br"):
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag == "p" and self._skip_depth > 0:
            # could be the mobile-date p closing
            self._skip_depth -= 1

        if not self._capturing or self._stop:
            return

        if tag == "div":
            self._entry_depth -= 1
            if self._entry_depth <= 0:
                self._stop = True

        if tag in ("p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"):
            self.chunks.append("\n")

    def handle_data(self, data):
        if self._capturing and not self._stop and self._skip_depth == 0:
            self.chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self.chunks)
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            line = " ".join(line.split())
            if line:
                cleaned.append(line)
        return "\n\n".join(cleaned)


def slug_from_url(url: str) -> str:
    parts = [p for p in url.strip("/").split("/") if p]
    return parts[-1] if parts else "unknown"


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def collect_entry_urls(years: list[int]) -> list[str]:
    """Collect entry URLs from yearly archive pages."""
    all_links: list[str] = []

    for year in years:
        url = f"{BASE}/{year}/"
        print(f"Fetching archive: {url}")
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        ext = ArchiveEntryExtractor()
        ext.feed(html)
        print(f"  Found {len(ext.links)} entries")
        all_links.extend(ext.links)
        time.sleep(DELAY)

    return all_links


def main():
    # Clear existing files
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect entry URLs from recent years
    entry_urls = collect_entry_urls([2026, 2025, 2024, 2023])
    print(f"\nTotal entry URLs collected: {len(entry_urls)}\n")

    # Fetch and save each post, filtering by word count
    saved = 0
    skipped_short = 0
    skipped_empty = 0
    total_words = 0

    for i, path in enumerate(entry_urls, 1):
        if saved >= TARGET_POSTS:
            print(f"\nReached target of {TARGET_POSTS} posts, stopping.")
            break

        url = BASE + path
        slug = slug_from_url(path)

        # Skip weeknotes - they're miscellaneous updates, not essays
        if "weeknotes" in slug.lower():
            print(f"[{i}] Skipping weeknotes: {slug}")
            continue

        print(f"[{i}] {url}")
        try:
            html = fetch(url)
        except Exception as e:
            print(f"  ERROR fetching: {e}")
            time.sleep(DELAY)
            continue

        # Extract title
        title_ext = TitleExtractor()
        title_ext.feed(html)
        title = title_ext.title.strip()

        # Extract body
        content_ext = PostContentExtractor()
        content_ext.feed(html)
        body = content_ext.get_text()

        if not body.strip():
            print(f"  WARNING: no content extracted, skipping")
            skipped_empty += 1
            time.sleep(DELAY)
            continue

        # Combine title + body
        text = f"{title}\n\n{body}" if title else body

        wc = len(text.split())
        if wc < MIN_WORDS:
            print(f"  Skipping: only {wc} words (min {MIN_WORDS})")
            skipped_short += 1
            time.sleep(DELAY)
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        total_words += wc
        saved += 1
        print(f"  Saved {out_path} ({wc} words)")

        time.sleep(DELAY)

    print(f"\nDone. Saved {saved} posts, {total_words} total words.")
    print(f"Skipped: {skipped_short} too short, {skipped_empty} empty.")


if __name__ == "__main__":
    main()
