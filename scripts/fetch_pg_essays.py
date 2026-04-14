"""Fetch Paul Graham's essays for use as a test corpus."""

import re
import time
from pathlib import Path

import requests
from html.parser import HTMLParser


class EssayLinkParser(HTMLParser):
    """Parse the articles page to find essay links."""
    def __init__(self):
        super().__init__()
        self.links = []
        self._in_td = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.endswith(".html") and not href.startswith("http"):
                self.links.append(href)


class EssayContentParser(HTMLParser):
    """Extract text content from an essay page."""
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip_tags = {"script", "style", "head"}
        self._skip_depth = 0
        self._in_font = False

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "font":
            self._in_font = True
        if tag == "br":
            self.text_parts.append("\n")
        if tag == "p":
            self.text_parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip_depth -= 1
        if tag == "font":
            self._in_font = False

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        # Clean up whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        return raw.strip()


def fetch_essay_links(max_essays: int = 50) -> list[str]:
    """Fetch essay links from the articles page."""
    resp = requests.get("https://paulgraham.com/articles.html", timeout=30)
    resp.raise_for_status()

    parser = EssayLinkParser()
    parser.feed(resp.text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for link in parser.links:
        if link not in seen:
            seen.add(link)
            unique.append(link)

    return unique[:max_essays]


def fetch_essay(slug: str) -> tuple[str, str]:
    """Fetch a single essay, return (title, text)."""
    url = f"https://paulgraham.com/{slug}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    parser = EssayContentParser()
    parser.feed(resp.text)
    text = parser.get_text()

    # Try to extract title (usually the first line or bold text)
    lines = text.strip().split("\n")
    title = lines[0].strip() if lines else slug.replace(".html", "")

    return title, text


def main():
    output_dir = Path(__file__).parent.parent / "corpus" / "paul_graham"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching essay list...")
    links = fetch_essay_links(max_essays=30)
    print(f"Found {len(links)} essays")

    for i, slug in enumerate(links):
        print(f"  [{i+1}/{len(links)}] Fetching {slug}...", end=" ", flush=True)
        try:
            title, text = fetch_essay(slug)
            if len(text) < 500:
                print("(too short, skipping)")
                continue

            # Save as text file
            safe_name = slug.replace(".html", "")
            path = output_dir / f"{safe_name}.txt"
            path.write_text(text, encoding="utf-8")
            print(f"OK ({len(text.split())} words)")

            time.sleep(0.5)  # be polite
        except Exception as e:
            print(f"FAILED: {e}")

    total = len(list(output_dir.glob("*.txt")))
    print(f"\nDone! Saved {total} essays to {output_dir}")


if __name__ == "__main__":
    main()
