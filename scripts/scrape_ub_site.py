"""Optional UB website scraper.

Use responsibly and respect robots.txt and the site's terms of use.
"""

import re
import sys
from collections import deque
from pathlib import Path
from typing import Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.config import UB_DATA_DIR  # noqa: E402

load_dotenv()

# Configuration constants (edit as needed)
ALLOWED_DOMAINS = ["engineering.buffalo.edu"]
SEED_URLS = [
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.html",
]
MAX_PAGES = 15  # enough to grab all faculty tabs/anchors
MAX_DEPTH = 2   # stay very near the seed page
OUTPUT_DIR = Path(UB_DATA_DIR)


def is_allowed_url(url: str) -> bool:
    """Return True if the URL is in the allowed domains and uses http(s)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    return any(parsed.netloc.endswith(domain) for domain in ALLOWED_DOMAINS)


def clean_text_from_html(html: str) -> str:
    """Convert HTML to readable plain text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def url_to_filename(url: str) -> str:
    """Convert a URL into a filesystem-safe filename.

    If the URL is part of the CSE faculty directory, use a descriptive,
    consolidated filename so all sections land in a single text file.
    """
    parsed = urlparse(url)
    path = parsed.path or "index"

    # Special case: CSE faculty directory and related anchors/subpaths
    if "computer-science-engineering/people/faculty-directory" in path:
        return "ub_cse_faculty_directory.txt"

    if path.endswith("/"):
        path += "index"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)
    return f"{parsed.netloc}{slug}.txt"


def crawl():
    """Breadth-first crawl starting from SEED_URLS."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visited: Set[str] = set()
    queue = deque([(url, 0) for url in SEED_URLS])
    pages_saved = 0

    print("Starting scrape from CSE faculty directory seeds...")

    while queue and pages_saved < MAX_PAGES:
        url, depth = queue.popleft()
        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)

        if not is_allowed_url(url):
            continue

        try:
            print(f"Fetching [{depth}] {url}")
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover
            print(f"Failed to fetch {url}: {exc}")
            continue

        text = clean_text_from_html(resp.text)

        filename = url_to_filename(url)
        out_path = OUTPUT_DIR / filename
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"# {url}\n\n")
            f.write(text)
        print(f"Saved cleaned text to {out_path}")

        pages_saved += 1

        soup = BeautifulSoup(resp.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            next_url = urljoin(url, href)
            if next_url not in visited and is_allowed_url(next_url):
                queue.append((next_url, depth + 1))

    print(f"Crawling complete. Saved {pages_saved} pages to {OUTPUT_DIR}.")


if __name__ == "__main__":
    crawl()
