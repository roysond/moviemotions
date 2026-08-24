"""Fetch each film's Wikipedia 'Plot' section — the scene-level narrative source.

Join path: IMDb ID -> Wikidata (property P345) -> English Wikipedia article -> Plot section.
IMDb IDs are unique identifiers, so this resolves the exact film with no title guessing
(the 2010 Karate Kid, never the 1984 one). That is the "join by identifier, not title" rule.

Reads IMDb IDs from data/raw/tmdb_*.json when present (in the repo); otherwise falls back to
the built-in list. Writes data/plots.json — the RAW plot text, kept untouched; chunking and
embedding derive from it later.

Uses httpx (not urllib): httpx verifies TLS against certifi's bundle, so it works on macOS
python.org Python, which ships no system CA bundle.
"""

import glob
import html
import json
import os
import re
import time

import httpx

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
PLOT_HEADINGS = {"plot", "plot summary", "synopsis", "plot synopsis"}

# Introduce ourselves the way a real client does: a descriptive User-Agent plus the
# Accept headers a browser sends. Wikimedia's anti-scraper filter refuses bare requests.
HEADERS = {
    "User-Agent": "MovieMotions/0.1 (movie mood-recommendation learning project) python-httpx",
    "Api-User-Agent": "MovieMotions/0.1 (movie mood-recommendation learning project)",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_client = httpx.Client(headers=HEADERS, timeout=45, follow_redirects=True)

# (title, imdb_id, tmdb_id) — used only when data/raw is absent
BUILTIN = [
    ("Predator", "tt0093773", 106),
    ("Finding Nemo", "tt0266543", 12),
    ("Rocky", "tt0075148", 1366),
    ("The Dark Knight", "tt0468569", 155),
    ("The Hangover", "tt1119646", 18785),
    ("Logan", "tt3315342", 263115),
    ("Inception", "tt1375666", 27205),
    ("The Shawshank Redemption", "tt0111161", 278),
    ("Terminator 2: Judgment Day", "tt0103064", 280),
    ("Jurassic Park", "tt0107290", 329),
    ("Alien", "tt0078748", 348),
    ("The Karate Kid", "tt1155076", 38575),
    ("Get Out", "tt5052448", 419430),
    ("Mortal Kombat", "tt0293429", 460465),
    ("Crazy, Stupid, Love.", "tt1570728", 50646),
    ("Spider-Man", "tt0145487", 557),
    ("Titanic", "tt0120338", 597),
    ("Harry Potter and the Philosopher's Stone", "tt0241527", 671),
    ("Home Alone", "tt0099785", 771),
    ("Toy Story", "tt0114709", 862),
]


def films():
    raws = sorted(glob.glob("data/raw/tmdb_*.json"))
    if raws:
        rows = []
        for f in raws:
            o = json.load(open(f))
            rows.append((o.get("title"), o.get("imdb_id"), o.get("id")))
        return rows
    return BUILTIN


def _get(url, params, tries=4):
    """GET JSON, backing off on HTTP 429 (rate limit)."""
    for attempt in range(tries):
        resp = _client.get(url, params=params)
        if resp.status_code == 429 and attempt < tries - 1:
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()


def imdb_to_wiki_title(imdb_id):
    """IMDb ID -> English Wikipedia article title, via Wikidata's Action API.

    Step 1: find the Wikidata item whose P345 (IMDb ID) statement equals imdb_id.
    Step 2: read that item's English-Wikipedia sitelink (the article title).
    """
    search = _get(WIKIDATA_API, {
        "action": "query", "list": "search",
        "srsearch": f"haswbstatement:P345={imdb_id}",
        "format": "json",
    })
    hits = search["query"]["search"]
    if not hits:
        return None
    qid = hits[0]["title"]   # e.g. "Q167726"
    entity = _get(WIKIDATA_API, {
        "action": "wbgetentities", "ids": qid,
        "props": "sitelinks", "sitefilter": "enwiki",
        "format": "json",
    })
    sitelink = entity["entities"].get(qid, {}).get("sitelinks", {}).get("enwiki")
    return sitelink["title"] if sitelink else None


def plot_for(title):
    """Return the plain-text Plot section of a Wikipedia article, or None."""
    secs = _get(WIKIPEDIA, {
        "action": "parse", "page": title, "prop": "sections", "format": "json",
    })["parse"]["sections"]
    idx = next((s["index"] for s in secs if s["line"].strip().lower() in PLOT_HEADINGS), None)
    if idx is None:
        idx = next((s["index"] for s in secs if "plot" in s["line"].lower()), None)
    if idx is None:
        return None
    raw = _get(WIKIPEDIA, {
        "action": "parse", "page": title, "section": idx, "prop": "text", "format": "json",
    })["parse"]["text"]["*"]
    t = re.sub(r"<ref[^>]*?>.*?</ref>", "", raw, flags=re.S)   # drop <ref>...</ref>
    t = re.sub(r"<ref[^>]*?/>", "", t)                          # drop self-closing <ref/>
    t = re.sub(r"<[^>]+>", "", t)                               # drop all remaining tags
    t = html.unescape(t)
    t = re.sub(r"\[\d+\]", "", t)                               # drop [12] citation marks
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t or None


def main():
    out = []
    for title, imdb_id, tmdb_id in films():
        try:
            wiki_title = imdb_to_wiki_title(imdb_id)
            plot = plot_for(wiki_title) if wiki_title else None
            out.append({
                "tmdb_id": tmdb_id, "imdb_id": imdb_id, "title": title,
                "wiki_title": wiki_title, "plot": plot,
                "plot_chars": len(plot) if plot else 0,
            })
            flag = "OK  " if plot else "MISS"
            print(f"{flag} {title:34.34} -> {wiki_title}  ({len(plot) if plot else 0} chars)")
        except httpx.HTTPStatusError as error:
            body = " ".join(error.response.text.split())[:200]   # what the server actually said
            out.append({
                "tmdb_id": tmdb_id, "imdb_id": imdb_id, "title": title,
                "wiki_title": None, "plot": None, "plot_chars": 0,
                "error": f"HTTP {error.response.status_code}: {body}",
            })
            print(f"ERR  {title:34.34} -> HTTP {error.response.status_code}: {body}")
        except Exception as error:
            out.append({
                "tmdb_id": tmdb_id, "imdb_id": imdb_id, "title": title,
                "wiki_title": None, "plot": None, "plot_chars": 0, "error": str(error),
            })
            print(f"ERR  {title:34.34} -> {type(error).__name__}: {error}")
        time.sleep(1)  # be polite to Wikidata / Wikipedia

    os.makedirs("data", exist_ok=True)
    with open("data/plots.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    got = sum(1 for o in out if o["plot"])
    chars = sorted(o["plot_chars"] for o in out)
    median = chars[len(chars) // 2] if chars else 0
    print(f"\nsaved data/plots.json — {got}/{len(out)} films have a plot; median ~{median} chars")


if __name__ == "__main__":
    main()
