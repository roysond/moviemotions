"""Fetch a hand-picked list of films from TMDB by title and year."""

import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {os.environ['TMDB_READ_TOKEN']}"}
RAW_DIR = "data/raw"

WANTED = [
    ("Jurassic Park", 1993),
    ("Predator", 1987),
    ("Harry Potter and the Philosopher's Stone", 2001),
    ("Spider-Man", 2002),
    ("Terminator 2: Judgment Day", 1991),
    ("Crazy, Stupid, Love.", 2011),
    ("Mortal Kombat", 2021),
    ("The Karate Kid", 2010),
    ("Logan", 2017),
    ("Alien", 1979),
    ("The Shawshank Redemption", 1994),
    ("Finding Nemo", 2003),
    ("The Dark Knight", 2008),
    ("Rocky", 1976),
    ("Inception", 2010),
    ("Titanic", 1997),
    ("The Hangover", 2009),
    ("Home Alone", 1990),
    ("Toy Story", 1995),
    ("Get Out", 2017),
]


def search(client: httpx.Client, title: str, year: int) -> dict | None:
    response = client.get(
        f"{BASE}/search/movie",
        params={"query": title, "primary_release_year": year},
    )
    response.raise_for_status()
    results = response.json()["results"]
    return results[0] if results else None


def fetch_detail(client: httpx.Client, movie_id: int) -> dict:
    response = client.get(
        f"{BASE}/movie/{movie_id}",
        params={"append_to_response": "keywords,credits,watch/providers"},
    )
    response.raise_for_status()
    return response.json()


os.makedirs(RAW_DIR, exist_ok=True)
found_ids = []

with httpx.Client(headers=HEADERS, timeout=30) as client:
    for title, year in WANTED:
        hit = search(client, title, year)
        if not hit:
            print(f"NOT FOUND   {title} ({year})")
            continue
        detail = fetch_detail(client, hit["id"])
        path = f"{RAW_DIR}/tmdb_{detail['id']}.json"
        with open(path, "w") as handle:
            json.dump(detail, handle)
        found_ids.append(detail["id"])
        print(f"{detail['id']:>7}  {detail['release_date']}  {detail['title']}")
        time.sleep(0.15)

with open("data/test_corpus_ids.json", "w") as handle:
    json.dump(found_ids, handle)

print(f"\n{len(found_ids)}/20 fetched → ids saved to data/test_corpus_ids.json")