"""
providers.py — the semantic layer for streaming offers.

WHAT A SEMANTIC LAYER IS, USING THIS FILE AS THE EXAMPLE
    TMDB tells us a film is on "Paramount Plus Essential", "Paramount Plus
    Premium", "Paramount+ Amazon Channel" and "Paramount+ Roku Premium Channel".
    Those are four rows in the database and one thing in a person's head.
    TMDB also says "Apple TV" and "Apple TV Store", which look almost identical
    and are completely different: one is a monthly subscription, the other is a
    shop where you rent a film.

    Nobody's API can settle that for us, because it is a question about OUR
    product, not about their data. Deciding what a name means, and what it costs,
    is our job. This file is that decision, written down in one place.

    The database keeps what TMDB said, unedited. This file translates it at the
    moment of display. Storage stays faithful; presentation gets to be sensible.

THREE RULES THIS FILE OBEYS
    1. Every price carries the date it was checked and the page it came from.
       Streaming prices move constantly -- Apple TV went from $12.99 to $14.99 on
       the very day this file was written. A price with no date is a rumour.
    2. A price we could not verify is marked `verified=False`. It is never
       guessed at and never quietly rounded to something plausible.
    3. Rent and buy prices are TYPICAL prices for an older catalogue title, not
       this film's price. TMDB does not publish per-film prices, so the honest
       word is "from", never an exact figure.

RUN IT
    python -m backend.providers     prints the table, then checks that every provider in
                            the graph actually has an entry here
"""

import os
from datetime import date

# ---------------------------------------------------------------- the bands
#
# The band comes from the EDGE TYPE in the graph, never from the provider --
# Plex is free to watch on and also rents films, so the same provider sits in
# two bands depending on the offer. Order here is the order on screen.

BANDS = ["free", "subscription", "rent", "buy", "needs_tv_provider"]

BAND_FROM_EDGE = {
    "AVAILABLE_FREE":     "free",           # free, e.g. with a library card
    "AVAILABLE_ADS":      "free",           # free, advertising supported
    "AVAILABLE_FLATRATE": "subscription",   # included in a monthly subscription
    "AVAILABLE_RENT":     "rent",
    "AVAILABLE_BUY":      "buy",
}

BAND_LABEL = {
    "free":              "Free",
    "subscription":      "Included in a subscription",
    "rent":              "Rent",
    "buy":               "Buy",
    "needs_tv_provider": "Needs a TV provider login",
}

# WHICH COUNTRY "AVAILABLE" MEANS
# Defined here, in the semantic layer, and imported by build_graph.py -- because a
# constant written down in two files is a constant that will eventually disagree
# with itself. `SOURCE` is what the availability edges record as their origin:
# "TMDB's US listing" is a different claim from "TMDB's UK listing".
REGION = "US"
SOURCE = f"tmdb:{REGION}"

# Everything below was checked on this date. The self-test warns when it is old.
PRICES_CHECKED_ON = date(2026, 8, 28)
STALE_AFTER_DAYS = 30

# Short names for the sources, so the table below stays readable.
SRC = {
    "netflix":   "https://help.netflix.com/en/node/24926",
    "disney":    "https://www.disneyplus.com/welcome/disney-hulu-espn-bundle",
    "peacock":   "https://www.peacocktv.com/help/article/price-increase",
    "starz":     "https://www.starz.com/us/en/buy",
    "philo":     "https://www.philo.com/",
    "youtubetv": "https://tv.youtube.com/welcome/",
    "plex":      "https://www.plex.tv/plans/",
    "kanopy":    "https://help.kanopy.com/en-us/4260.htm",
    "tomsguide": "https://www.tomsguide.com/entertainment/streaming/what-streaming-costs-in-2026-the-price-of-netflix-disney-plus-max-and-more",
    "appletv":   "https://techcrunch.com/2026/08/28/apple-tv-is-raising-its-subscription-prices-again/",
    "justwatch": "https://www.justwatch.com/us/movie/the-matrix",
    "tubi":      "https://tubitv.com/help-center/About-Tubi/articles/4409953376411",
}


def s(display, monthly=None, rent=None, buy=None, note="", verified=True,
      source="", resold_from=None):
    """One row of the table. Keeping this a function rather than raw dicts means
    a typo in a key name fails here, loudly, instead of silently reading as None."""
    return {"display": display, "monthly": monthly, "rent": rent, "buy": buy,
            "note": note, "verified": verified, "source": source,
            "resold_from": resold_from}


# ---------------------------------------------------------------- the table
#
# Keys are exactly what build_graph.slug() produces from TMDB's provider_name,
# so a graph node `provider:netflix` looks up as SERVICES["netflix"].
# US prices, US dollars.

SERVICES = {

    # ---- free -------------------------------------------------------------
    "kanopy":       s("Kanopy",       monthly=0.0, note="free with a library card",
                      source=SRC["kanopy"]),
    "hoopla":       s("Hoopla",       monthly=0.0, note="free with a library card",
                      source="https://sfpl.libanswers.com/faq/129915"),
    "youtube-free": s("YouTube Free", monthly=0.0, note="free, with ads",
                      source=SRC["tubi"]),
    "fandango-at-home-free": s("Fandango at Home Free", monthly=0.0,
                      note="free, with ads",
                      source="https://thestreamable.com/everything-to-know-about-free-streaming-service-fandango-at-home"),
    "plex":         s("Plex",         monthly=0.0, note="free tier, with ads; rentals priced separately",
                      source=SRC["plex"]),

    # ---- subscriptions ----------------------------------------------------
    # Where a service has tiers, the CHEAPEST way in is listed, and the note
    # says what you give up for it.
    "netflix":      s("Netflix", monthly=19.99, note="Standard, ad-free",
                      source=SRC["netflix"]),
    "netflix-standard-with-ads": s("Netflix (with ads)", monthly=8.99,
                      note="Standard with Ads", source=SRC["netflix"]),
    "hbo-max":      s("HBO Max", monthly=10.99, note="Basic with Ads",
                      source=SRC["tomsguide"]),
    "hulu":         s("Hulu", monthly=11.99, note="with ads", verified=False,
                      source="https://www.yardbarker.com/entertainment/streaming/articles/hulu_review_plans_pricing_channels_bundles_and_more/s1_17261_42225127"),
    "disney-plus":  s("Disney+", monthly=11.99, note="with ads",
                      source=SRC["disney"]),
    "peacock-premium":      s("Peacock Premium", monthly=12.99,
                      note="with ads; raised 18 Aug 2026", source=SRC["peacock"]),
    "peacock-premium-plus": s("Peacock Premium Plus", monthly=19.99,
                      note="still has ads on live channels", source=SRC["peacock"]),
    "paramount-plus-essential": s("Paramount+ Essential", monthly=8.99,
                      note="with ads", source=SRC["tomsguide"]),
    "paramount-plus-premium":   s("Paramount+ Premium", monthly=13.99,
                      note="includes Showtime; ads remain on live CBS",
                      source=SRC["tomsguide"]),
    "apple-tv":     s("Apple TV", monthly=14.99,
                      note="raised from $12.99 on 28 Aug 2026", source=SRC["appletv"]),
    "mgm-plus":     s("MGM+", monthly=6.99, verified=False,
                      note="not confirmed on MGM+'s own page",
                      source="https://www.lowermysubs.com/blog/mgm-plus-streaming-subscription-worth-it-2026"),
    "philo":        s("Philo", monthly=25.00, note="live TV, ads unavoidable",
                      source=SRC["philo"]),
    "fubotv":       s("fuboTV", monthly=73.99, verified=False,
                      note="sources disagree; cheapest full English plan",
                      source="https://www.cabletv.com/fubotv"),
    "youtube-tv":   s("YouTube TV", monthly=82.99, note="live TV",
                      source=SRC["youtubetv"]),
    "flixfling":    s("FlixFling", verified=False, note="price not found",
                      source=""),

    # ---- the same subscription, sold through somebody else's checkout ------
    # Same content, different till. The price is set by the reseller and is not
    # published in a form we can verify, so it is left blank on purpose.
    "paramount-amazon-channel":       s("Paramount+", resold_from="Amazon",
                      verified=False, note="resold; price set by Amazon"),
    "paramount-roku-premium-channel": s("Paramount+", resold_from="Roku",
                      verified=False, note="resold; price set by Roku"),
    "hbo-max-amazon-channel":         s("HBO Max", resold_from="Amazon",
                      verified=False, note="resold; price set by Amazon"),
    "apple-tv-amazon-channel":        s("Apple TV", resold_from="Amazon",
                      verified=False, note="resold; price set by Amazon"),
    "mgm-amazon-channel":             s("MGM+", resold_from="Amazon",
                      verified=False, note="resold; price set by Amazon"),
    "mgm-plus-roku-premium-channel":  s("MGM+", resold_from="Roku",
                      verified=False, note="resold; price set by Roku"),
    "starz-apple-tv-channel":         s("Starz", monthly=11.99, resold_from="Apple TV",
                      note="Starz list price; Apple may differ", source=SRC["starz"]),

    # ---- shops: rent and buy ----------------------------------------------
    # TYPICAL prices for an older catalogue title. Not this film's price --
    # TMDB does not publish per-film prices, so these are shown as "from".
    "amazon-video":     s("Amazon Video", rent=3.99, buy=7.99,
                      note="typical for a catalogue title", source=SRC["justwatch"]),
    "apple-tv-store":   s("Apple TV Store", rent=4.99, buy=14.99,
                      note="typical for a catalogue title", source=SRC["justwatch"]),
    "fandango-at-home": s("Fandango At Home", rent=4.99, buy=14.99,
                      note="typical for a catalogue title", source=SRC["justwatch"]),
    "google-play-movies": s("Google Play Movies", verified=False,
                      note="store prices could not be verified", source=""),
    "youtube":          s("YouTube", verified=False,
                      note="store prices could not be verified", source=""),

    # ---- needs an existing TV subscription --------------------------------
    "fxnow":            s("FXNow", note="sign in with your TV provider"),
    "spectrum-on-demand": s("Spectrum On Demand", note="sign in with your TV provider"),
}

# Providers that are not really a price at all -- you need a cable or satellite
# account you already pay for elsewhere. Their own band.
TV_PROVIDER_ONLY = {"fxnow", "spectrum-on-demand"}

UNKNOWN = s("", verified=False, note="no entry in providers.py")


# ---------------------------------------------------------------- lookup

def describe(provider_slug, provider_name, edge_type):
    """Turn one graph edge into one line for a person to read.

    Returns a dict the caller can render or sort. `price_text` is always a
    string a human can read, including when we do not know the price -- an
    empty cell in a UI reads as "free", which would be a lie.
    """
    known = SERVICES.get(provider_slug)
    row = known or UNKNOWN

    if provider_slug in TV_PROVIDER_ONLY:
        band = "needs_tv_provider"
    else:
        band = BAND_FROM_EDGE.get(edge_type, "subscription")

    amount, price_text = None, "price unknown"

    if band == "needs_tv_provider":
        price_text = "with your TV provider"
    elif band == "free":
        amount, price_text = 0.0, "$0"
    elif band == "subscription" and row["monthly"] is not None:
        amount = row["monthly"]
        price_text = "$0" if amount == 0 else f"${amount:.2f}/mo"
    elif band == "rent" and row["rent"] is not None:
        amount = row["rent"]
        price_text = f"from ${amount:.2f}"
    elif band == "buy" and row["buy"] is not None:
        amount = row["buy"]
        price_text = f"from ${amount:.2f}"

    return {
        "slug":        provider_slug,
        "display":     row["display"] or provider_name,   # fall back to TMDB's name
        "band":        band,
        "amount":      amount,                # None means "we do not know"
        "price_text":  price_text,
        "note":        row["note"],
        "verified":    row["verified"] and amount is not None,
        "source":      row["source"],
        "resold_from": row["resold_from"],
        "known":       known is not None,
    }


def sort_key(offer):
    """Band first, then cheapest inside the band, with unknown prices last.

    Sorting across bands by dollar value would put a $3.99 one-off rental above
    a $8.99 subscription, which compares two different kinds of cost. Banding
    first is the whole point.
    """
    return (BANDS.index(offer["band"]),
            0 if offer["amount"] is not None else 1,
            offer["amount"] if offer["amount"] is not None else 0.0,
            offer["display"])


def staleness_days():
    return (date.today() - PRICES_CHECKED_ON).days


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    print(f"providers.py — {len(SERVICES)} services priced, "
          f"checked {PRICES_CHECKED_ON.isoformat()} "
          f"({staleness_days()} days ago)")
    if staleness_days() > STALE_AFTER_DAYS:
        print(f"  ** STALE ** prices are more than {STALE_AFTER_DAYS} days old — re-check them")

    unverified = [k for k, v in SERVICES.items() if not v["verified"]]
    print(f"  {len(SERVICES) - len(unverified)} verified · {len(unverified)} unverified: "
          + ", ".join(sorted(unverified)))

    print("\nthe table")
    for slug in sorted(SERVICES):
        r = SERVICES[slug]
        price = ("free" if r["monthly"] == 0 else
                 f"${r['monthly']:.2f}/mo" if r["monthly"] else
                 f"rent ${r['rent']:.2f} · buy ${r['buy']:.2f}" if r["rent"] else
                 "—")
        flag = " " if r["verified"] else "?"
        via = f"  via {r['resold_from']}" if r["resold_from"] else ""
        print(f" {flag} {slug:<34} {r['display']:<22} {price:<26}{via}")

    # ---- coverage: does the graph contain a provider this file does not know?
    # This is how a missing price becomes visible instead of becoming a blank
    # cell on the screen.
    try:
        import psycopg
        from dotenv import load_dotenv
        load_dotenv()
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            rows = conn.execute(
                "SELECT node_key, name FROM graph_nodes "
                "WHERE node_type = 'provider' ORDER BY name").fetchall()
        missing = [(k, n) for k, n in rows if k.split(":", 1)[1] not in SERVICES]
        extra = sorted(set(SERVICES) - {k.split(":", 1)[1] for k, _ in rows})
        print(f"\ncoverage — {len(rows)} providers in the graph")
        if missing:
            print("  ** MISSING FROM providers.py **")
            for k, n in missing:
                print(f"      {k:<40} {n}")
        else:
            print("  every provider in the graph has an entry here")
        if extra:
            print(f"  {len(extra)} entries here are not currently used by any film: "
                  + ", ".join(extra))
    except Exception as error:
        print(f"\ncoverage check skipped — {type(error).__name__}: {error}")
