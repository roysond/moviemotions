"""The price table and its sorting — pure data and pure functions, no network."""

from datetime import date

from backend import providers


# ── the sorting rule ────────────────────────────────────────────────────────

def offers(*specs):
    return [providers.describe(slug, name, edge) for slug, name, edge in specs]


def test_bands_never_interleave():
    mixed = offers(
        ("amazon-video", "Amazon Video", "AVAILABLE_RENT"),        # $3.99
        ("paramount-plus-essential", "Paramount+", "AVAILABLE_FLATRATE"),  # $8.99
        ("kanopy", "Kanopy", "AVAILABLE_FREE"),                    # $0
    )
    mixed.sort(key=providers.sort_key)
    assert [o["band"] for o in mixed] == ["free", "subscription", "rent"]


def test_a_cheap_rental_does_not_outrank_a_subscription():
    # THE WHOLE POINT OF BANDING. $3.99 once and $8.99 a month are different kinds
    # of cost. A plain numeric sort would put the rental first and mislead.
    both = offers(
        ("amazon-video", "Amazon Video", "AVAILABLE_RENT"),
        ("paramount-plus-essential", "Paramount+", "AVAILABLE_FLATRATE"),
    )
    both.sort(key=providers.sort_key)
    assert both[0]["band"] == "subscription"


def test_cheapest_first_inside_a_band():
    subs = offers(
        ("youtube-tv", "YouTube TV", "AVAILABLE_FLATRATE"),                # 82.99
        ("paramount-plus-essential", "Paramount+", "AVAILABLE_FLATRATE"),  # 8.99
        ("peacock-premium", "Peacock", "AVAILABLE_FLATRATE"),              # 12.99
    )
    subs.sort(key=providers.sort_key)
    assert [o["amount"] for o in subs] == [8.99, 12.99, 82.99]


def test_unknown_prices_sort_last_not_first():
    # None must never read as zero, or "we don't know" becomes "it's free".
    some = offers(
        ("paramount-amazon-channel", "Paramount+ Amazon Channel", "AVAILABLE_FLATRATE"),
        ("paramount-plus-essential", "Paramount+", "AVAILABLE_FLATRATE"),
    )
    some.sort(key=providers.sort_key)
    assert some[0]["amount"] == 8.99
    assert some[1]["amount"] is None


# ── describe(), including the cases that used to be silent ──────────────────

def test_an_unknown_provider_degrades_honestly():
    offer = providers.describe("brand-new-service", "Brand New Service",
                               "AVAILABLE_FLATRATE")
    assert offer["known"] is False
    assert offer["display"] == "Brand New Service"     # falls back to TMDB's name
    assert offer["price_text"] == "price unknown"      # never an empty string
    assert offer["verified"] is False


def test_price_text_is_never_empty():
    for slug in providers.SERVICES:
        for edge in providers.BAND_FROM_EDGE:
            offer = providers.describe(slug, slug, edge)
            assert offer["price_text"].strip(), f"{slug} / {edge} produced a blank price"


def test_a_tv_provider_service_is_its_own_band_whatever_the_edge_says():
    offer = providers.describe("fxnow", "FXNow", "AVAILABLE_FLATRATE")
    assert offer["band"] == "needs_tv_provider"


def test_rent_and_buy_prices_are_shown_as_approximate():
    rent = providers.describe("amazon-video", "Amazon Video", "AVAILABLE_RENT")
    assert rent["price_text"].startswith("from ")   # TMDB gives no per-film price


# ── the table itself ────────────────────────────────────────────────────────

def test_every_band_has_a_label_and_a_sort_position():
    for band in providers.BAND_FROM_EDGE.values():
        assert band in providers.BAND_LABEL
        assert band in providers.BANDS
    assert set(providers.BAND_LABEL) == set(providers.BANDS)


def test_no_price_is_negative_or_absurd():
    for slug, row in providers.SERVICES.items():
        for field in ("monthly", "rent", "buy"):
            value = row[field]
            if value is not None:
                assert 0 <= value < 200, f"{slug}.{field} = {value}"


def test_an_unverified_price_carries_a_note_explaining_why():
    for slug, row in providers.SERVICES.items():
        if not row["verified"]:
            assert row["note"], f"{slug} is unverified but says nothing about why"


def test_the_checked_on_date_is_not_in_the_future():
    assert providers.PRICES_CHECKED_ON <= date.today()


def test_region_and_source_agree():
    assert providers.SOURCE.endswith(providers.REGION)
