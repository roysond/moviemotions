"""Matching an answer against the catalogue — the rule that keeps the panel grounded."""

from backend.api import films_mentioned, reasons_for

CATALOGUE = ["Predator", "Alien", "Terminator 2: Judgment Day", "Toy Story",
             "The Dark Knight", "Inception"]


def test_only_films_the_answer_actually_names_appear():
    answer = "I would go with Predator tonight, or Alien if you want slower dread."
    assert films_mentioned(answer, CATALOGUE) == ["Predator", "Alien"]


def test_a_film_never_mentioned_never_appears():
    # The panel is under the same grounding rule as the agent: it may not show a
    # film that was not named.
    assert "Inception" not in films_mentioned("Watch Predator.", CATALOGUE)


def test_the_longest_matching_title_wins():
    # "Terminator 2" is a substring of "Terminator 2: Judgment Day". Without the
    # longest-first rule the short name matches and the real title never gets a turn.
    found = films_mentioned("Try Terminator 2: Judgment Day.", CATALOGUE)
    assert found == ["Terminator 2: Judgment Day"]


def test_films_come_back_in_the_order_the_answer_names_them():
    answer = "Alien first, then Predator, then Toy Story to recover."
    assert films_mentioned(answer, CATALOGUE) == ["Alien", "Predator", "Toy Story"]


def test_matching_ignores_case():
    assert films_mentioned("predator is the one", CATALOGUE) == ["Predator"]


def test_reasons_are_the_agents_own_words():
    answer = "- **Predator (1987)**: commandos hunted through a jungle\n- Alien: slower"
    reasons = reasons_for(answer, "Predator")
    assert reasons == ["Predator (1987): commandos hunted through a jungle"]


def test_reasons_are_capped_so_one_film_cannot_flood_the_panel():
    answer = "\n".join(f"Predator line {n}" for n in range(10))
    assert len(reasons_for(answer, "Predator")) == 3


# ── the bugs the browser found on 30 Aug, now guarded ───────────────────────

REAL_ANSWER = """Based on the results from the search, here are some films that fit the
description of being tense and frightening, where people are hunted by dangerous
creatures, but are not "Jurassic Park":

1. **Predator (1987)**
   - A team of elite commandos on a secret mission in a Central American jungle
     come to find themselves hunted by an extraterrestrial warrior.

2. **Alien (1979)**
   - Tense and foreboding. Moods: suspense, horror, danger.
"""


def test_a_film_named_only_to_be_ruled_out_is_not_recommended():
    # The panel proudly showed Jurassic Park as pick #1 of an answer whose first
    # sentence was "...but are not Jurassic Park".
    found = films_mentioned(REAL_ANSWER, CATALOGUE)
    assert "Jurassic Park" not in found
    assert found == ["Predator", "Alien"]


def test_negation_only_applies_when_it_is_actually_nearby():
    answer = "Predator is not for everyone. Alien is the safer pick."
    assert films_mentioned(answer, CATALOGUE) == ["Predator", "Alien"]


def test_a_title_the_agent_excluded_is_dropped_even_without_a_negation():
    # Belt and braces: the tool call said exclude_title="Jurassic Park", so the
    # panel refuses it regardless of how the sentence is worded.
    answer = "Jurassic Park is great, and so is Predator."
    found = films_mentioned(answer, CATALOGUE, exclude=["Jurassic Park"])
    assert found == ["Predator"]


def test_the_reason_comes_from_the_line_below_the_title():
    # Predator's row showed no reason at all: the description sits on the NEXT
    # line and does not contain the word "Predator".
    reasons = reasons_for(REAL_ANSWER, "Predator")
    assert reasons, "Predator had no reason at all"
    assert "elite commandos" in reasons[0]


def test_the_heading_is_not_repeated_as_a_reason():
    # The panel already prints the title beside the poster.
    for reason in reasons_for(REAL_ANSWER, "Alien"):
        assert reason.lower() not in ("alien", "alien (1979)")


def test_the_preamble_is_not_mistaken_for_a_reason():
    # "Based on the results from the search..." was being shown as Jurassic Park's
    # reason, because that sentence happens to contain the title.
    for reason in reasons_for(REAL_ANSWER, "Predator"):
        assert not reason.startswith("Based on the results")
