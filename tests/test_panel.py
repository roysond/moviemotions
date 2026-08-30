"""Matching an answer against the catalogue — the rule that keeps the panel grounded."""

from api import films_mentioned, reasons_for

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
