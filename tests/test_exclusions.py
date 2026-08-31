"""An excluded title must name a film that exists.

WHAT HAPPENED
    Asked for "something like a bachelor's night out with rave party, madness and fun
    adventure", the agent sent exclude_title="A Bachelor's Night Out" — reading a
    description of an EVENING as the title of a film. No such film exists, so the SQL
    excluded nothing and the search was unharmed. The answer was not: it opened with
    "excluding 'A Bachelor's Night Out'" and built five numbered points on top of a
    film that has never existed.

    The catalogue is a table. "Is this a real title" has exactly one right answer, and
    the exact machine can give it — so the fuzzy one should never have been trusted with
    it. These tests pin the rule that decides.
"""

from backend.tools import names_a_real_film

CATALOGUE = ["Terminator 2: Judgment Day", "The Hangover", "Jurassic Park",
             "Alien", "The Seventh Seal"]


def test_an_exact_title_is_real():
    assert names_a_real_film("Jurassic Park", CATALOGUE)


def test_case_and_whitespace_do_not_matter():
    assert names_a_real_film("  jurassic park ", CATALOGUE)


def test_a_shortened_title_still_counts():
    # The SQL excludes on a LIKE, so "Terminator 2" really does remove the full title.
    # This rule must agree with the SQL or the tool will report one thing and do another.
    assert names_a_real_film("Terminator 2", CATALOGUE)


def test_a_description_is_not_a_title():
    # The actual bug, in one line.
    assert not names_a_real_film("A Bachelor's Night Out", CATALOGUE)


def test_a_longer_phrase_containing_a_title_is_not_a_title():
    # "Alien" is in the catalogue; "Alien Resurrection" is not, and the SQL would
    # exclude nothing for it. Saying otherwise would be a second kind of lie.
    assert not names_a_real_film("Alien Resurrection", CATALOGUE)


def test_nothing_is_not_something():
    assert not names_a_real_film("", CATALOGUE)
    assert not names_a_real_film(None, CATALOGUE)
    assert not names_a_real_film("   ", CATALOGUE)


# ── constraint precedence ─────────────────────────────────────────────────────
# "A horror comedy under 90 minutes" that matches nothing has two possible replies.
# "No films found" is useless. "There are horror comedies, they all run longer, shall I
# show them?" is an answer. That difference is a precedence order, and this pins it.

from backend.tools import genre_key, kept_description, relaxation_steps


def test_genre_key_matches_the_graph():
    # Must agree with pipeline/build_graph.py or the EXISTS clause silently matches nothing.
    assert genre_key("Science Fiction") == "genre:science-fiction"
    assert genre_key("  horror ") == "genre:horror"


def test_length_is_surrendered_before_the_year():
    steps = relaxation_steps({"genre": "Horror", "max_runtime": 90, "after_year": 1990})
    assert [label for _, label in steps] == ["the length limit", "the year range"]


def test_relaxation_is_cumulative():
    # Step two keeps step one dropped — otherwise the second attempt re-imposes a limit
    # already proven to be blocking.
    steps = relaxation_steps({"max_runtime": 90, "before_year": 1999})
    assert steps[0][0] == ("max_runtime",)
    assert steps[1][0] == ("max_runtime", "before_year")


def test_a_constraint_never_offered_is_never_in_the_list():
    # Genre, actor and director are the request. Offering to drop them answers a
    # different question than the one asked.
    steps = relaxation_steps({"genre": "Horror", "actor": "Sam Neill", "director": "Nolan"})
    assert steps == []


def test_the_kept_constraints_are_said_back_to_the_user():
    assert kept_description({"genre": "Horror", "max_runtime": 90}) == "genre=Horror"
    assert kept_description({"max_runtime": 90}) == ""


def test_every_filter_name_reaches_a_parameter_search_really_has():
    """The wiring, not the pieces.

    genre_key() was right, relaxation_steps() was right, every unit test passed — and
    the first real call died with `search() got an unexpected keyword argument 'genre'`,
    because nothing had ever checked that the dict one function builds fits the function
    it is handed to. Two correct halves, no test of the join.
    """
    import inspect

    from backend.retrieval import search
    from backend.tools import search_kwargs

    accepted = set(inspect.signature(search).parameters)
    everything = {"genre": "Horror", "actor": "Sam Neill", "director": "Nolan",
                  "max_runtime": 90, "min_runtime": 60,
                  "after_year": 1990, "before_year": 1999,
                  "exclude_title": "Alien"}
    nothing = {key: None for key in everything}

    for active in (everything, nothing):
        unknown = set(search_kwargs(active)) - accepted
        assert not unknown, f"search() has no parameter(s) named {sorted(unknown)}"


# ── the offer wording ─────────────────────────────────────────────────────────
from backend.tools import relaxation_message


def test_a_capped_list_is_never_reported_as_a_count():
    # MAX_RESULTS stops the list at 5, so five is a floor. Printing "5 films" when the
    # real answer is "twenty" states a cap as a fact.
    full = relaxation_message("the length limit", "genre=Horror",
                              ["A", "B", "C", "D", "E"], capped=True)
    assert "at least 5" in full
    exact = relaxation_message("the length limit", "genre=Horror", ["Alien", "Get Out"],
                               capped=False)
    assert "2 film(s)" in exact and "at least" not in exact


def test_no_titles_are_named_when_nothing_was_kept():
    # "under 30 minutes" with no genre: the relaxed search is just the catalogue, and
    # naming five of it reads as a recommendation of five arbitrary films.
    message = relaxation_message("the length limit", "", ["Predator", "Inception"],
                                 capped=True)
    assert "Predator" not in message and "Inception" not in message
    assert "ASK whether to ignore it" in message


def test_no_docstring_denies_a_filter_its_tool_actually_has():
    """A tool that lists an argument and then says it has no such filter is worse than
    one that says nothing: the model reads both, and the later sentence usually wins.

    This happened the same hour genre/actor/director were added — the old boundary
    paragraph, four paragraphs below the new one, still said 'this tool has no genre,
    cast or director filter'.
    """
    import inspect

    from backend.tools import TOOLS

    for tool in TOOLS:
        text = " ".join(tool.description.lower().split())
        for name in inspect.signature(tool.func).parameters:
            for denial in (f"has no {name} filter",
                           f"no {name}, cast or director filter",
                           f"does not filter by {name}"):
                assert denial not in text, (
                    f"{tool.name} takes '{name}' but its docstring says \"{denial}\"")
