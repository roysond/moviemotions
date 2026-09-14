"""The contract between the reasoner and the writer.

WHY THIS FILE CAN BE TRUSTED
    Every function under test is pure: text in, data out. No key, no network, no
    database, no model. So these tests say something definite rather than something
    that happened to be true of one run — and they are the ONLY tests in the project
    that can say a hallucination was blocked, because blocking it is a code decision.

WHAT IS ACTUALLY BEING PROTECTED
    The writer is handed the output of `validate` and nothing else. Anything that
    slips through here reaches the person as a recommendation, in a confident voice,
    with no way for them to tell. So the interesting cases are all the ways a verdict
    can be wrong, not the way it can be right.
"""

from backend import handoff

# Exactly the shape backend/tools.py prints. If that renderer changes, these tests
# should fail — the contract is parsed FROM this text, and a silent format change is
# how the gate would quietly start letting everything through.
EVIDENCE = """2 films, best first.

1. Predator (1987) · 107 min · relevance 0.711 (vector had it #3)
   MATCHED mood — on its feel, rank 1.00 among text of that kind, raw 0.680
   FEELS   Tense and airless, the jungle closing in.
   PREMISE A commando team is hunted by something they cannot see.

2. Home Alone (1990) · 103 min · relevance 0.551 (vector had it #1)
   MATCHED mood — on its feel, rank 0.94 among text of that kind, raw 0.671
   FEELS   Giddy and warm, a house turned into a playground.
   PREMISE A boy left behind defends his home from two burglars.
"""


def test_films_in_evidence_reads_title_and_year():
    found = handoff.films_in_evidence(EVIDENCE)
    assert found == {("predator", 1987): "Predator",
                     ("home alone", 1990): "Home Alone"}


def test_extract_survives_a_code_fence():
    """Small models fence their JSON however firmly they are told not to."""
    payload = handoff.extract('```json\n{"films": [], "nothing_found": true}\n```')
    assert payload == {"films": [], "nothing_found": True}


def test_extract_returns_none_rather_than_guessing():
    assert handoff.extract("I could not decide.") is None
    assert handoff.extract("{not json at all") is None
    assert handoff.extract("") is None


def test_a_real_verdict_survives_intact():
    verdict, problems = handoff.validate(
        {"films": [{"title": "Predator", "year": 1987,
                    "why": ["creatures hunting people", "no safe ground"]}]},
        EVIDENCE)
    assert problems == []
    assert verdict == {"films": [{"title": "Predator", "year": 1987,
                                  "why": ["creatures hunting people",
                                          "no safe ground"]}],
                       "nothing_found": False}


def test_a_film_the_tools_never_returned_is_dropped():
    """THE GATE. Jaws is not in the catalogue and must not reach the writer."""
    verdict, problems = handoff.validate(
        {"films": [{"title": "Jaws", "year": 1975, "why": ["something in the water"]}]},
        EVIDENCE)
    assert verdict["films"] == []
    assert verdict["nothing_found"] is True
    assert any("Jaws" in problem for problem in problems)


def test_a_near_miss_is_dropped_and_never_corrected():
    """A wrong year is a different film. Correcting it to the nearest match would
    turn a visible error into an invisible one."""
    verdict, _ = handoff.validate(
        {"films": [{"title": "Predator", "year": 1990, "why": ["tense"]}]}, EVIDENCE)
    assert verdict["films"] == []


def test_the_catalogues_spelling_wins():
    verdict, _ = handoff.validate(
        {"films": [{"title": "predator", "year": 1987, "why": ["tense"]}]}, EVIDENCE)
    assert verdict["films"][0]["title"] == "Predator"


def test_a_film_with_no_reason_is_dropped():
    """A recommendation with no reason is a name, and the writer would invent one."""
    verdict, problems = handoff.validate(
        {"films": [{"title": "Predator", "year": 1987, "why": []}]}, EVIDENCE)
    assert verdict["films"] == []
    assert any("no reason" in problem for problem in problems)


def test_nothing_found_is_derived_not_believed():
    """A reasoner that says it found nothing and then lists a film has contradicted
    itself. The list is the evidence."""
    verdict, _ = handoff.validate(
        {"films": [{"title": "Predator", "year": 1987, "why": ["tense"]}],
         "nothing_found": True}, EVIDENCE)
    assert verdict["nothing_found"] is False


def test_a_missing_films_key_is_survivable():
    verdict, problems = handoff.validate({"nothing_found": True}, EVIDENCE)
    assert verdict == {"films": [], "nothing_found": True}
    assert problems


def test_reasons_are_trimmed_to_phrases():
    long_reason = "x" * (handoff.MAX_REASON_CHARS + 50)
    verdict, _ = handoff.validate(
        {"films": [{"title": "Predator", "year": 1987,
                    "why": ["one.", "two", "three", long_reason]}]}, EVIDENCE)
    why = verdict["films"][0]["why"]
    assert len(why) == handoff.MAX_REASONS
    assert why[0] == "one"                       # the full stop belongs to the writer
    assert all(len(phrase) <= handoff.MAX_REASON_CHARS for phrase in why)


def test_the_writer_sees_titles_and_reasons_and_nothing_else():
    """The guarantee is structural: what is not on this page cannot be written."""
    verdict, _ = handoff.validate(
        {"films": [{"title": "Predator", "year": 1987, "why": ["creatures hunting"]}]},
        EVIDENCE)
    page = handoff.for_writer(verdict)
    assert "Predator (1987)" in page
    assert "creatures hunting" in page
    assert "relevance" not in page
    assert "MATCHED" not in page
    assert "commando" not in page                # the premise never crosses over
