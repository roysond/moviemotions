"""The damped sum — the one piece of arithmetic the whole ranking rests on.

It has never been verified against a hand-computed number. Everything else about
retrieval is a model's opinion; THIS is our own code, and it is checkable.
"""

from backend.config import EVIDENCE_CHARS
from backend.retrieval import _collapse_to_films


def chunk(movie_id, score, text="x"):
    return {"movie_id": movie_id, "score": score, "content": text,
            "title": f"film {movie_id}", "source_field": "plot"}


def test_damped_sum_matches_a_number_computed_by_hand():
    # one film, three chunks: 0.9 + 0.6/2 + 0.3/3 = 0.9 + 0.3 + 0.1 = 1.3
    films = _collapse_to_films([chunk(1, 0.9), chunk(1, 0.6), chunk(1, 0.3)], limit=5)
    assert films[0]["score"] == 1.3


def test_only_the_top_three_chunks_count():
    # a fourth chunk must change nothing — otherwise a long film wins by being long
    three = _collapse_to_films([chunk(1, 0.9), chunk(1, 0.6), chunk(1, 0.3)], limit=5)
    four = _collapse_to_films([chunk(1, 0.9), chunk(1, 0.6), chunk(1, 0.3),
                               chunk(1, 0.9)], limit=5)
    assert three[0]["score"] == four[0]["score"]


def test_breadth_of_evidence_beats_one_lucky_chunk():
    # THE BUG THIS SCORING EXISTS TO FIX. Finding Nemo had one strong barracuda chunk
    # and beat Predator, which is about nothing else. Max-pooling would score Nemo
    # higher here (0.80 > 0.70); the damped sum must not.
    nemo = chunk(1, 0.80)
    predator = [chunk(2, 0.70), chunk(2, 0.65), chunk(2, 0.60)]
    films = _collapse_to_films([nemo] + predator, limit=5)
    assert films[0]["movie_id"] == 2, "breadth of evidence must win"
    assert films[0]["score"] > films[1]["score"]


def test_the_maximum_possible_score_is_not_one():
    # three perfect chunks: 1 + 1/2 + 1/3 = 1.8333. Anyone reading a score as a
    # percentage is misreading it, and this test is where that is written down.
    films = _collapse_to_films([chunk(1, 1.0), chunk(1, 1.0), chunk(1, 1.0)], limit=5)
    assert films[0]["score"] == 1.8333


def test_the_representative_chunk_is_still_the_best_one():
    films = _collapse_to_films([chunk(1, 0.9, "best"), chunk(1, 0.6, "second")], limit=5)
    assert films[0]["content"] == "best"
    assert films[0]["best_chunk_score"] == 0.9
    assert films[0]["supporting_chunks"] == 1


def test_evidence_budget_is_the_measured_value():
    # 320 truncated a fifth of the evidence mid-sentence. If someone lowers this,
    # they should have to change a test and explain themselves.
    assert EVIDENCE_CHARS == 640
