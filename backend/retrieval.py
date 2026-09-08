"""Find films. Hard facts build a wall; meaning ranks whatever is left standing inside it.

THE TWO HALVES, AND WHY THEY ARE NOT THE SAME MACHINERY
    A runtime, a year, a title — these have a yes/no answer, so they belong in a WHERE
    clause where the database can be trusted to enforce them exactly. A mood has no
    yes/no answer, so it belongs in a vector where closeness is the whole point.
    Vectors capture topic, not truth value: "under two hours" embeds as a mood, and
    "not a cartoon" embeds next to "a cartoon". Never let a vector decide a fact.

    The wall is built FIRST. Ranking a film only to throw it away afterwards wastes the
    ranking; worse, it lets a filtered film occupy a place in the top N.

SCORING — A FILM IS AS GOOD AS ITS WEAKEST SATISFIED CONSTRAINT
    Someone who asks for "warm and funny" wants both. Averaging lets a film that is
    gloriously warm and not remotely funny beat one that is decently both, because a
    high score on one side pays for a failure on the other. Taking the MINIMUM refuses
    that trade: to score 0.6 a film must be at least 0.6 on everything asked for.

    A constraint the person did NOT state is not scored at all. Absent is not zero —
    zero is a judgement, and nobody made one.

WHAT COMES BACK, AND WHAT NEVER DOES
    Every result carries the text it matched on, because a tool that returns no evidence
    leaves a gap and a model fills gaps from its training. But `theme` is a MATCHING
    surface, never a display one: for a film whose meaning is its twist, an honest theme
    sentence gives the ending away. It scores. It is never shown. That rule lives in
    DISPLAYABLE below rather than in an instruction someone has to remember.

RUN
    python -m backend.retrieval        a few real searches, printed
"""

import bisect

import psycopg

from backend.config import DATABASE_URL
from backend.models import embed, rerank
from backend.tracing import traceable
from backend.vectors import EMBED_VARIANT, as_literal

# Which stored text answers which kind of question.
# A constraint may read MORE THAN ONE kind of stored text, and the best row wins
# whichever kind it came from.
#
#   situation reads the premise AND the plot scenes. The premise is the first act; a
#   scene is any moment in the film. "A group trapped with no way out" is usually not
#   in the setup, so premise alone can only find films that BEGIN that way.
#
# The two are not interchangeable and the difference is spoilers: a premise is safe to
# print, a scene is not. Which kind actually matched is recorded per film, and only a
# DISPLAYABLE one has its text returned.
CONSTRAINT_KIND = {
    "mood": ("mood_feel",),                  # how it feels to watch
    "theme": ("theme",),                     # what it is about underneath
    "situation": ("premise", "plot_scene"),  # who, where, what happens
}

# What a result may SHOW a person.
#   premise     written first-act-only. The display surface.
#   mood_feel   written with no plot and no destination. Safe to print.
#   theme       MATCHING ONLY — for the films that need it most, an honest theme IS the
#               twist. Absent from this set on purpose.
#   plot_scene  MATCHING ONLY — a scene from anywhere in the film, including the end.
#               It is the sharpest thing to search and the least safe to print.
DISPLAYABLE = {"premise", "mood_feel"}

# Constraints that may be inherited from a "similar to X" film.
#
# MOOD ONLY, and both exclusions were measured on 7 Sep 2026 rather than assumed.
#
#   premise is out because a premise is a SETUP: seeding from it finds films with the
#   same situation rather than the same feeling, which is a different question.
#
#   theme is out because it does not discriminate. Theme sentences are abstract moral
#   statements, and abstract moral statements all look alike to an embedder: Toy Story's
#   nearest theme in the catalogue is TITANIC at 0.827, because both are built as "not
#   status, but X". Mood sentences are sensory and concrete, so they separate.
#   Measured: mood alone ranks Finding Nemo and Home Alone top for Toy Story — the two
#   answers a person would give. Adding theme replaced them with Mortal Kombat.
#
# AND THE PART THAT GENERALISES: theme's scores are systematically lower than mood's, so
# the MINIMUM of the two was ALWAYS the theme score — the combination never combined
# anything, it just handed the decision to the weaker constraint. min() compares raw
# numbers and therefore assumes the numbers are on one scale. Two kinds of text are not.
# The general fix is to rank within each constraint before taking the minimum; that is
# the same lesson as "fuse on rank, never on score thresholds", one level up.
SEEDABLE = ("mood",)

# WHY THE GRAPH IS IN THE ***WHERE*** CLAUSE
#
# "Anything by Nolan" has a yes/no answer. It has no degrees, so it must never touch a
# vector — a vector would return films that FEEL like Nolan films, which is a different
# question answered confidently and wrongly. Facts filter; meaning ranks whatever
# survives. The graph produces a SET, and a set belongs in a WHERE clause.
#
# Each gate is written so an absent constraint costs nothing: a NULL array short-
# circuits before the EXISTS is ever evaluated.
GRAPH_GATE = """
  AND (%(KEYS)s::text[] IS NULL OR EXISTS (
        SELECT 1 FROM graph_edges g
        WHERE g.from_key  = 'film:' || m.tmdb_id
          AND g.edge_type = 'EDGE'
          AND g.to_key    = ANY(%(KEYS)s::text[])))"""

CANDIDATES = """
SELECT m.movie_id, m.title,
       EXTRACT(YEAR FROM m.release_date)::int AS year,
       m.runtime_minutes,
       m.tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies m
-- The ::int casts are not decoration. A bare NULL parameter has no type Postgres
-- can infer from `$1 IS NULL`, so it refuses the query rather than guessing. The cast
-- is how you say "this is an integer that happens to be absent".
WHERE (%(max_runtime)s::int IS NULL OR m.runtime_minutes <= %(max_runtime)s::int)
  AND (%(min_year)s::int   IS NULL OR m.release_date >= make_date(%(min_year)s::int, 1, 1))
  AND (%(max_year)s::int   IS NULL OR m.release_date <= make_date(%(max_year)s::int, 12, 31))
"""

# The three graph gates, stamped out of one template so the shape cannot drift between
# them. Written as a loop rather than three copies for the same reason REGION is defined
# once: a rule spelled out three times eventually becomes three different rules.
for _keys, _edge in (("director_keys", "DIRECTED"),
                     ("actor_keys", "ACTED_IN"),
                     ("genre_keys", "HAS_GENRE")):
    CANDIDATES += GRAPH_GATE.replace("KEYS", _keys).replace("EDGE", _edge)

CANDIDATES += "\nORDER BY m.movie_id\n"

COUNT_FILMS = "SELECT count(*) FROM movies"

SCORES = """
SELECT d.movie_id, d.content, d.data_kind,
       1 - (v.embedding <=> %(query)s::vector) AS score
FROM movie_data d
JOIN movie_vectors v
  ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
WHERE d.data_kind = ANY(%(kinds)s)
  AND d.movie_id = ANY(%(allowed)s)
"""

# HOW MANY FILMS REACH THE RERANKER
# Retrieve wide, rerank to narrow. The right answer may sit at rank 4-8 by vector, and a
# top-3 retrieval never shows it to the reranker at all — the wide net is what gives the
# reranker something to rescue. At 20 films this is the whole catalogue; it starts
# costing something at a thousand.
RERANK_CANDIDATES = 50

SEED_TEXT = """
SELECT content
FROM movie_data
WHERE movie_id = %(movie_id)s AND data_kind = 'mood_feel' AND seq = 0
"""

SEED_VECTORS = """
SELECT d.data_kind, v.embedding
FROM movie_data d
JOIN movie_vectors v
  ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
WHERE d.movie_id = %(movie_id)s AND d.data_kind = ANY(%(kinds)s)
"""

DISPLAY_TEXT = """
SELECT movie_id, data_kind, content
FROM movie_data
WHERE movie_id = ANY(%(ids)s) AND data_kind = ANY(%(kinds)s) AND seq = 0
"""

BY_NAME = """
SELECT node_key, name
FROM graph_nodes
WHERE node_type = %(type)s AND lower(name) = lower(%(name)s)
"""

LIKE_NAME = """
SELECT node_key, name
FROM graph_nodes
WHERE node_type = %(type)s AND name ILIKE %(pattern)s
ORDER BY name
"""

BY_TITLE = """
SELECT movie_id, title,
       tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies
WHERE lower(title) = lower(%(title)s)
"""

LIKE_TITLE = """
SELECT movie_id, title,
       tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies
WHERE title ILIKE %(pattern)s
ORDER BY title
"""


def best_per_film(rows):
    """Each film's strongest row — ranked WITHIN ITS OWN KIND, not by raw score.

    THE BUG THIS EXISTS TO KILL, measured 7 Sep 2026
        For "a prisoner digging a tunnel" the scores came out in three separate bands:
        scenes 0.338-0.508, premises 0.457-0.569, themes 0.513-0.561. Not overlapping
        ranges with one winning on merit — different SCALES. Taking the maximum across
        kinds therefore always picked the kind centred highest, and a plot scene could
        never win however well it matched. Across the whole catalogue, ONE film in
        twenty had its best scene beat its own premise. The 148 scene rows were not
        losing the competition; they were never in it.

        A raw cosine is comparable within one kind of text and meaningless across two.
        Same law as "fuse on rank, never on score thresholds" — one level down.

    SO: each row is scored against the rows of ITS OWN KIND for THIS query, and what
    comes back is its position in that band, 0 to 1. A scene in the top 2% of scenes
    beats a premise in the top 40% of premises, even though its number is smaller.

    Returns {movie_id: (rank, content, kind, raw, over_floor)} where
        rank        0-1, position among text of the same kind. What ranking uses.
        raw         the cosine itself. Kept because rank alone cannot say whether
                    ANYTHING here is any good — with twenty films the best is always
                    rank 1.0, even when nothing fits.
        over_floor  raw minus the worst raw score OF THAT KIND. An absolute margin that
                    is still comparable across kinds, because each is measured against
                    its own band's bottom.
    """
    bands = {}
    for _, _, kind, score in rows:
        bands.setdefault(kind, []).append(score)
    for kind in bands:
        bands[kind].sort()

    def rank_within(kind, score):
        band = bands[kind]
        if len(band) < 2:
            return 1.0                     # a band of one has no position to report
        return bisect.bisect_left(band, score) / (len(band) - 1)

    best = {}
    for movie_id, content, kind, score in rows:
        rank = rank_within(kind, score)
        floor = bands[kind][0]
        candidate = (rank, content, kind, score, round(score - floor, 4))
        # Rank first, raw score as the tie-break: two rows can share a rank in a small
        # band, and the closer one should represent the film.
        if movie_id not in best or (rank, score) > (best[movie_id][0], best[movie_id][3]):
            best[movie_id] = candidate
    return best


def resolve_title(conn, title):
    """A written title -> one film. Returns (row, note). Exactly one of them is None.

    Nought matches and several matches are both NORMAL outcomes of a person typing a
    film's name from memory, not errors. Neither is answered by guessing: the note comes
    back so the caller can ask one short question instead.
    """
    row = conn.execute(BY_TITLE, {"title": title}).fetchone()
    if row:
        return row, None

    near = conn.execute(LIKE_TITLE, {"pattern": f"%{title}%"}).fetchall()
    if len(near) == 1:
        return near[0], None
    if not near:
        return None, f"No film called {title!r} is in the catalogue."
    names = ", ".join(r[1] for r in near[:6])
    return None, f"{title!r} matches several films here: {names}. Which one?"


def resolve_node(conn, node_type, name):
    """A written name -> node keys. Returns (keys, note); exactly one is None.

    Exact first, then a contains-match, because people type "Nolan" and the graph holds
    "Christopher Nolan". SEVERAL matches are kept, not narrowed — two actors sharing a
    surname is a real thing, and returning both films is a better answer than silently
    picking one. NO match is reported and stops the search: a filter that matches nothing
    would otherwise delete the entire catalogue and report an empty result as though the
    question had been understood.
    """
    rows = conn.execute(BY_NAME, {"type": node_type, "name": name}).fetchall()
    if not rows:
        rows = conn.execute(LIKE_NAME, {"type": node_type,
                                        "pattern": f"%{name}%"}).fetchall()
    if not rows:
        return None, (f"No {node_type} called {name!r} appears in this catalogue, so "
                      f"nothing was searched. This is a fact, not a weak match — the "
                      f"name is simply not here.")
    return [key for key, _ in rows], None


def attach_display_text(conn, films):
    """Give each returned film its spoiler-free text. Used by BOTH result paths.

    Written once because a listed film and a ranked film need exactly the same thing,
    and a rule copied into two branches is a rule that will diverge in one of them.
    """
    if not films:
        return
    rows = conn.execute(DISPLAY_TEXT, {"ids": [f["movie_id"] for f in films],
                                       "kinds": sorted(DISPLAYABLE)}).fetchall()
    texts = {}
    for movie_id, kind, content in rows:
        texts.setdefault(movie_id, {})[kind] = content
    for film in films:
        film["show"] = texts.get(film["movie_id"], {})


@traceable(run_type="retriever", name="retrieval.search")
def search(mood=None, theme=None, situation=None, similar_to=None,
           director=None, actor=None, genre=None,
           max_runtime=None, min_year=None, max_year=None, limit=3):
    """Returns (results, notes). `notes` are FACTS for the caller to use or ignore.

    Facts, not instructions. A tool result that says "drop this constraint and try
    again" is obeyed in the wrong cases and ignored in the right ones; a tool result
    that says "the runtime limit removed 10 of 20 films" survives being read a hundred
    times and leaves the judgement where it belongs.
    """
    notes = []

    with psycopg.connect(DATABASE_URL) as conn:
        total = conn.execute(COUNT_FILMS).fetchone()[0]

        # Names resolve to node keys BEFORE any film is looked at. A name that is not in
        # the graph stops the search rather than filtering everything away and returning
        # a confident nothing.
        gates, gate_used = {}, False
        for field, node_type, value in (("director_keys", "person", director),
                                        ("actor_keys", "person", actor),
                                        ("genre_keys", "genre", genre)):
            if not value:
                gates[field] = None
                continue
            keys, problem = resolve_node(conn, node_type, value)
            if problem:
                return [], notes + [problem]
            gates[field] = keys
            gate_used = True
            notes.append(f"{node_type.title()} {value!r} matched {len(keys)} "
                         f"node(s) in the graph; only films with that edge were kept.")

        rows = conn.execute(CANDIDATES, {"max_runtime": max_runtime,
                                         "min_year": min_year,
                                         "max_year": max_year,
                                         **gates}).fetchall()
        films = {r[0]: {"movie_id": r[0], "title": r[1], "year": r[2],
                        "runtime_minutes": r[3], "series": r[4]} for r in rows}

        removed = total - len(films)
        if removed:
            # A hard filter is enforced exactly, which is its value AND its danger: a
            # spurious constraint does not degrade the answer, it deletes the right one
            # and leaves no trace. So it always says what it took.
            notes.append(f"Hard filters removed {removed} of {total} films.")
        if not films:
            notes.append("Nothing survived the filters, so nothing was ranked.")
            return [], notes

        # label -> (data_kind, query vector). Keyed by LABEL, not by constraint, so
        # that "like Alien" and "funny" can both be live at once: they are both
        # mood_feel comparisons, so they share a scale and min() over them is honest.
        wanted = {}

        if similar_to:
            found, problem = resolve_title(conn, similar_to)
            if problem:
                return [], notes + [problem]

            seed_id, seed_title, seed_series = found
            seed_kinds = [kind for c in SEEDABLE for kind in CONSTRAINT_KIND[c]]
            seeded = dict(conn.execute(SEED_VECTORS, {
                "movie_id": seed_id, "variant": EMBED_VARIANT,
                "kinds": seed_kinds}).fetchall())

            # The film's OWN stored vectors are the query. Re-describing it in words and
            # embedding that again would cost a call and land somewhere slightly else,
            # for nothing: this is already the exact point in the space to search around.
            for constraint in SEEDABLE:
                for kind in CONSTRAINT_KIND[constraint]:
                    if kind in seeded:
                        wanted[f"like {seed_title}"] = ((kind,), seeded[kind])

            # Itself, obviously. And its sequels: they are the most similar films in
            # existence and the least useful answer, because the person has seen them.
            dropped = [f for f in films.values()
                       if f["movie_id"] == seed_id
                       or (seed_series and f["series"] == seed_series)]
            for film in dropped:
                del films[film["movie_id"]]

            siblings = len(dropped) - 1
            notes.append(
                f"Searching by what {seed_title} FEELS LIKE — not by its year, length, "
                f"cast or subject, which would ask for films like it on paper rather "
                f"than films like it to watch."
                + (f" {siblings} other film(s) in {seed_series} were excluded as "
                   f"sequels." if siblings else ""))

        # What the person SAID is added alongside what was copied from the film, not
        # instead of it. "Similar to Alien, but funnier" is a modification, not a
        # replacement: they want a film that is Alien-ish AND funny, and dropping the
        # seed turns the request into a plain search for funny films. Measured 7 Sep —
        # replacing gave Home Alone, which is funny and nothing like Alien.
        for constraint, text in (("mood", mood), ("theme", theme),
                                 ("situation", situation)):
            if text:
                wanted[constraint] = (CONSTRAINT_KIND[constraint],
                                      as_literal(embed(text)))    # kinds is a tuple

        if not wanted:
            # A FACT QUERY IS A SET, AND A SET IS A COMPLETE ANSWER.
            #
            # "Films by Nolan" needs listing, not ranking. Returning nothing because
            # there was no mood to sort by threw away two films the graph had correctly
            # found — the filter worked and the code then reported it as a failure.
            #
            # Newest first, because with no relevance to order by, recency is the only
            # ordering that is honest about being arbitrary. Nothing here is a score,
            # and nothing pretends to be.
            if not gate_used:
                notes.append("Nothing to search on: no mood, situation, theme, "
                             "comparison film, person or genre was given.")
                return [], notes

            listed = sorted(films.values(),
                            key=lambda f: (f["year"] or 0), reverse=True)[:limit]
            for film in listed:
                film["listed_only"] = True
            notes.append(
                f"{len(films)} film(s) carry that fact. They are LISTED, not ranked — "
                f"nothing was asked about how they feel, so no order here means "
                f"anything. Say what they are; do not imply one is a better fit.")
            attach_display_text(conn, listed)
            return listed, notes

        allowed = list(films)
        scored = {}
        for label, (kinds, query_vector) in wanted.items():
            rows = conn.execute(SCORES, {"query": query_vector, "kinds": list(kinds),
                                         "variant": EMBED_VARIANT,
                                         "allowed": allowed}).fetchall()
            scored[label] = best_per_film(rows)

        results = []
        unscorable = 0
        for movie_id, film in films.items():
            per = {c: scored[c].get(movie_id) for c in wanted}
            if any(hit is None for hit in per.values()):
                # No stored text of that kind, so this film cannot be judged on a
                # constraint that WAS asked for. Silence is not a low score.
                unscorable += 1
                continue
            film = dict(film)

            # A film is as good as its WEAKEST satisfied constraint, and the comparison
            # is now between ranks, which are on one scale by construction. Taking the
            # minimum of two raw cosines was what let theme silently decide every
            # "similar to" result this morning.
            film["score"] = min(hit[0] for hit in per.values())
            film["per_constraint"] = {
                label: {"rank": round(hit[0], 3), "raw": round(hit[3], 3),
                        "over_floor": hit[4]}
                for label, hit in per.items()
            }
            film["over_floor"] = min(hit[4] for hit in per.values())

            # The kind that actually WON, per film — not the kinds that were searched.
            # One film may match on its premise and the next on a scene from its final
            # act, and only the first of those may be printed.
            film["matched_kind"] = {label: hit[2] for label, hit in per.items()
                                    if hit[2] in DISPLAYABLE}
            # A constraint can match on text that must never be shown — a theme, or a
            # scene from late in the film. Say the evidence EXISTS and is withheld. A
            # result with a score and no reason is a gap, and a gap gets filled from
            # the model's training.
            film["matched_unseen"] = [
                label for label, hit in per.items()
                if hit[2] not in DISPLAYABLE
            ]
            # The text the reranker reads. It is the film's STRONGEST matched row,
            # displayable or not: a third-act scene may not be printed to a person, and
            # is exactly what a cross-encoder should be judging. This never leaves the
            # server — it is stripped before the tool renders anything.
            strongest = max(per.values(), key=lambda hit: hit[0])
            film["_rerank_text"] = strongest[1]
            results.append(film)

        if unscorable:
            notes.append(f"{unscorable} film(s) had no text for something that was "
                         f"asked for and were not ranked.")

        results.sort(key=lambda f: (f["score"], f["over_floor"]), reverse=True)

        if results:
            notes.append(
                "Ranked by how each film's best text scores AMONG TEXT OF ITS OWN KIND "
                "— scenes against scenes, premises against premises. Their raw numbers "
                "sit in different bands, so comparing them directly would always pick "
                "the same kind and ignore the others.")

            # RANK SAYS WHICH FILM IS BEST. IT CANNOT SAY WHETHER ANY OF THEM IS GOOD:
            # the best film is rank 1.00 whether it fits perfectly or not at all. The
            # margin over its own band's floor is the absolute signal, and it is
            # reported rather than acted on — whether 0.02 is worth answering is a
            # judgement, and judgements belong to the caller.
            top = results[0]
            notes.append(
                f"Best film: raw {top['per_constraint'][list(top['per_constraint'])[0]]['raw']:.3f}, "
                f"{top['over_floor']:+.3f} above the bottom of its own band. A margin "
                f"near zero means nothing stood out, however high the rank looks.")

        # ── RERANK ────────────────────────────────────────────────────────────
        # A vector compares two summaries of meaning. A cross-encoder reads the query
        # and the film's text TOGETHER, which is how "creatures hunting people" is told
        # apart from "a tiger in the bathroom" — the thing one vector cannot do.
        #
        # And it solves the band problem for free: it judges relevance directly rather
        # than measuring distance, so its scores ARE comparable between a premise and a
        # plot scene. Vector rank chooses which text represents each film; the reranker
        # chooses the order.
        # THE RERANK QUERY MUST CARRY EVERY CONSTRAINT, NOT JUST THE TYPED ONES.
        #
        # It was built from the typed text alone, so "similar to Alien, but funny" asked
        # the reranker only for "funny" — and it returned Crazy, Stupid, Love. at #1, a
        # film scoring the WORST possible rank on the Alien half. The vector stage had
        # honoured both constraints and put it 18th; the reranker, seeing half the
        # request, overruled it. A final ranker given a partial query does not refine
        # the earlier work, it discards it.
        parts = [t for t in (mood, situation, theme) if t]
        if similar_to:
            seed_row = conn.execute(SEED_TEXT, {"movie_id": seed_id}).fetchone()
            if seed_row:
                parts.insert(0, seed_row[0])
        query_text = " ".join(parts)

        pool = results[:RERANK_CANDIDATES]
        if query_text and len(pool) > 1:
            try:
                ordered = rerank(query_text,
                                 [f["_rerank_text"] for f in pool],
                                 top_n=len(pool))
                by_index = {r["index"]: r["score"] for r in ordered}
                for position, film in enumerate(pool):
                    film["rerank"] = round(by_index.get(position, 0.0), 4)
                    film["vector_place"] = position + 1
                pool.sort(key=lambda f: f["rerank"], reverse=True)

                moved = sum(1 for new, film in enumerate(pool)
                            if film["vector_place"] != new + 1)
                top = pool[0]["rerank"]
                runner_up = pool[1]["rerank"] if len(pool) > 1 else 0.0
                notes.append(
                    f"Reranked {len(pool)} films by reading the query and each film's "
                    f"text together; {moved} changed position.")
                notes.append(
                    f"Relevance of the best film {top:.3f}, next best {runner_up:.3f}. "
                    f"UNLIKE the vector scores, this number means something on its own: "
                    f"it is relevance judged directly, not distance. Measured on this "
                    f"catalogue — a query with a real answer tops 0.3 and often 0.5, "
                    f"and a query with no answer here leaves everything under 0.1. "
                    f"A whole field below 0.1 means say you have nothing, not pick the "
                    f"highest of them.")
                results = pool
            except Exception as error:
                # A vendor outage costs QUALITY, never AVAILABILITY. The vector order
                # is a worse answer, not a failed one.
                notes.append(f"Reranker unavailable ({type(error).__name__}); films are "
                             f"in vector order, which is a weaker ranking.")

        results = results[:limit]
        for film in results:
            film.pop("_rerank_text", None)      # internal only; never leaves the server

        attach_display_text(conn, results)

    return results, notes


def _print(title, results, notes):
    print(f"\n\n=== {title}")
    for note in notes:
        print(f"  · {note}")
    if not results:
        print("  (nothing)")
    for film in results:
        print(f"\n  {film['score']:.3f}  {film['title']} ({film['year']}) "
              f"· {film['runtime_minutes']}m  {film['per_constraint']}")
        print(f"         {film['show'].get('premise', '')[:150]}")


if __name__ == "__main__":
    for label, kwargs in [
        ("mood: warm and comforting for a rainy evening",
         {"mood": "warm and comforting, cosy, for a rainy evening"}),
        ("mood: uneasy, something is wrong",
         {"mood": "uneasy, creeping dread, something is wrong"}),
        ("mood + runtime under 100 minutes",
         {"mood": "funny and chaotic", "max_runtime": 100}),
        ("similar to Alien", {"similar_to": "Alien"}),
        ("similar to Toy Story", {"similar_to": "Toy Story"}),
        ("similar to Alien, but funnier", {"similar_to": "Alien", "mood": "funny"}),
        ("a title that is not here", {"similar_to": "Jaws"}),
    ]:
        _print(label, *search(**kwargs))
