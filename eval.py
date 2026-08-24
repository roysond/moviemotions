"""Run the eval set against retrieval and report hits."""

import json

from core import search

TOP_N = 3

cases = json.load(open("eval_cases.json"))
passed = failed = watched = 0

for case in cases:
    results = search(case["query"], limit=TOP_N)
    titles = [film["title"] for film in results]
    expected = case["expect_any_of"]
    hits = [title for title in titles if title in expected]

    if not expected:
        verdict, symbol = "WATCH", "~"
        watched += 1
    elif hits:
        verdict, symbol = f"PASS {len(hits)}/{len(expected)}", "+"
        passed += 1
    else:
        verdict, symbol = f"FAIL 0/{len(expected)}", "x"
        failed += 1

    print(f'\n[{symbol}] {verdict}  "{case["query"]}"')
    for position, film in enumerate(results, start=1):
        mark = "*" if film["title"] in expected else " "
        print(f"      {position}.{mark} {film['score']:.4f}  [{film['method']:>6}]  {film['title']}")
    if case["note"]:
        print(f"      note: {case['note']}")

print(f"\npassed {passed}   failed {failed}   watch {watched}")