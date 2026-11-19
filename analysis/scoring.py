"""Rule-compliance scoring for the A/B/C comparison.

Wraps the MT-Eval category checkers, adds the checks they are too shallow to
make, and reports paired differences with source-level bootstrap intervals.

Run directly to score the existing 02_strategy batch:

    python scoring.py --repo /path/to/dialogue-migration-handoff
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from splits import bootstrap_ci, paired_bootstrap_ci

# --------------------------------------------------------------------------
# MT-Eval checkers.  inst_name -> (class name in global_inst, kwargs from
# inst_args).  Imported lazily so this module can be unit-tested without the
# third_party tree present.
# --------------------------------------------------------------------------
CHECKER_CLASS = {
    "startend:end_phrase": "EndPhrase",
    "format:json_format": "JsonFormat",
    "format:bullet_list": "BulletList",
    "combination:two_responses": "TwoResponses",
}


def load_mt_eval(repo: Path):
    sys.path.insert(0, str(repo / "third_party" / "mt_eval"))
    import global_inst  # noqa: E402
    return global_inst


def make_checker(global_inst, inst_name: str, inst_args: dict) -> Callable[[str], bool]:
    cls = getattr(global_inst, CHECKER_CLASS[inst_name])
    inst = cls(**inst_args) if inst_args else cls()
    return inst.check_following


# --------------------------------------------------------------------------
# Known shallowness of the upstream checkers.
#
# TwoResponses only asks that the two halves be unequal as strings.  "A" and
# "A." pass.  We keep the upstream verdict as the primary measure (it is what
# was pre-registered) and report these flags beside it, so that a difference
# driven by a degenerate second response is visible rather than silent.
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower().rstrip(".!?"))


def two_responses_degenerate(response: str) -> bool:
    """True when the halves pass the upstream check but are near-identical."""
    parts = [p for p in response.split("******") if p.strip()]
    if len(parts) != 2:
        return False
    a, b = _norm(parts[0]), _norm(parts[1])
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    # one is a prefix of the other, or they differ by under 5% of characters
    return longer.startswith(shorter) or (
        len(longer) and abs(len(a) - len(b)) / len(longer) < 0.05 and a[:40] == b[:40]
    )


def empty_or_truncated(row: dict) -> bool:
    return row.get("truncated", False) or not (row.get("text") or "").strip()


QUALITY_FLAGS: dict[str, Callable[[dict], bool]] = {
    "degenerate_two_responses": lambda r: (
        r["inst_name"] == "combination:two_responses"
        and two_responses_degenerate(r.get("text", ""))
    ),
    "empty_or_truncated": empty_or_truncated,
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_rows(rows: list[dict], histories: dict[str, dict], global_inst) -> list[dict]:
    """Attach ``compliant`` and the quality flags to every row."""
    checkers: dict[str, Callable[[str], bool]] = {}
    scored = []
    for r in rows:
        cid = r["case_id"]
        if cid not in checkers:
            h = histories[cid]
            checkers[cid] = make_checker(global_inst, h["inst_name"],
                                         json.loads(h["inst_args"])
                                         if isinstance(h["inst_args"], str)
                                         else h["inst_args"])
        out = dict(r)
        out["compliant"] = bool(checkers[cid](r.get("text", "")))
        for name, fn in QUALITY_FLAGS.items():
            out[name] = bool(fn(out))
        scored.append(out)
    return scored


def compliance_by_method(scored: list[dict]) -> dict[str, dict]:
    """Per-method rate with a source-level bootstrap interval."""
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in scored:
        by_method[r["method"]][r["case_id"]].append(float(r["compliant"]))

    out = {}
    for method, per_source in by_method.items():
        point, lo, hi = bootstrap_ci(per_source)
        n = sum(len(v) for v in per_source.values())
        out[method] = dict(rate=point, lo=lo, hi=hi, n=n, sources=len(per_source))
    return out


def paired_difference(scored: list[dict], method_a: str, method_b: str) -> dict:
    """mean(a - b) over source conversations, with a paired bootstrap CI."""
    def per_source_mean(method: str) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for r in scored:
            if r["method"] == method:
                acc[r["case_id"]].append(float(r["compliant"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    a, b = per_source_mean(method_a), per_source_mean(method_b)
    diff, lo, hi = paired_bootstrap_ci(a, b)
    return dict(a=method_a, b=method_b, diff=diff, lo=lo, hi=hi,
                sources=len(set(a) & set(b)))


def manual_review_sample(scored: list[dict], k: int = 30, seed: int = 453) -> list[dict]:
    """A reproducible subset for human checking.

    Stratified by rule and by method, and every flagged row is forced in: the
    cases where the checker is most likely to be wrong are exactly the ones a
    human should see.
    """
    rng = random.Random(seed)
    flagged = [r for r in scored
               if any(r.get(f) for f in QUALITY_FLAGS)]
    pool = [r for r in scored if r not in flagged]

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        strata[(r["inst_name"], r["method"])].append(r)

    picks = list(flagged)
    keys = sorted(strata, key=str)
    while len(picks) < k and keys:
        for key in list(keys):
            if len(picks) >= k:
                break
            bucket = strata[key]
            if not bucket:
                keys.remove(key)
                continue
            picks.append(bucket.pop(rng.randrange(len(bucket))))
    return picks[:max(k, len(flagged))]


def agreement_rate(reviewed: list[dict], human_key: str = "human_compliant") -> dict:
    """Checker/human agreement on the reviewed subset."""
    pairs = [(bool(r["compliant"]), bool(r[human_key]))
             for r in reviewed if human_key in r]
    if not pairs:
        return dict(n=0, agreement=None,
                    note=f"no rows carry '{human_key}' yet")
    agree = sum(c == h for c, h in pairs)
    return dict(n=len(pairs), agreement=agree / len(pairs),
                checker_only=sum(c and not h for c, h in pairs),
                human_only=sum(h and not c for c, h in pairs))


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--experiment", default="02_strategy")
    ap.add_argument("--outputs", default="outputs.jsonl")
    args = ap.parse_args()

    exp = args.repo / "experiments" / args.experiment
    rows = [json.loads(l) for l in (exp / args.outputs).open()]
    hist = {json.loads(l)["case_id"]: json.loads(l)
            for l in (exp / "histories.jsonl").open()}

    scored = score_rows(rows, hist, load_mt_eval(args.repo))

    print(f"scored {len(scored)} rows over {len({r['case_id'] for r in scored})} "
          f"source conversations\n")

    print("compliance by method (source-level bootstrap 95% CI)")
    for m, s in sorted(compliance_by_method(scored).items()):
        print(f"  {m:<14} {s['rate']:.3f}  [{s['lo']:.3f}, {s['hi']:.3f}]"
              f"   n={s['n']:3d}  sources={s['sources']}")

    methods = {r["method"] for r in scored}
    print("\npaired differences")
    for a, b in [("BReplay", "CPasted"), ("AOnline", "CPasted"),
                 ("CSpecificRule", "CPasted")]:
        if {a, b} <= methods:
            d = paired_difference(scored, a, b)
            print(f"  {a} - {b:<14} {d['diff']:+.3f}  "
                  f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  sources={d['sources']}")

    print("\nquality flags")
    for name in QUALITY_FLAGS:
        hits = [r for r in scored if r[name]]
        print(f"  {name:<26} {len(hits)}")
        for r in hits[:5]:
            print(f"      {r['case_id']} q{r['query']} {r['method']}")

    sample = manual_review_sample(scored)
    print(f"\nmanual review sample: {len(sample)} rows "
          f"({sum(1 for r in sample if any(r[f] for f in QUALITY_FLAGS))} flagged)")
    print(" ", agreement_rate(sample))


if __name__ == "__main__":
    main()
