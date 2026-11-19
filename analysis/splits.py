"""Source-level splitting and source-level bootstrap.

Both rows derived from one source conversation (native and pasted) are highly
correlated.  Splitting at the row level would put one member of a pair in train
and the other in test and inflate accuracy, so every split and every confidence
interval in this project is taken over *source conversations*, never over rows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np

__all__ = ["split_sources", "assign_rows", "bootstrap_ci", "paired_bootstrap_ci"]


def _stable_hash(key: str, salt: str) -> float:
    """Deterministic uniform value in [0, 1) for a key.

    Uses a hash rather than a shuffle so that adding new conversations does not
    move existing ones between splits.
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_sources(
    source_ids: Iterable[Hashable],
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    salt: str = "stat453",
    group_of: Callable[[Hashable], Hashable] | None = None,
) -> dict[Hashable, str]:
    """Assign each source conversation to ``train`` / ``val`` / ``test``.

    ``group_of`` keeps related conversations together: near-duplicates or
    excerpts of the same underlying document must not straddle a split.  When it
    is None each source is its own group.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    source_ids = list(dict.fromkeys(source_ids))  # de-duplicate, keep order
    group_of = group_of or (lambda s: s)

    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for sid in source_ids:
        groups[group_of(sid)].append(sid)

    train_hi = fractions[0]
    val_hi = fractions[0] + fractions[1]

    assignment: dict[Hashable, str] = {}
    for gid, members in groups.items():
        u = _stable_hash(str(gid), salt)
        name = "train" if u < train_hi else ("val" if u < val_hi else "test")
        for sid in members:
            assignment[sid] = name
    return assignment


def assign_rows(rows: Sequence[dict], assignment: dict[Hashable, str],
                source_key: str = "case_id") -> list[str]:
    """Map each row to its source's split.  Raises on an unseen source."""
    out = []
    for r in rows:
        sid = r[source_key]
        if sid not in assignment:
            raise KeyError(f"source {sid!r} has no split assignment")
        out.append(assignment[sid])
    return out


def bootstrap_ci(values_by_source: dict[Hashable, Sequence[float]],
                 n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 453) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling *sources* with replacement.

    ``values_by_source`` maps a source id to the measurements taken on it.  The
    statistic is the mean over all measurements; resampling happens at the
    source level so the interval reflects variation across conversations rather
    than the near-zero variation within one pair.
    """
    sources = list(values_by_source)
    if not sources:
        raise ValueError("no sources")
    per_source = [np.asarray(values_by_source[s], dtype=float) for s in sources]

    point = float(np.concatenate(per_source).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sources), size=(n_boot, len(sources)))
    stats = np.array([np.concatenate([per_source[i] for i in row]).mean()
                      for row in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_ci(a_by_source: dict[Hashable, float],
                        b_by_source: dict[Hashable, float],
                        n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 453) -> tuple[float, float, float]:
    """Paired difference ``mean(a - b)`` with a source-level bootstrap CI.

    Both conditions are run on the same conversation, so the estimand is the
    paired difference.  Pairing removes between-conversation heterogeneity,
    which is large relative to the effect being measured.
    """
    shared = sorted(set(a_by_source) & set(b_by_source), key=str)
    if not shared:
        raise ValueError("no sources present in both conditions")
    diffs = np.array([a_by_source[s] - b_by_source[s] for s in shared], float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    stats = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)
