"""Build the frozen intervention vectors from the released v2.1 activations (CPU only)."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, rows, sha, write

LAYER = 16
FOLDS = 5
SALT = 'intervention-v1'


def pairs(path, layer):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    assert blob['is_synthetic'] is False and blob['run_kind'] == 'confirmation'
    by = defaultdict(dict)
    for r in blob['records']:
        if int(r['layer']) == layer:
            by[r['source_id']][r['condition']] = r['vector'].double()
    assert all(set(v) == {'nat', 'flat'} for v in by.values())
    return {s: v['nat'] - v['flat'] for s, v in by.items()}


def fold_of(sources):
    order = sorted(sources, key=lambda s: hashlib.sha256((s + SALT).encode()).hexdigest())
    return {s: i % FOLDS for i, s in enumerate(order)}


def build(handoff_dir, run_dir):
    ordinary = pairs(handoff_dir / 'activations.pt', LAYER)
    change = pairs(handoff_dir / 'activations_change.pt', LAYER)
    histories = {h['source_id']: h for h in rows(run_dir / 'histories.jsonl')}
    sources = sorted(ordinary)
    assert sorted(change) == sources == sorted(histories) and len(sources) == 120
    fold = fold_of(sources)
    fold_mean = {k: torch.stack([ordinary[s] for s in sources if fold[s] != k]).mean(0) for k in range(FOLDS)}
    by_rule = defaultdict(list)
    for s in sources:
        by_rule[histories[s]['inst_name']].append(s)
    wrong_of = {}
    for members in by_rule.values():
        members = sorted(members)
        for i, s in enumerate(members):
            wrong_of[s] = members[(i + 1) % len(members)]
    vec, meta = {}, {}
    for i, s in enumerate(sources):
        m = fold_mean[fold[s]]
        g = np.random.default_rng(453 + i).standard_normal(m.numel())
        rnd = torch.tensor(g / np.linalg.norm(g), dtype=torch.float64) * m.norm()
        vec[s] = dict(mean=m.float(), random=rnd.float(),
                      own=dict(ordinary=ordinary[s].float(), change=change[s].float()),
                      wrong=dict(ordinary=ordinary[wrong_of[s]].float(), change=change[wrong_of[s]].float()))
        meta[s] = dict(fold=fold[s], wrong_source=wrong_of[s], rule=histories[s]['inst_name'],
                       norm_mean=float(m.norm()), norm_own_ordinary=float(ordinary[s].norm()),
                       cos_own_mean=float(ordinary[s] @ m / (ordinary[s].norm() * m.norm())))
    full = torch.stack([ordinary[s] for s in sources]).mean(0).float()
    return dict(schema_version=1, layer=LAYER, folds=FOLDS, salt=SALT, sources=sources, vectors=vec,
                full_sample_mean=full, meta=meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff', type=Path, required=True, help='folder with activations.pt and activations_change.pt')
    p.add_argument('--run', type=Path, required=True, help='extracted confirmation-v2_1 run folder')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Directions are frozen; write a new file instead of overwriting'
    d = build(a.handoff, a.run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    norms = [m['norm_mean'] for m in d['meta'].values()]
    cos = [m['cos_own_mean'] for m in d['meta'].values()]
    summary = dict(directions_sha256=sha(a.out), layer=d['layer'], sources=len(d['sources']),
                   inputs_sha256={f: sha(a.handoff / f) for f in ['activations.pt', 'activations_change.pt']},
                   histories_sha256=sha(a.run / 'histories.jsonl'),
                   mean_norm=[min(norms), max(norms)], cos_own_with_crossfit_mean=[min(cos), float(np.mean(cos)), max(cos)],
                   fold_sizes=[sum(m['fold'] == k for m in d['meta'].values()) for k in range(FOLDS)])
    write(a.out.with_suffix('.json'), summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
