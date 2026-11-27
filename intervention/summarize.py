"""Summarize a finished intervention run under intervention/protocol.json (CPU only)."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, write


def boot(stat, n, repeats=2000, seed=453):
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(repeats)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(point=float(point), ci95=[float(lo), float(hi)])


def analyse_type(units, cfg):
    n = len(units)
    kl = {ref: {c: np.array([u['teacher_forced'][ref][c]['kl_mean'] for u in units]) for c in units[0]['teacher_forced'][ref]}
          for ref in ('B', 'C')}
    KB, KC = kl['B'], kl['C']
    G = boot(lambda i: 1 - KB['C_plus_mean'][i].mean() / KB['C'][i].mean(), n)
    D = boot(lambda i: (KB['C_plus_random'][i] - KB['C_plus_mean'][i]).mean(), n)
    margin = 0.10
    if G['ci95'][0] > 0 and D['ci95'][0] > 0:
        decision = 'direction_specific_effect'
    elif -margin <= G['ci95'][0] and G['ci95'][1] <= margin:
        decision = 'effectively_inert'
    else:
        decision = 'inconclusive'
    secondary = {c: boot(lambda i, c=c: 1 - KB[c][i].mean() / KB['C'][i].mean(), n)
                 for c in ('C_minus_mean', 'C_plus_own', 'C_plus_wrong', 'C_plus_random')}
    secondary['B_minus_mean_toward_C'] = boot(lambda i: 1 - KC['B_minus_mean'][i].mean() / KC['B'][i].mean(), n)
    gens = defaultdict(list)
    for u in units:
        for c, g in u['generations'].items():
            gens[c].append(g)
    behaviour = {c: dict(passed=sum(g['score']['passed'] for g in gs), exact_vs_B=sum(g['exact_vs_B'] for g in gs),
                         exact_vs_C=sum(g['exact_vs_C'] for g in gs), terminations=dict(Counter(g['termination'] for g in gs)),
                         mean_words=float(np.mean([g['score']['words'] for g in gs])), n=len(gs)) for c, gs in gens.items()}
    return dict(n_sources=n,
                mean_kl_along_B={c: float(v.mean()) for c, v in KB.items()},
                mean_kl_along_C={c: float(v.mean()) for c, v in KC.items()},
                mean_first_token_kl_along_B={c: float(np.mean([u['teacher_forced']['B'][c]['kl_first'] for u in units])) for c in KB},
                primary_gap_closed=G, primary_specificity_random_minus_mean=D, decision=decision, equivalence_margin=margin,
                secondary_gap_closed=secondary, behaviour=behaviour,
                v21_reproduction={c: sum(u['v21_reproduction'][c] for u in units) for c in ('B', 'C')})


def cross_format(run, handoff):
    import torch
    files = sorted((run / 'cross_format').glob('*.pt'))
    if not files or handoff is None:
        return None
    blob = torch.load(handoff / 'activations.pt', map_location='cpu', weights_only=True)
    by = defaultdict(dict)
    for r in blob['records']:
        by[int(r['layer'])].setdefault(r['source_id'], {})[r['condition']] = r['vector'].double()
    out = {}
    states = [torch.load(f, map_location='cpu', weights_only=True) for f in files]
    for layer in states[0]['layers']:
        w = torch.stack([v['nat'] - v['flat'] for v in by[layer].values()]).mean(0)
        wu = w / w.norm()
        row = {}
        base = None
        for variant in ('C', 'C_qa', 'C_json'):
            D = torch.stack([s['states']['B'][layer].double() - s['states'][variant][layer].double() for s in states])
            m = D.mean(0)
            if variant == 'C':
                base = m
            row[variant] = dict(cos_mean_with_v21_dir=float(m @ wu / m.norm()), paired_sign=float(((D @ wu) > 0).double().mean()),
                                cos_with_this_run_C=float(m @ base / (m.norm() * base.norm())), norm_ratio=float(m.norm() / w.norm()))
        out[layer] = row
    return dict(sources=len(states), layers=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--handoff', type=Path, help='folder with v2.1 activations.pt, for the cross-format check')
    a = p.parse_args()
    cfg = read(a.run / 'protocol.json')
    assert read(a.run / 'status.json')['phase'] == 'completed', 'run not completed'
    units = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((a.run / 'units').glob('*.json'))]
    assert len(units) == cfg['sources'] * len(cfg['request_types'])
    result = {rtype: analyse_type([u for u in units if u['request_type'] == rtype], cfg) for rtype in cfg['request_types']}
    result['cross_format'] = cross_format(a.run, a.handoff)
    result['note'] = 'Primary = ordinary. Change requests and everything outside primary_* are secondary or descriptive.'
    write(a.run / 'summary.json', result)
    print(json.dumps({k: (v['decision'], v['primary_gap_closed'], v['primary_specificity_random_minus_mean']) for k, v in result.items()
                      if isinstance(v, dict) and 'decision' in v}, indent=2))


if __name__ == '__main__':
    main()
