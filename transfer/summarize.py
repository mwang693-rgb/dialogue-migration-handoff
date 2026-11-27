"""Summarize a finished transfer run under transfer/protocol.json (CPU only)."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import read, write

GROUPS = ['short', 'open', 'structured', 'near', 'external']
TRANSFER = ('short', 'open', 'structured')
MARGIN = 0.05


def cluster_boot(values, clusters, repeats=4000, seed=453, level=97.5):
    """Mean of per-unit values with a history-cluster percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    keys = sorted(set(clusters))
    members = [np.array([i for i, c in enumerate(clusters) if c == k]) for k in keys]
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for r in range(repeats):
        pick = rng.integers(0, len(keys), len(keys))
        draws[r] = values[np.concatenate([members[j] for j in pick])].mean()
    tail = (100 - level) / 2
    lo, hi = np.percentile(draws, [tail, 100 - tail])
    return dict(point=float(values.mean()), ci=[float(lo), float(hi)], level=level, n_units=len(values), n_histories=len(keys))


def logn(u, c):
    return np.log(u['generations'][c]['n'])


def kl(u, ref, c, field='kl_mean'):
    return u['teacher_forced'][ref][c][field]


def idx(a, b):
    return 0.0 if a + b == 0 else (a - b) / (a + b)


def stats(units, level=97.5):
    if not units:
        return None
    cl = [u['history_key'] for u in units]
    b = lambda f: cluster_boot([f(u) for u in units], cl, level=level)
    out = dict(
        L_sign=b(lambda u: logn(u, 'C_plus_mean') - logn(u, 'C_minus_mean')),
        L_random=b(lambda u: logn(u, 'C_plus_mean') - logn(u, 'C_plus_random')),
        S_random=b(lambda u: idx(kl(u, 'B', 'C_plus_random'), kl(u, 'B', 'C_plus_mean'))),
        S_sign=b(lambda u: idx(kl(u, 'B', 'C_minus_mean'), kl(u, 'B', 'C_plus_mean'))),
        length_B_minus_C=b(lambda u: logn(u, 'B') - logn(u, 'C')),
        length_Cplus_minus_C=b(lambda u: logn(u, 'C_plus_mean') - logn(u, 'C')),
        length_Cminus_minus_C=b(lambda u: logn(u, 'C_minus_mean') - logn(u, 'C')),
        length_Crandom_minus_C=b(lambda u: logn(u, 'C_plus_random') - logn(u, 'C')),
        length_Bminus_minus_B=b(lambda u: logn(u, 'B_minus_mean') - logn(u, 'B')),
        S_random_first=b(lambda u: idx(kl(u, 'B', 'C_plus_random', 'kl_first'), kl(u, 'B', 'C_plus_mean', 'kl_first'))),
        S_sign_first=b(lambda u: idx(kl(u, 'B', 'C_minus_mean', 'kl_first'), kl(u, 'B', 'C_plus_mean', 'kl_first'))),
    )
    base = np.mean([kl(u, 'B', 'C') for u in units])
    out['gap_closed'] = {c: float(1 - np.mean([kl(u, 'B', c) for u in units]) / base) if base > 0 else None
                         for c in ('C_plus_mean', 'C_minus_mean', 'C_plus_random')}
    out['mean_kl_along_B'] = {c: float(np.mean([kl(u, 'B', c) for u in units])) for c in units[0]['teacher_forced']['B']}
    out['mean_kl_along_C'] = {c: float(np.mean([kl(u, 'C', c) for u in units])) for c in units[0]['teacher_forced']['C']}
    gens = defaultdict(list)
    for u in units:
        for c, g in u['generations'].items():
            gens[c].append(g)
    out['behaviour'] = {c: dict(mean_tokens=float(np.mean([g['n'] for g in gs])), median_tokens=float(np.median([g['n'] for g in gs])),
                                exact_vs_B=sum(g['exact_vs_B'] for g in gs), exact_vs_C=sum(g['exact_vs_C'] for g in gs),
                                terminations=dict(Counter(g['termination'] for g in gs))) for c, gs in gens.items()}
    return out


def decide(s):
    p1 = s['P1']
    if p1['L_sign']['ci'][0] > 0 and p1['L_random']['ci'][0] > 0:
        d1 = 'length_effect_transfers'
    elif -MARGIN <= p1['L_sign']['ci'][0] and p1['L_sign']['ci'][1] <= MARGIN:
        d1 = 'no_length_effect'
    else:
        d1 = 'inconclusive'
    p2 = s['P2']
    inside = lambda x: -MARGIN <= x['ci'][0] and x['ci'][1] <= MARGIN
    if p2['S_random']['ci'][0] > 0 and p2['S_sign']['ci'][0] > 0:
        d2 = 'kl_specificity_transfers'
    elif inside(p2['S_random']) and inside(p2['S_sign']):
        d2 = 'no_kl_specificity'
    else:
        d2 = 'inconclusive'
    return d1, d2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true', help='smoke tests only')
    a = p.parse_args()
    cfg = read(a.run / 'protocol.json')
    if not a.allow_incomplete:
        assert read(a.run / 'status.json')['phase'] == 'completed', 'run not completed'
    units = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((a.run / 'units').glob('*.json'))]
    if not a.allow_incomplete:
        assert len(units) == cfg['budget']['units']
    by_group = {g: [u for u in units if u['group'] == g] for g in GROUPS}
    open_s = stats(by_group['open'])
    transfer_s = stats([u for u in units if u['group'] in TRANSFER])
    result = dict(
        P1=dict(population='open', L_sign=open_s['L_sign'], L_random=open_s['L_random']),
        P2=dict(population='short+open+structured', S_random=transfer_s['S_random'], S_sign=transfer_s['S_sign']))
    result['decision_P1'], result['decision_P2'] = decide(result)
    result['by_group'] = {g: stats(us) for g, us in by_group.items()}
    result['transfer_set'] = transfer_s
    pools = sorted({(u['group'], u['pool']) for u in units})
    result['by_pool'] = {f'{g}/{p_}': {k: v for k, v in stats([u for u in units if u['pool'] == p_ and u['group'] == g], level=95).items()
                                       if k in ('L_sign', 'L_random', 'S_random', 'S_sign', 'length_B_minus_C', 'behaviour')}
                         for g, p_ in pools}
    result['note'] = 'Only decision_P1 and decision_P2 are confirmatory (97.5% intervals). by_group, by_pool and all other fields are secondary or descriptive; by_pool uses 95% intervals.'
    write(a.run / 'summary.json', result)
    show = lambda x: f"{x['point']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    print('P1 (open):', result['decision_P1'], 'L_sign', show(result['P1']['L_sign']), 'L_random', show(result['P1']['L_random']))
    print('P2 (transfer):', result['decision_P2'], 'S_random', show(result['P2']['S_random']), 'S_sign', show(result['P2']['S_sign']))


if __name__ == '__main__':
    main()
