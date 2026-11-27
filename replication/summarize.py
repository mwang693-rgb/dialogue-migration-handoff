"""Summarize one model's replication run (CPU only): H1/H2 decisions plus the secondary analyses."""
import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirmation.common import read, rows, write

_spec = importlib.util.spec_from_file_location('transfer_summarize', ROOT / 'transfer' / 'summarize.py')
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)
MARGIN = 0.05


def decide(s):
    if s['L_sign']['ci'][0] > 0 and s['L_random']['ci'][0] > 0:
        return 'replicates'
    if -MARGIN <= s['L_sign']['ci'][0] and s['L_sign']['ci'][1] <= MARGIN:
        return 'absent'
    return 'inconclusive'


def probe(run, layers, sources):
    import torch
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        LogisticRegression = None
    st = {}
    for s in sources:
        for r in torch.load(run / 'activations' / f'{s}.pt', weights_only=True)['records']:
            st[(s, r['request_type'], r['condition'], r['layer'])] = r['vector'].double().numpy()
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(sources)), 5)
    res = {}
    for rtype in ('ordinary', 'change'):
        res[rtype] = {}
        for L in layers:
            N = np.stack([st[s, rtype, 'B', L] for s in sources])
            F = np.stack([st[s, rtype, 'C', L] for s in sources])
            D = N - F
            dom = lr = 0
            for f in folds:
                te = np.zeros(len(sources), bool)
                te[f] = True
                w = D[~te].mean(0)
                thr = 0.5 * ((N[~te] @ w).mean() + (F[~te] @ w).mean())
                dom += ((N[te] @ w) > thr).sum() + ((F[te] @ w) <= thr).sum()
                if LogisticRegression is not None and L > 0:
                    X = np.vstack([N[~te], F[~te]])
                    y = np.r_[np.ones((~te).sum()), np.zeros((~te).sum())]
                    sc = StandardScaler().fit(X)
                    clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X), y)
                    lr += (clf.predict(sc.transform(np.vstack([N[te], F[te]]))) == np.r_[np.ones(te.sum()), np.zeros(te.sum())]).sum()
            res[rtype][L] = dict(dom_acc=float(dom / (2 * len(sources))), lr_acc=float(lr / (2 * len(sources))) if (LogisticRegression and L > 0) else None)
    return res, st


def direction_checks(st, sources, layers, steer_layer, rule):
    u = lambda v: v / np.linalg.norm(v)
    out = {}
    D = {s: st[s, 'ordinary', 'B', steer_layer] - st[s, 'ordinary', 'C', steer_layer] for s in sources}
    Dc = {s: st[s, 'change', 'B', steer_layer] - st[s, 'change', 'C', steer_layer] for s in sources}
    lor = []
    for r in sorted(set(rule.values())):
        w = np.mean([D[s] for s in sources if rule[s] != r], 0)
        lor += [float(D[s] @ w > 0) for s in sources if rule[s] == r]
    w = np.mean([D[s] for s in sources], 0)
    out['leave_one_rule_out_paired_sign'] = float(np.mean(lor))
    out['ordinary_to_change_paired_sign'] = float(np.mean([Dc[s] @ w > 0 for s in sources]))
    out['cross_format_cos'] = {}
    for L in layers[1:]:
        m = lambda form: np.mean([st[s, 'ordinary', 'B', L] - st[s, 'ordinary', form, L] for s in sources], 0)
        base = m('C')
        out['cross_format_cos'][L] = {f: float(u(m(f)) @ u(base)) for f in ('C_qa', 'C_json')}
    out['norm_mean_dir_steer_layer'] = float(np.linalg.norm(w))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    run = a.run
    if not a.allow_incomplete:
        assert read(run / 'status.json')['phase'] == 'completed'
    env = read(run / 'environment.json')
    layers, steer_layer = env['capture_layers'], env['steer_layer']
    H = {h['source_id']: h for h in rows(run / 'histories.jsonl')}
    sources = sorted(H)
    units = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((run / 'units').glob('*.json'))]
    opens_all = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((run / 'open_units').glob('*.json'))]
    opens = [u for u in opens_all if 'excluded' not in u]
    ordinary = [u for u in units if u['request_type'] == 'ordinary']
    change = [u for u in units if u['request_type'] == 'change']
    res = dict(family=env['family'], steer_layer=steer_layer, capture_layers=layers, admission=read(run / 'admission.json'))
    s1 = _ts.stats(ordinary)
    s2 = _ts.stats(opens) if opens else None
    res['H1_length_v21'] = dict(L_sign=s1['L_sign'], L_random=s1['L_random'], decision=decide(s1))
    res['H2_length_open'] = dict(L_sign=s2['L_sign'], L_random=s2['L_random'], decision=decide(s2)) if s2 else None
    res['ordinary'] = s1
    res['change'] = _ts.stats(change) if change else None
    res['open'] = s2
    res['open_excluded'] = dict(Counter(u['excluded'] for u in opens_all if 'excluded' in u))
    res['open_by_pool'] = {p_: {k: v for k, v in _ts.stats([u for u in opens if u['pool'] == p_], level=95).items()
                                if k in ('L_sign', 'L_random', 'S_random', 'S_sign', 'length_B_minus_C')} for p_ in sorted({u['pool'] for u in opens})}
    base = {s: read(run / 'baseline' / f'{s}.json') for s in sources}
    res['behaviour_compliance'] = {rt: {f: dict(passed=sum(base[s][rt][f]['score']['passed'] for s in sources),
                                                mean_words=float(np.mean([base[s][rt][f]['score']['words'] for s in sources])),
                                                mean_tokens=float(np.mean([base[s][rt][f]['n'] for s in sources])), n=len(sources))
                                         for f in ('B', 'C')} for rt in ('ordinary', 'change')}
    res['steered_compliance'] = {rt: {c: sum(u['generations'][c]['score']['passed'] for u in units if u['request_type'] == rt)
                                      for c in units[0]['generations']} for rt in ('ordinary', 'change')}
    pr, st = probe(run, layers, sources)
    res['probe'] = pr
    res['directions'] = direction_checks(st, sources, layers, steer_layer, {s: H[s]['inst_name'] for s in sources})
    res['note'] = 'H1 and H2 are the registered tests (97.5% intervals). Everything else is secondary or descriptive.'
    write(run / 'summary.json', res)
    sh = lambda x: f"{x['point']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    print(env['family'], 'admitted', res['admission'])
    print('H1 (v2.1-style, ordinary):', res['H1_length_v21']['decision'], 'L_sign', sh(s1['L_sign']), 'L_random', sh(s1['L_random']))
    if s2:
        print('H2 (open):', res['H2_length_open']['decision'], 'L_sign', sh(s2['L_sign']), 'L_random', sh(s2['L_random']), 'excluded', res['open_excluded'])


if __name__ == '__main__':
    main()
