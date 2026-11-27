"""Summarize the registered shallow-depth (k = 2) follow-up of one model's run (CPU only).

Uses the same decision rule as replication/summarize.py (decide) and the same statistics as
transfer/summarize.py (stats). Also puts the k = 2 and k = 4 numbers side by side on the same units.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirmation.common import read, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ts = load('transfer_summarize', 'transfer/summarize.py')
_rs = load('replication_summarize', 'replication/summarize.py')
KEYS = ('L_sign', 'L_random', 'S_random', 'S_sign', 'length_B_minus_C', 'length_Cplus_minus_C',
        'length_Cminus_minus_C', 'length_Crandom_minus_C', 'length_Bminus_minus_B', 'S_random_first', 'S_sign_first', 'gap_closed')


def units(d):
    return [json.loads(f.read_text(encoding='utf-8')) for f in sorted(d.glob('*.json'))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='replication run directory that contains shallow/')
    a = p.parse_args()
    out = a.run / 'shallow'
    assert read(out / 'status.json')['phase'] == 'completed'
    env = read(out / 'environment.json')
    k2 = units(out / 'units')
    k2_open = units(out / 'open_units')
    k4 = {u['unit_id']: u for u in units(a.run / 'units')}
    k4_open = {u['unit_id']: u for u in units(a.run / 'open_units') if 'excluded' not in u}
    res = dict(family=env['family'], depth=env['depth'], k=env['k'])
    groups = dict(ordinary=[u for u in k2 if u['request_type'] == 'ordinary'],
                  change=[u for u in k2 if u['request_type'] == 'change'], open=k2_open)
    for g, us in groups.items():
        s = _ts.stats(us)
        ref = [k4[u['unit_id']] for u in us] if g != 'open' else [k4_open[u['unit_id']] for u in us]
        s4 = _ts.stats(ref)
        res[g] = dict(k2={x: s[x] for x in KEYS if x in s}, k4_same_units={x: s4[x] for x in KEYS if x in s4},
                      behaviour=s.get('behaviour'), n_units=len(us))
    res['H1s_length_v21'] = dict(L_sign=res['ordinary']['k2']['L_sign'], L_random=res['ordinary']['k2']['L_random'],
                                 decision=_rs.decide(res['ordinary']['k2']))
    res['H2s_length_open'] = dict(L_sign=res['open']['k2']['L_sign'], L_random=res['open']['k2']['L_random'],
                                  decision=_rs.decide(res['open']['k2']))
    res['note'] = 'H1s and H2s are the registered k = 2 follow-up tests (97.5% intervals); k = 4 results stand as the primary replication results.'
    write(out / 'summary.json', res)
    sh = lambda x: f"{x['point']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    print(env['family'], 'k=2 depth', env['depth'])
    for h, g in (('H1s', 'ordinary'), ('H2s', 'open')):
        d = res[f'{h}_length_v21' if g == 'ordinary' else f'{h}_length_open']
        print(f'{h} ({g}):', d['decision'], 'L_sign', sh(d['L_sign']), 'L_random', sh(d['L_random']),
              '| k=4 same units L_sign', sh(res[g]['k4_same_units']['L_sign']))
    print('change k=2 L_sign', sh(res['change']['k2']['L_sign']))


if __name__ == '__main__':
    main()
