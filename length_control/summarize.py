"""Summarize a finished length-direction control run (CPU only)."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from confirmation.common import read, write

_spec = importlib.util.spec_from_file_location('transfer_summarize', ROOT / 'transfer' / 'summarize.py')
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)
cluster_boot, idx = _ts.cluster_boot, _ts.idx
MARGIN = 0.05


def logn(u, c):
    return np.log(u['generations'][c]['n'])


def kl(u, c):
    return u['teacher_forced']['B'][c]['kl_mean']


def stats(units, level=95):
    cl = [u['history_key'] for u in units]
    b = lambda f: cluster_boot([f(u) for u in units], cl, level=level)
    return dict(
        M=b(lambda u: logn(u, 'C_plus_len') - logn(u, 'C')),
        L_perp=b(lambda u: logn(u, 'C_plus_dperp') - logn(u, 'C_minus_dperp')),
        S_len=b(lambda u: idx(kl(u, 'C_plus_len'), kl(u, 'C_plus_d'))),
        shift_d=b(lambda u: logn(u, 'C_plus_d') - logn(u, 'C')),
        shift_dperp=b(lambda u: logn(u, 'C_plus_dperp') - logn(u, 'C')),
        shift_dpar=b(lambda u: logn(u, 'C_plus_dpar') - logn(u, 'C')),
        length_B_minus_C=b(lambda u: logn(u, 'B') - logn(u, 'C')),
        S_sign_perp=b(lambda u: idx(kl(u, 'C_minus_dperp'), kl(u, 'C_plus_dperp'))),
        S_len_vs_perp=b(lambda u: idx(kl(u, 'C_plus_len'), kl(u, 'C_plus_dperp'))),
        mean_kl_along_B={c: float(np.mean([kl(u, c) for u in units])) for c in units[0]['teacher_forced']['B']},
        mean_tokens={c: float(np.mean([u['generations'][c]['n'] for u in units])) for c in units[0]['generations']},
        cap_hits={c: sum(u['generations'][c]['termination'] == 'cap' for u in units) for c in units[0]['generations']})


def decide(s):
    if s['M']['ci'][0] <= 0:
        return 'invalid_length_control'
    if s['L_perp']['ci'][0] > 0 and s['S_len']['ci'][0] > 0:
        return 'not_just_length'
    if -MARGIN <= s['L_perp']['ci'][0] and s['L_perp']['ci'][1] <= MARGIN and s['S_len']['ci'][1] <= MARGIN:
        return 'mostly_length'
    return 'mixed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    if not a.allow_incomplete:
        assert read(a.run / 'status.json')['phase'] == 'completed'
    units = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((a.run / 'units').glob('*.json'))]
    s = stats(units)
    s['decision'] = decide(s)
    s['length_direction'] = read(a.run / 'length_direction.json')
    s['vectors'] = read(a.run / 'vectors_summary.json')
    s['by_pool'] = {p_: {k: v for k, v in stats([u for u in units if u['pool'] == p_]).items() if k in ('M', 'L_perp', 'S_len', 'shift_d', 'shift_dperp')}
                    for p_ in sorted({u['pool'] for u in units})}
    s['note'] = 'Only decision (M, L_perp, S_len) is the registered test; everything else is descriptive.'
    write(a.run / 'summary.json', s)
    sh = lambda x: f"{x['point']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    print('decision:', s['decision'], '| M', sh(s['M']), '| L_perp', sh(s['L_perp']), '| S_len', sh(s['S_len']))
    print('cos(d_L, l_L):', {L: round(c, 3) for L, c in s['length_direction']['cos_d_l'].items()})


if __name__ == '__main__':
    main()
