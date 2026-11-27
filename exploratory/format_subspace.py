"""Exploratory (CPU): decompose the native-minus-pasted format delta into shared,
request-type, history-specific and residual parts, in the style of Bürger et al.
(NeurIPS 2024, "Truth is Universal"), and test which parts the earlier tasks share.

Inputs:
- v2.1 activations: activations.pt (ordinary) and activations_change.pt (change),
  nat/flat at the last prompt token, layers 0-28 in steps of 4.
- intervention v1 units, for per-source behaviour (teacher-forced KL, answer length).
- saved N-P vectors from dialogue-migration-recovery/results (same loader as the
  earlier cross-direction check).

Not a registered analysis; nothing here enters a confirmatory claim.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import rows, write

LAYERS = [4, 8, 12, 16, 20, 24, 28]
RNG = np.random.default_rng(453)


def ld(p):
    return torch.load(p, map_location='cpu', weights_only=True)


def f64(t):
    return t.float().numpy().astype(np.float64)


def unit(v):
    return v / np.linalg.norm(v)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    r = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(2000)])
    return dict(rho=round(r, 3), p_perm=round(float((np.abs(null) >= abs(r)).mean()), 4))


def eta2(x, labels):
    x = np.asarray(x, float)
    groups = defaultdict(list)
    for v, g in zip(x, labels):
        groups[g].append(v)
    ss_b = sum(len(v) * (np.mean(v) - x.mean()) ** 2 for v in groups.values())
    e = ss_b / ((x - x.mean()) ** 2).sum()
    null = []
    for _ in range(2000):
        perm = RNG.permutation(labels)
        gg = defaultdict(list)
        for v, g in zip(x, perm):
            gg[g].append(v)
        null.append(sum(len(v) * (np.mean(v) - x.mean()) ** 2 for v in gg.values()) / ((x - x.mean()) ** 2).sum())
    return dict(eta2=round(float(e), 3), p_perm=round(float((np.array(null) >= e).mean()), 4))


def load_v21(handoff):
    D, meta = {}, {}
    for rtype, name in (('ordinary', 'activations'), ('change', 'activations_change')):
        by = defaultdict(dict)
        for r in ld(f'{handoff}/{name}.pt')['records']:
            by[(int(r['layer']), r['source_id'])][r['condition']] = r
        for (L, s), v in by.items():
            D[(L, s, rtype)] = f64(v['nat']['vector']) - f64(v['flat']['vector'])
            meta[(s, rtype)] = dict(n_nat=int(v['nat']['n_tokens']), n_flat=int(v['flat']['n_tokens']))
    return D, meta


def load_old(R):
    """Format N-P vectors from the earlier exploratory runs (same files as the cross-direction check)."""
    items = []

    def add(exp, kind, L, lab, v):
        items.append((exp, kind, L, lab, f64(v)))
    for k, v in ld(f'{R}/steering_artifacts_v1/vectors.pt')['deltas'].items():
        add('editing(steering_artifacts)', 'cue', 16, k, v)
    for k, v in ld(f'{R}/smoke_v2/vectors.pt')['deltas'].items():
        add('pilot(smoke_v2)', 'cue', 16, k, v)
    for k, v in ld(f'{R}/protected_projection_dev_v2/vectors.pt')['deltas'].items():
        add('editing(projection_v2)', 'cue', 16, k, v)
    for k, v in ld(f'{R}/steering_scope_dev_v1/vectors.pt').items():
        h, L, kind = k.split('|')
        if kind == 'cue':
            add('editing(scope)', 'cue', int(L), h, v)
    for p in sorted(glob.glob(f'{R}/frozen_update_stress_v1/*_vectors.pt')):
        add('update-table', 'cue', 16, os.path.basename(p)[:20], ld(p)['DAll'])
    for p in sorted(glob.glob(f'{R}/open_text_fidelity_v1/*_vectors.pt')):
        x = ld(p)
        b = os.path.basename(p).split('_')[0]
        add('wildchat(open_text)', 'cue', 16, b, x['cue'])
        add('wildchat(open_text)', 'query', 16, b, x['query'])
    for p in sorted(glob.glob(f'{R}/response_frame_dev_v1/approved-*.pt')):
        x = ld(p)
        b = os.path.basename(p)[:-3]
        add('approved-plan(frame)', 'cue' if b.endswith('cue') else 'query', 16, b, x['delta'])
    for k, x in ld(f'{R}/mti_static_steering_dev_v1/vectors.pt').items():
        add('mti(static)', 'cue', 16, k, x['CueD'])
        add('mti(static)', 'query', 16, k, x['QueryD'])
    for h, d in ld(f'{R}/migration_exploration_methods_v1/vectors.pt')['states'].items():
        for L, s in d.items():
            add('mti+approved(explore)', 'cue', {15: 16, 19: 20}[int(L)], h, s['N'] - s['P'])
    for name, key in (('migration_terminal_bound_v1', 'delta'), ('migration_cue_terminal_v1', 'cue_delta')):
        for h, s in ld(f'{R}/{name}/states.pt').items():
            add(f"approved-plan({name.split('_')[1]})", 'query' if 'terminal_bound' in name else 'cue', 28, h, s[key])
    for h, s in ld(f'{R}/policy_step_intervention_v1/states.pt').items():
        add('policy-step', 'query', 28, h, s['delta'])
    for h, s in ld(f'{R}/existing_privacy_controls_v1/states.pt').items():
        add('privacy(MTI)', 'query', 28, h, s['delta'])
    x = ld(f'{R}/migration_gap_probe_v1/vectors.pt')
    add('gap-probe', 'raw', 20, 'raw', x['raw'])
    return items


def decompose(D, sources):
    """D[s, r] = g + p_r q + a_s + e_sr with p_ordinary = +1, p_change = -1 (two-way layout in vector space)."""
    X = np.stack([[D[s, 'ordinary'], D[s, 'change']] for s in sources])  # (S, 2, d)
    g = X.mean((0, 1))
    q = (X[:, 0].mean(0) - X[:, 1].mean(0)) / 2
    a = X.mean(1) - g  # source effects, (S, d)
    fit = g + a[:, None, :] + np.stack([q, -q])[None]
    e = X - fit
    ss = lambda Y: float((Y ** 2).sum())
    total = ss(X)
    parts = dict(shared=ss(np.broadcast_to(g, X.shape)), request=ss(np.broadcast_to(np.stack([q, -q])[None], X.shape)),
                 history=ss(np.broadcast_to(a[:, None, :], X.shape)), residual=ss(e))
    return X, g, q, a, e, {k: round(v / total, 4) for k, v in parts.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff', type=Path, required=True)
    p.add_argument('--v21', type=Path, required=True, help='extracted confirmation-v2_1 run folder')
    p.add_argument('--v1-units', type=Path, required=True, help='intervention v1 units folder')
    p.add_argument('--recovery-results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a_ = p.parse_args()

    D, meta = load_v21(a_.handoff)
    sources = sorted({s for (_, s, _) in D})
    rule = {h['source_id']: h['inst_name'] for h in rows(a_.v21 / 'histories.jsonl')}
    units = {(json.loads(f.read_text(encoding='utf-8'))['source_id'], json.loads(f.read_text(encoding='utf-8'))['request_type']):
             json.loads(f.read_text(encoding='utf-8')) for f in sorted(a_.v1_units.glob('*.json'))}
    out = dict(note='exploratory; not registered', layers={})

    print('Per-layer decomposition of D = nat - flat (share of total squared norm):')
    print(f"{'L':>3} {'shared g':>9} {'request q':>10} {'history a_s':>12} {'residual':>9} {'|q|/|g|':>8} {'perm p(q)':>10} "
          f"{'cos(q halves)':>14} {'cos(D_o,D_c)':>13} {'mean cos(D_so,D_sc)':>20}")
    for L in LAYERS:
        DL = {(s, r): D[(L, s, r)] for s in sources for r in ('ordinary', 'change')}
        X, g, q, a, e, parts = decompose(DL, sources)
        # paired permutation: swap request labels within source
        qn = np.linalg.norm(q)
        null = []
        for _ in range(2000):
            flip = RNG.integers(0, 2, len(sources)) * 2 - 1
            null.append(np.linalg.norm(((X[:, 0] - X[:, 1]) * flip[:, None]).mean(0) / 2))
        p_q = float((np.array(null) >= qn).mean())
        half = RNG.permutation(len(sources))
        h1, h2 = half[:60], half[60:]
        q1 = (X[h1, 0] - X[h1, 1]).mean(0) / 2
        q2 = (X[h2, 0] - X[h2, 1]).mean(0) / 2
        Do, Dc = X[:, 0].mean(0), X[:, 1].mean(0)
        within = float(np.mean([unit(X[i, 0]) @ unit(X[i, 1]) for i in range(len(sources))]))
        row = dict(parts=parts, q_over_g=round(qn / np.linalg.norm(g), 3), perm_p_q=p_q, cos_q_halves=round(float(unit(q1) @ unit(q2)), 3),
                   cos_Do_Dc=round(float(unit(Do) @ unit(Dc)), 3), mean_cos_same_history=round(within, 3))
        out['layers'][L] = row
        print(f"{L:>3} {parts['shared']:>9.3f} {parts['request']:>10.4f} {parts['history']:>12.3f} {parts['residual']:>9.3f} {row['q_over_g']:>8.3f} "
              f"{p_q:>10.4f} {row['cos_q_halves']:>14.3f} {row['cos_Do_Dc']:>13.3f} {within:>20.3f}")

    # ---- layer 16 detail: what structures the history-specific part a_s?
    L = 16
    DL = {(s, r): D[(L, s, r)] for s in sources for r in ('ordinary', 'change')}
    X, g, q, a, e, parts = decompose(DL, sources)
    U, S, Vt = np.linalg.svd(a, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    gu = unit(g)
    feats = dict(
        rule=[rule[s] for s in sources],
        dn_tokens=[np.mean([meta[s, r]['n_nat'] - meta[s, r]['n_flat'] for r in ('ordinary', 'change')]) for s in sources],
        prompt_tokens=[np.mean([meta[s, r]['n_flat'] for r in ('ordinary', 'change')]) for s in sources],
        kl_B_C=[np.mean([units[s, r]['teacher_forced']['B']['C']['kl_mean'] for r in ('ordinary', 'change')]) for s in sources],
        words_B_minus_C=[np.mean([units[s, r]['generations']['B']['score']['words'] - units[s, r]['generations']['C']['score']['words']
                                  for r in ('ordinary', 'change')]) for s in sources],
        gain_plus_mean=[np.mean([units[s, r]['teacher_forced']['B']['C']['kl_mean'] - units[s, r]['teacher_forced']['B']['C_plus_mean']['kl_mean']
                                 for r in ('ordinary', 'change')]) for s in sources])
    scores = {'proj_on_g': a @ gu, 'norm_D': np.linalg.norm(X, axis=2).mean(1)}
    for k in range(3):
        scores[f'PC{k + 1}'] = U[:, k] * S[k]
    detail = dict(history_effect_pc_variance=[round(float(v), 4) for v in var[:8]],
                  cos_history_pcs_with_g=[round(float(unit(Vt[k]) @ gu), 3) for k in range(5)],
                  cos_q_with_g=round(float(unit(q) @ gu), 3), correlations={})
    print(f'\nL16 history-specific part a_s: variance share of top PCs {detail["history_effect_pc_variance"][:5]}')
    print(f'  cos(PC_k, shared g) = {detail["cos_history_pcs_with_g"]};  cos(request q, g) = {detail["cos_q_with_g"]}')
    print(f"  {'score':<10}" + ''.join(f'{k:>22}' for k in feats))
    for name, sc in scores.items():
        cells = {}
        for k, v in feats.items():
            cells[k] = eta2(sc, v) if k == 'rule' else spearman(sc, np.asarray(v, float))
        detail['correlations'][name] = cells
        print(f'  {name:<10}' + ''.join(f"{(('eta2 ' + str(c['eta2'])) if 'eta2' in c else ('rho ' + str(c['rho']))) + ' p' + str(c['p_perm']):>22}"
                                         for c in cells.values()))
    # per-rule means of the shared projection
    by_rule = defaultdict(list)
    for s, v in zip(sources, scores['proj_on_g']):
        by_rule[rule[s]].append(v)
    detail['proj_on_g_by_rule'] = {k: round(float(np.mean(v)), 3) for k, v in by_rule.items()}
    print('  mean projection of a_s on g by rule:', detail['proj_on_g_by_rule'])
    out['L16_detail'] = detail

    # ---- earlier tasks: which v2.1 component do they share?
    items = load_old(str(a_.recovery_results))
    print('\nEarlier tasks vs v2.1 components (cosine of each old N-P vector; mean over vectors in the group):')
    print(f"{'experiment':<28}{'kind':<6}{'L':>3}{'n':>4}{'ordinary-only D_o':>18}{'shared g':>10}{'request q':>10}{'span top-k capture (k=1/3/10)':>32}{'random':>9}")
    comp = {}
    for Lx in LAYERS:
        DL = {(s, r): D[(Lx, s, r)] for s in sources for r in ('ordinary', 'change')}
        X, g, q, a, e, _ = decompose(DL, sources)
        _, _, Vt = np.linalg.svd(a, full_matrices=False)
        comp[Lx] = dict(Do=unit(X[:, 0].mean(0)), g=unit(g), q=unit(q), basis=Vt)
    groups = defaultdict(list)
    for exp, kind, Lx, lab, v in items:
        groups[(exp, kind, Lx)].append(v)
    old_rows = []
    for (exp, kind, Lx), vs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        c = comp[Lx]
        V = np.stack([unit(v) for v in vs])
        caps = []
        for k in (1, 3, 10):
            B = np.linalg.qr(np.stack([c['g'], c['q']] + list(c['basis'][:k])).T)[0]
            caps.append(float(((V @ B) ** 2).sum(1).mean()))
        rand = (2 + 10) / V.shape[1]
        r = dict(experiment=exp, kind=kind, layer=Lx, n=len(vs), cos_Do=float((V @ c['Do']).mean()), cos_g=float((V @ c['g']).mean()),
                 cos_q=float((V @ c['q']).mean()), capture_g_q_plus_k=caps, random_capture_k10=rand)
        old_rows.append(r)
        print(f"{exp:<28}{kind:<6}{Lx:>3}{len(vs):>4}{r['cos_Do']:>18.3f}{r['cos_g']:>10.3f}{r['cos_q']:>10.3f}"
              f"{'  ' + ' / '.join(f'{x:.3f}' for x in caps):>32}{rand:>9.4f}")
    allc = lambda key: float(np.mean([r[key] for r in old_rows]))
    print(f"\nmean over groups: cos with D_o {allc('cos_Do'):.3f}, with balanced g {allc('cos_g'):.3f}, with q {allc('cos_q'):.3f}")
    out['earlier_tasks'] = old_rows
    write(a_.out, out)


if __name__ == '__main__':
    main()
