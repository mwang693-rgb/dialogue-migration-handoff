"""Registered shallow-depth follow-up (GPU). See replication/protocol_shallow.json.

Reuses a finished replication run (its histories, baseline B/C answers, open-unit histories and
B/C answers, and fold assignment) and repeats only the steered conditions at depth k = 2.
"""
import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'replication'))
sys.path.insert(0, str(ROOT / 'intervention'))
from confirmation.common import append, read, request_text, rows, sha, write
from engine import FAMILIES, Engine, depth_layers

_spec = importlib.util.spec_from_file_location('replication_run', ROOT / 'replication' / 'run.py')
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def strip(g):
    return {k: v for k, v in g.items() if k not in ('exact_vs_B', 'exact_vs_C', 'prompt_tokens')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=sorted(FAMILIES), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='finished replication run directory')
    p.add_argument('--v21', type=Path, required=True)
    p.add_argument('--open-units', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'replication/protocol_shallow.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', type=int, default=0)
    a = p.parse_args()
    cfg = read(a.protocol)
    if a.smoke:
        reg = dict(status='smoke_test_not_registered')
    else:
        reg = read(a.registration)
        assert cfg['status'] == 'reviewed' and reg['status'] == 'registered_before_generation'
        assert sha(a.protocol) == reg['protocol_sha256'] and sha(a.open_units) == reg['open_units_sha256']
        for f, d in reg['code_sha256'].items():
            assert sha(ROOT / f) == d, f'code changed after registration: {f}'
        assert read(a.run / 'status.json')['phase'] == 'completed', 'the replication run must be complete'
    out = a.run / 'shallow'
    if a.resume:
        assert out.is_dir() and read(out / 'registration.json') == reg
        append(out / 'restarts.jsonl', dict(utc=R.now()))
    else:
        assert not out.exists()
        (out / 'units').mkdir(parents=True)
        (out / 'open_units').mkdir()
        write(out / 'registration.json', reg)
        write(out / 'protocol.json', cfg)
    status = lambda **k: write(out / 'status.json', dict(utc=R.now(), family=a.family, **k))
    import torch
    try:
        status(phase='loading')
        e = Engine(a.model, a.family)
        layers = depth_layers(e.n_layers)
        depth = layers[cfg['k']]
        env = read(a.run / 'environment.json')
        assert env['capture_layers'] == layers and env['family'] == a.family
        cap = 8 if a.smoke else cfg['cap']
        write(out / 'environment.json', dict(family=a.family, depth=depth, k=cfg['k'], capture_layers=layers, cap=cap))
        v21cfg = read(a.v21 / 'protocol.json')
        H = {h['source_id']: h for h in rows(a.run / 'histories.jsonl')}
        sources = sorted(H)
        dirs = torch.load(a.run / 'directions.pt', weights_only=True)
        fold = dirs['folds']
        D = {}
        for s in sources:
            st = {(r['request_type'], r['condition'], r['layer']): r['vector'].double()
                  for r in torch.load(a.run / 'activations' / f'{s}.pt', weights_only=True)['records']}
            D[s] = st['ordinary', 'B', depth] - st['ordinary', 'C', depth]
        vec = {}
        for i, s in enumerate(sources):
            others = [x for x in sources if fold[x] != fold[s]] or [x for x in sources if x != s] or [s]
            m = torch.stack([D[x] for x in others]).mean(0)
            g = torch.Generator().manual_seed(cfg['seed_v21'] + i)
            r = torch.randn(m.numel(), generator=g, dtype=torch.float64)
            vec[s] = dict(mean=m.float(), random=(r / r.norm() * m.norm()).float())
        full = torch.stack([D[s] for s in sources]).mean(0).float()
        torch.save(dict(depth=depth, vectors=vec, full_mean=full), out / 'directions_shallow.pt')

        t0 = time.perf_counter()
        for s in sources:
            for rtype in ('ordinary', 'change'):
                path = out / 'units' / f'{s}_{rtype}.json'
                if path.exists():
                    continue
                status(phase='steering', source_id=s, request_type=rtype, done=len(list((out / 'units').glob('*.json'))),
                       total=2 * len(sources), seconds=time.perf_counter() - t0)
                h = H[s]
                q = request_text(h['question'], rtype, 'B', v21cfg, h)
                ids = {f: e.prompt(h['history'], q, f) for f in ('B', 'C')}
                base = read(a.run / 'baseline' / f'{s}.json')[rtype]
                assert all(len(ids[f]) == base[f]['prompt_tokens'] for f in ids), 'prompt differs from the registered run'
                gens, tf = R.steered_unit(e, ids, {f: strip(base[f]) for f in ids}, vec[s], depth, cap, h, rtype)
                R.save_json(path, dict(unit_id=f'{s}_{rtype}', source_id=s, history_key=s, group='v21', request_type=rtype, inst_name=h['inst_name'],
                                       depth=depth, prompt_tokens={f: len(ids[f]) for f in ids}, generations=gens, teacher_forced=tf))
                if e.device == 'cuda':
                    torch.cuda.empty_cache()

        spec = {u['unit_id']: u for u in rows(a.open_units)}
        regen = {r['history_key']: r for r in rows(a.run / 'open_histories.jsonl')}
        done_open = sorted((a.run / 'open_units').glob('*.json'))
        for i, f in enumerate(done_open):
            reg_unit = json.loads(f.read_text(encoding='utf-8'))
            path = out / 'open_units' / f.name
            if path.exists() or 'excluded' in reg_unit:
                continue
            status(phase='open', unit_id=reg_unit['unit_id'], done=len(list((out / 'open_units').glob('*.json'))), total=len(done_open))
            u = spec[reg_unit['unit_id']]
            hist = regen[u['history_key']]['history']
            ids = {k: e.prompt(hist, u['query'], k, u['serializer']) for k in ('B', 'C')}
            assert {k: len(v) for k, v in ids.items()} == reg_unit['prompt_tokens'], 'prompt differs from the registered run'
            g = torch.Generator().manual_seed(cfg['seed_open'] + i)
            r = torch.randn(full.numel(), generator=g, dtype=torch.float64)
            v = dict(mean=full, random=(r / r.norm() * full.double().norm()).float())
            gens, tf = R.steered_unit(e, ids, {k: strip(reg_unit['generations'][k]) for k in ('B', 'C')}, v, depth, cap)
            R.save_json(path, dict(unit_id=u['unit_id'], pool=u['pool'], group='open', history_key=u['history_key'], key=u['key'], depth=depth,
                                   prompt_tokens={k: len(ids[k]) for k in ids}, generations=gens, teacher_forced=tf))
            if e.device == 'cuda':
                torch.cuda.empty_cache()
        status(phase='completed', units=len(list((out / 'units').glob('*.json'))), open_units=len(list((out / 'open_units').glob('*.json'))))
    except Exception:
        status(phase='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
