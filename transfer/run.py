"""Registered format-direction transfer run (GPU). See transfer/protocol.json."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'intervention'))
from confirmation.common import ROOT, append, read, rows, sha, write

TF = {'B': ['C', 'C_plus_mean', 'C_minus_mean', 'C_plus_random'], 'C': ['B', 'B_minus_mean']}


def plan(mean, rand):
    return {'B': ('B', None), 'C': ('C', None), 'C_plus_mean': ('C', mean), 'C_minus_mean': ('C', -mean),
            'C_plus_random': ('C', rand), 'B_minus_mean': ('B', -mean)}


def run_unit(model, tok, unit, mean, rand, layer, cap, eos, im_end):
    from steer import Steer, generate, kl_summary, teacher_forced_logprobs
    ids = {'B': unit['B'], 'C': unit['C']}
    conds = plan(mean, rand)
    gens = {}
    for name, (form, v) in conds.items():
        t = time.perf_counter()
        with Steer(model, layer, v) as st:
            g = generate(model, ids[form], eos, cap, st)
        if v is not None:
            assert len(st.calls) == len(g['generated_ids']) and st.calls[0] == (len(ids[form]), len(ids[form]) - 1)
            assert all(c == (1, 0) for c in st.calls[1:])
        last = g['generated_ids'][-1]
        term = 'cap' if g['truncated'] else 'assistant_eos' if last == im_end else 'other_stop'
        gens[name] = dict(form=form, steered=v is not None, text=tok.decode(g['token_ids'], skip_special_tokens=True),
                          termination=term, n=max(1, len(g['token_ids'])), seconds=time.perf_counter() - t, **g)
    for g in gens.values():
        g['exact_vs_B'] = g['generated_ids'] == gens['B']['generated_ids']
        g['exact_vs_C'] = g['generated_ids'] == gens['C']['generated_ids']
    tf = {}
    for ref, members in TF.items():
        reference = gens[ref]['generated_ids']
        with Steer(model, layer, None) as st:
            ref_logp = teacher_forced_logprobs(model, ids[ref], reference, st)
        tf[ref] = {}
        for name in members:
            form, v = conds[name]
            with Steer(model, layer, v) as st:
                logp = teacher_forced_logprobs(model, ids[form], reference, st)
            if v is not None:
                assert st.calls == [(len(ids[form]) + len(reference) - 1, len(ids[form]) - 1)]
            tf[ref][name] = kl_summary(ref_logp, logp, reference)
            del logp
        del ref_logp
    return gens, tf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='folder with units.jsonl and vectors.pt')
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'transfer/protocol.json')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='skip finished units of the same registered run')
    p.add_argument('--smoke', type=int, default=0, help='CPU smoke test on a tiny model: first N units per group, no registration')
    a = p.parse_args()

    cfg = read(a.protocol)
    units_path, vectors_path = a.inputs / 'units.jsonl', a.inputs / 'vectors.pt'
    if a.smoke:
        reg = dict(status='smoke_test_not_registered')
    else:
        reg = read(a.registration)
        assert cfg['status'] == 'reviewed' and reg['status'] == 'registered_before_generation'
        assert sha(a.protocol) == reg['protocol_sha256'], 'protocol changed after registration'
        assert sha(units_path) == reg['units_sha256'] and sha(vectors_path) == reg['vectors_sha256']
        for f, d in reg['code_sha256'].items():
            assert sha(ROOT / f) == d, f'code changed after registration: {f}'
    if a.resume:
        assert a.out.is_dir() and read(a.out / 'registration.json') == reg
        append(a.out / 'restarts.jsonl', dict(utc=datetime.now(timezone.utc).isoformat()))
    else:
        assert not a.out.exists(), 'Use a new run directory'
        (a.out / 'units').mkdir(parents=True)
        write(a.out / 'registration.json', reg)
        write(a.out / 'protocol.json', cfg)
    status = lambda **k: write(a.out / 'status.json', dict(utc=datetime.now(timezone.utc).isoformat(), **k))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steer import Steer
    try:
        units = rows(units_path)
        vec = torch.load(vectors_path, map_location='cpu', weights_only=True)
        assert vec['unit_ids'] == [u['unit_id'] for u in units] and vec['layer'] == cfg['site']['layer']
        if a.smoke:
            per = {}
            units = [u for u in units if per.setdefault(u['group'], []).append(u) or len(per[u['group']]) <= a.smoke]
        status(phase='preflight')
        if not a.resume and not a.smoke:
            files = {f: sha(a.model / f) for f in cfg['pinned_sha256']}
            assert files == cfg['pinned_sha256'], 'model files differ from the pinned snapshot'
        tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
        tail = tok('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
        assert all(u['B'][-len(tail):] == tail and u['C'][-len(tail):] == tail for u in units)
        device = 'cuda' if torch.cuda.is_available() and not a.smoke else 'cpu'
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        torch.manual_seed(cfg['seed'])
        model = AutoModelForCausalLM.from_pretrained(str(a.model), dtype=dtype, attn_implementation='sdpa',
                                                     local_files_only=True).to(device).eval()
        model.requires_grad_(False)
        index = {uid: i for i, uid in enumerate(vec['unit_ids'])}
        mean = vec['mean'] if not a.smoke else vec['mean'][:model.config.hidden_size]
        assert mean.numel() == model.config.hidden_size
        gc = model.generation_config
        eos = {gc.eos_token_id} if isinstance(gc.eos_token_id, int) else set(gc.eos_token_id)
        eos.add(tok.convert_tokens_to_ids('<|im_start|>'))
        im_end = tok.convert_tokens_to_ids('<|im_end|>')
        layer, cap = cfg['site']['layer'], (8 if a.smoke else cfg['cap'])
        if not a.resume:
            check = {}
            for form in ('B', 'C'):
                x = torch.tensor([units[0][form]], device=device)
                with torch.inference_mode():
                    base = model(input_ids=x, logits_to_keep=1).logits
                    with Steer(model, layer, torch.zeros(model.config.hidden_size)) as st:
                        st.start = x.shape[1] - 1
                        zero = model(input_ids=x, logits_to_keep=1).logits
                check[form] = bool(torch.equal(base, zero))
            assert all(check.values()), check
            write(a.out / 'preflight.json', dict(units=len(units), model_sha256_match=not a.smoke, zero_hook_bitwise=check,
                                                 device=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                                                 torch=torch.__version__, transformers=__import__('transformers').__version__,
                                                 eos=sorted(eos), cap=cap))
        start, done = time.perf_counter(), 0
        for u in units:
            path = a.out / 'units' / f"{u['unit_id']}.json"
            if path.exists():
                done += 1
                continue
            status(phase='running', unit_id=u['unit_id'], group=u['group'], units_done=done, units_total=len(units),
                   seconds=time.perf_counter() - start)
            rand = vec['random'][index[u['unit_id']]]
            rand = rand if not a.smoke else rand[:model.config.hidden_size]
            gens, tf = run_unit(model, tok, u, mean, rand, layer, cap, eos, im_end)
            record = dict({k: u[k] for k in ('unit_id', 'pool', 'group', 'origin', 'key', 'history_key', 'id_source')},
                          prompt_tokens={f: len(u[f]) for f in ('B', 'C')}, generations=gens, teacher_forced=tf)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, path)
            done += 1
            if device == 'cuda':
                torch.cuda.empty_cache()
        n = len(list((a.out / 'units').glob('*.json')))
        assert n == len(units)
        status(phase='completed', units=n, seconds=time.perf_counter() - start)
    except Exception:
        status(phase='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
