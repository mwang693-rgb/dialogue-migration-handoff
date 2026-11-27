"""Registered length-direction control run (GPU). See length_control/protocol.json."""
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

TF_B = ['C', 'C_plus_d', 'C_plus_dperp', 'C_minus_dperp', 'C_plus_dpar', 'C_plus_len']
TAIL_TEXT = '<|im_end|>\n<|im_start|>assistant\n'


def with_instruction(tok, ids, text):
    """Insert the instruction's tokens right before the final <|im_end|> of the last user turn."""
    tail = tok(TAIL_TEXT, add_special_tokens=False)['input_ids']
    assert ids[-len(tail):] == tail
    return ids[:-len(tail)] + tok(text, add_special_tokens=False)['input_ids'] + tail


def last_states(model, ids, layers):
    import torch
    states, handles = {}, []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            states[layer] = h[0, -1].detach().double().cpu().clone()
        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return states


def length_direction(model, tok, units, layers, texts):
    sums, n = {L: 0 for L in layers}, 0
    for u in units:
        for form in ('B', 'C'):
            hd = last_states(model, with_instruction(tok, u[form], texts['detailed']), layers)
            hb = last_states(model, with_instruction(tok, u[form], texts['brief']), layers)
            for L in layers:
                sums[L] = sums[L] + (hd[L] - hb[L])
            n += 1
    return {L: (sums[L] / n).float() for L in layers}, n


def vectors(d, l):
    lu = l / l.norm()
    d_par = (d @ lu) * lu
    return dict(d=d, d_par=d_par, d_perp=d - d_par, len=lu * d.norm())


def plan(v):
    return {'B': ('B', None), 'C': ('C', None), 'C_plus_d': ('C', v['d']), 'C_plus_dperp': ('C', v['d_perp']),
            'C_minus_dperp': ('C', -v['d_perp']), 'C_plus_dpar': ('C', v['d_par']), 'C_plus_len': ('C', v['len'])}


def run_unit(model, tok, unit, v, layer, cap, eos, im_end):
    from steer import Steer, generate, kl_summary, teacher_forced_logprobs
    ids = {'B': unit['B'], 'C': unit['C']}
    conds = plan(v)
    gens = {}
    for name, (form, vec) in conds.items():
        t = time.perf_counter()
        with Steer(model, layer, vec) as st:
            g = generate(model, ids[form], eos, cap, st)
        if vec is not None:
            assert len(st.calls) == len(g['generated_ids']) and st.calls[0] == (len(ids[form]), len(ids[form]) - 1)
        term = 'cap' if g['truncated'] else 'assistant_eos' if g['generated_ids'][-1] == im_end else 'other_stop'
        gens[name] = dict(form=form, steered=vec is not None, text=tok.decode(g['token_ids'], skip_special_tokens=True),
                          termination=term, n=max(1, len(g['token_ids'])), seconds=time.perf_counter() - t, **g)
    reference = gens['B']['generated_ids']
    with Steer(model, layer, None) as st:
        ref_logp = teacher_forced_logprobs(model, ids['B'], reference, st)
    tf = {'B': {}}
    for name in TF_B:
        form, vec = conds[name]
        with Steer(model, layer, vec) as st:
            logp = teacher_forced_logprobs(model, ids[form], reference, st)
        tf['B'][name] = kl_summary(ref_logp, logp, reference)
        del logp
    return gens, tf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transfer-inputs', type=Path, required=True, help='folder with transfer v1 units.jsonl')
    p.add_argument('--format-dirs', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'length_control/protocol.json')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', type=int, default=0, help='CPU smoke test on a tiny model: N units per role, no registration')
    a = p.parse_args()
    cfg = read(a.protocol)
    units_path = a.transfer_inputs / 'units.jsonl'
    if a.smoke:
        reg = dict(status='smoke_test_not_registered')
    else:
        reg = read(a.registration)
        assert cfg['status'] == 'reviewed' and reg['status'] == 'registered_before_generation'
        assert sha(a.protocol) == reg['protocol_sha256'] and sha(units_path) == reg['units_sha256']
        assert sha(a.format_dirs) == reg['format_dirs_sha256']
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
        test = [u for u in units if u['group'] == 'open']
        source = [u for u in units if u['group'] != 'open']
        if a.smoke:
            test, source = test[:a.smoke], source[:a.smoke]
        fd = torch.load(a.format_dirs, map_location='cpu', weights_only=True)
        layers = fd['layers']
        status(phase='preflight')
        if not a.smoke and not a.resume:
            pinned = read(ROOT / 'transfer/protocol.json')['pinned_sha256']
            assert {f: sha(a.model / f) for f in pinned} == pinned, 'model files differ from the pinned snapshot'
        tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
        device = 'cuda' if torch.cuda.is_available() and not a.smoke else 'cpu'
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(str(a.model), dtype=dtype, attn_implementation='sdpa',
                                                     local_files_only=True).to(device).eval()
        model.requires_grad_(False)
        hs = model.config.hidden_size
        d = {L: fd['d'][L][:hs] for L in layers}
        texts = cfg['length_direction']['instructions']
        ld_path = a.out / 'length_direction.pt'
        if ld_path.exists():
            ldir = torch.load(ld_path, map_location='cpu', weights_only=True)['l']
        else:
            status(phase='length_direction', units=len(source))
            ldir, n = length_direction(model, tok, source, layers, texts)
            torch.save(dict(l=ldir, n_pairs=n, layers=layers), ld_path)
            cos = {L: float(d[L] @ ldir[L] / (d[L].norm() * ldir[L].norm())) for L in layers}
            write(a.out / 'length_direction.json', dict(n_pairs=n, cos_d_l=cos, l_norm={L: float(ldir[L].norm()) for L in layers},
                                                        d_norm={L: float(d[L].norm()) for L in layers}))
        layer = 16
        v = vectors(d[layer], ldir[layer])
        write(a.out / 'vectors_summary.json', {k: float(x.norm()) for k, x in v.items()} | dict(
            cos_d_l=float(v['d'] @ v['len'] / (v['d'].norm() * v['len'].norm()))))
        gc = model.generation_config
        eos = {gc.eos_token_id} if isinstance(gc.eos_token_id, int) else set(gc.eos_token_id)
        eos.add(tok.convert_tokens_to_ids('<|im_start|>'))
        im_end = tok.convert_tokens_to_ids('<|im_end|>')
        cap = 8 if a.smoke else cfg['cap']
        if not a.resume:
            x = torch.tensor([test[0]['C']], device=device)
            with torch.inference_mode():
                base = model(input_ids=x, logits_to_keep=1).logits
                with Steer(model, layer, torch.zeros(hs)) as st:
                    st.start = x.shape[1] - 1
                    zero = model(input_ids=x, logits_to_keep=1).logits
            assert torch.equal(base, zero)
            write(a.out / 'preflight.json', dict(test_units=len(test), direction_units=len(source), zero_hook_bitwise=True,
                                                 device=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu', cap=cap))
        start, done = time.perf_counter(), 0
        for u in test:
            path = a.out / 'units' / f"{u['unit_id']}.json"
            if path.exists():
                done += 1
                continue
            status(phase='running', unit_id=u['unit_id'], units_done=done, units_total=len(test), seconds=time.perf_counter() - start)
            gens, tf = run_unit(model, tok, u, v, layer, cap, eos, im_end)
            rec = dict({k: u[k] for k in ('unit_id', 'pool', 'group', 'origin', 'key', 'history_key')}, generations=gens, teacher_forced=tf)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, path)
            done += 1
            if device == 'cuda':
                torch.cuda.empty_cache()
        assert len(list((a.out / 'units').glob('*.json'))) == len(test)
        status(phase='completed', units=len(test), seconds=time.perf_counter() - start)
    except Exception:
        status(phase='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
