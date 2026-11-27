"""Registered format-direction intervention run (GPU). See intervention/protocol.json."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, append, check_rule, read, request_text, rows, sha, write

SYSTEM = {'role': 'system', 'content': 'You are a helpful assistant.'}
CAPTURE = [4, 8, 12, 16, 20, 24, 28]


def messages(history, query, form):
    """B/C match reference_code/migration_exploration_core_v1.messages; C_qa/C_json are cross-format probes."""
    if form == 'B':
        return [SYSTEM] + history + [{'role': 'user', 'content': query}]
    if form in ('C', 'C_qa'):
        label = {'C': ('User', 'Assistant'), 'C_qa': ('Q', 'A')}[form]
        body = '\n\n'.join(label[m['role'] != 'user'] + ':\n' + m['content'] for m in history)
        body += '\n\n--- Current user request ---\n' + query
    elif form == 'C_json':
        body = json.dumps([{'role': m['role'], 'content': m['content']} for m in history], ensure_ascii=False, indent=1)
        body += '\n\nCurrent user request:\n' + query
    else:
        raise ValueError(form)
    return [SYSTEM, {'role': 'user', 'content': body}]


def encode(tok, history, query, form):
    rendered = tok.apply_chat_template(messages(history, query, form), tokenize=False, add_generation_prompt=True)
    return tok(rendered, add_special_tokens=False)['input_ids']


def plan(v, rtype):
    return {'B': ('B', None), 'C': ('C', None), 'C_plus_mean': ('C', v['mean']), 'C_plus_random': ('C', v['random']),
            'C_minus_mean': ('C', -v['mean']), 'C_plus_own': ('C', v['own'][rtype]), 'C_plus_wrong': ('C', v['wrong'][rtype]),
            'B_minus_mean': ('B', -v['mean'])}


def capture_states(model, ids, layers):
    import torch
    states, handles = {}, []
    for layer in layers:
        def hook(module, args, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            states[layer] = h[0, -1].detach().float().cpu().clone()
        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v21', type=Path, required=True, help='extracted confirmation-v2_1 run folder')
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT / 'intervention/protocol.json')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='skip finished units of the same registered run')
    a = p.parse_args()

    cfg, reg = read(a.protocol), read(a.registration)
    assert cfg['status'] == 'reviewed' and reg['status'] == 'registered_before_generation'
    assert sha(a.protocol) == reg['protocol_sha256'] and sha(a.directions) == reg['directions_sha256']
    for f, d in reg['code_sha256'].items():
        assert sha(ROOT / f) == d, f'code changed after registration: {f}'
    for f, d in reg['v21_sha256'].items():
        assert sha(a.v21 / f) == d, f'v2.1 input differs: {f}'
    if a.resume:
        assert a.out.is_dir() and read(a.out / 'registration.json') == reg
        append(a.out / 'restarts.jsonl', dict(utc=datetime.now(timezone.utc).isoformat()))
    else:
        assert not a.out.exists(), 'Use a new run directory'
        (a.out / 'units').mkdir(parents=True)
        (a.out / 'cross_format').mkdir()
        write(a.out / 'registration.json', reg)
        write(a.out / 'protocol.json', cfg)
    status = lambda **k: write(a.out / 'status.json', dict(utc=datetime.now(timezone.utc).isoformat(), **k))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / 'intervention'))
    from steer import Steer, generate, kl_summary, teacher_forced_logprobs

    try:
        v21_cfg = read(a.v21 / 'protocol.json')
        histories = {h['source_id']: h for h in rows(a.v21 / 'histories.jsonl')}
        old = {(r['source_id'], r['request_type'], r['condition']): r for r in rows(a.v21 / 'outputs.jsonl')}
        directions = torch.load(a.directions, map_location='cpu', weights_only=True)
        sources = directions['sources']
        assert sorted(histories) == sources and len(sources) == cfg['sources']

        status(phase='preflight')
        if not a.resume:
            files = {f: sha(a.model / f) for f in cfg['pinned_sha256']}
            assert files == cfg['pinned_sha256'], 'model files differ from the pinned snapshot'
        tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
        prompts = {}
        for s in sources:
            h = histories[s]
            for rtype in cfg['request_types']:
                q = request_text(h['question'], rtype, 'B', v21_cfg, h)
                assert q == request_text(h['question'], rtype, 'C', v21_cfg, h)
                ids = {form: encode(tok, h['history'], q, form) for form in ('B', 'C', 'C_qa', 'C_json')}
                assert ids['B'] == old[s, rtype, 'B']['input_ids'] and ids['C'] == old[s, rtype, 'C']['input_ids'], (s, rtype)
                assert len({ids[f][-1] for f in ids}) == 1
                prompts[s, rtype] = ids
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(str(a.model), dtype=torch.bfloat16, attn_implementation='sdpa',
                                                     local_files_only=True).to('cuda').eval()
        model.requires_grad_(False)
        gc = model.generation_config
        eos = {gc.eos_token_id} if isinstance(gc.eos_token_id, int) else set(gc.eos_token_id)
        eos.add(tok.convert_tokens_to_ids('<|im_start|>'))
        im_end = tok.convert_tokens_to_ids('<|im_end|>')
        layer = cfg['site']['layer']
        if not a.resume:
            check = {}
            for form in ('B', 'C'):
                ids = prompts[sources[0], 'ordinary'][form]
                x = torch.tensor([ids], device='cuda')
                with torch.inference_mode():
                    base = model(input_ids=x, logits_to_keep=1).logits
                    with Steer(model, layer, torch.zeros(model.config.hidden_size)) as st:
                        st.start = len(ids) - 1
                        zero = model(input_ids=x, logits_to_keep=1).logits
                check[form] = bool(torch.equal(base, zero))
            assert all(check.values()), check
            write(a.out / 'preflight.json', dict(prompt_ids_match_v21=len(prompts), model_sha256_match=True, zero_hook_bitwise=check,
                                                 device=torch.cuda.get_device_name(0), torch=torch.__version__,
                                                 transformers=__import__('transformers').__version__))

        start = time.perf_counter()
        done = 0
        for rtype in cfg['request_types']:
            for s in sources:
                unit = a.out / 'units' / f'{s}_{rtype}.json'
                if unit.exists():
                    done += 1
                    continue
                status(phase='running', request_type=rtype, source_id=s, units_done=done, seconds=time.perf_counter() - start)
                h, ids, vec = histories[s], prompts[s, rtype], directions['vectors'][s]
                gens = {}
                for name, (form, v) in plan(vec, rtype).items():
                    t = time.perf_counter()
                    with Steer(model, layer, v) as st:
                        g = generate(model, ids[form], eos, cfg['cap'], st)
                    if v is not None:
                        assert len(st.calls) == len(g['generated_ids']) and st.calls[0] == (len(ids[form]), len(ids[form]) - 1)
                        assert all(c == (1, 0) for c in st.calls[1:])
                    text = tok.decode(g['token_ids'], skip_special_tokens=True)
                    term = 'cap' if g['truncated'] else 'assistant_eos' if g['generated_ids'][-1] == im_end else 'unexpected_role_stop'
                    gens[name] = dict(form=form, steered=v is not None, text=text, termination=term,
                                      score=check_rule(text, rtype, term != 'assistant_eos', cfg['word_limit'], h['inst_name'], h['inst_args']),
                                      seconds=time.perf_counter() - t, **g)
                for name, g in gens.items():
                    g['exact_vs_B'] = g['generated_ids'] == gens['B']['generated_ids']
                    g['exact_vs_C'] = g['generated_ids'] == gens['C']['generated_ids']
                tf = {}
                for ref_name, members in (('B', cfg['teacher_forced_kl']['along_B_reference']), ('C', cfg['teacher_forced_kl']['along_C_reference'])):
                    reference = gens[ref_name]['generated_ids']
                    conds = plan(vec, rtype)
                    with Steer(model, layer, None) as st:
                        ref_logp = teacher_forced_logprobs(model, ids[ref_name], reference, st)
                    tf[ref_name] = {}
                    for name in members:
                        form, v = conds[name]
                        with Steer(model, layer, v) as st:
                            logp = teacher_forced_logprobs(model, ids[form], reference, st)
                        if v is not None:
                            assert st.calls == [(len(ids[form]) + len(reference) - 1, len(ids[form]) - 1)]
                        tf[ref_name][name] = kl_summary(ref_logp, logp, reference)
                        del logp
                    del ref_logp
                reproduction = {c: gens[c]['generated_ids'] == old[s, rtype, c]['generated_ids'] for c in ('B', 'C')}
                if rtype == 'ordinary':
                    states = {form: capture_states(model, ids[form], CAPTURE) for form in ('B', 'C', 'C_qa', 'C_json')}
                    torch.save(dict(source_id=s, layers=CAPTURE, n_tokens={f: len(ids[f]) for f in ids}, states=states),
                               a.out / 'cross_format' / f'{s}.pt')
                record = dict(source_id=s, case_id=h['case_id'], inst_name=h['inst_name'], request_type=rtype,
                              prompt_tokens={f: len(ids[f]) for f in ids}, generations=gens, teacher_forced=tf,
                              v21_reproduction=reproduction)
                tmp = unit.with_suffix('.tmp')
                tmp.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
                os.replace(tmp, unit)
                done += 1
                torch.cuda.empty_cache()
        units = len(list((a.out / 'units').glob('*.json')))
        assert units == len(sources) * len(cfg['request_types'])
        status(phase='completed', units=units, seconds=time.perf_counter() - start,
               cross_format_files=len(list((a.out / 'cross_format').glob('*.pt'))))
    except Exception:
        status(phase='failed', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
