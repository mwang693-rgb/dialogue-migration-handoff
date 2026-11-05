"""Bounded new A/B/C histories using the preserved original OnlineEngine.

This is a new-run convenience wrapper, not a claim of byte-identical reproduction.
No steering or automatic parameter search. Requires a local Qwen model and CUDA.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handoff import rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['strategy', 'style'], required=True)
    parser.add_argument('--model', required=True, help='Local model directory; no automatic download')
    parser.add_argument('--max-cases', type=int, default=1)
    parser.add_argument('--cap', type=int, default=1536)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.max_cases < 1 or args.cap < 1:
        parser.error('positive bounds are required')
    return args


def append(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False)+'\n')


def history_users(source):
    return [m['content'] for m in source['history'] if m['role'] == 'user']


def query_items(source):
    queries = source['queries']
    return list(queries.items()) if isinstance(queries, dict) else [(str(i), q) for i, q in enumerate(queries)]


def main():
    args = parse_args()
    assert not args.out.exists(), 'Choose a NEW output directory; archived results are immutable.'
    assert args.model and Path(args.model).is_dir(), 'Model must be available locally.'
    args.out.mkdir(parents=True)
    folder = '02_strategy' if args.experiment == 'strategy' else '03_style'
    selected = rows(f'experiments/{folder}/histories.jsonl')[:args.max_cases]
    cfg = dict(experiment=args.experiment, model=args.model, cases=[c['case_id'] for c in selected], cap=args.cap, methods=['AOnline','BReplay','CPasted'], history_regenerated=True, historical_outputs_reused_as_history=False, sampling=False, repetition_penalty=1.0, new_run_not_original_reproduction=True, new_wrapper_gpu_validated=False)
    (args.out/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    sys.path.insert(0, str(ROOT/'reference_code'))
    import migration_exploration_core_v1 as core
    core.MODEL = str(Path(args.model).resolve())
    from run_policy_threeway_v1 import OnlineEngine
    e = OnlineEngine()
    e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'))
    try:
        for source in selected:
            history, consumed, cache = [], [], None
            for turn, user in enumerate(history_users(source)):
                canonical = e.serialize(history)
                full = e.serialize(history, user)
                if history:
                    assert consumed == canonical and full[:len(canonical)] == canonical, 'Retokenization drift; save and inspect, do not silently repair.'
                    suffix = full[len(canonical):]
                else:
                    suffix = full
                actual = consumed + suffix
                e.calls = []
                logits, cache = e.feed(suffix, cache)
                reply, cache, consumed = e.decode(logits, cache, actual, args.cap)
                append(args.out/'history_attempts.jsonl', dict(case_id=source['case_id'], turn=turn, calls=e.calls.copy(), **reply))
                assert not reply['truncated'] and reply['generated_ids'][-1] == e.tok.convert_tokens_to_ids('<|im_end|>'), 'Invalid history stop; not a valid continuation sample.'
                assert not set(reply['token_ids']).intersection(e.tok.all_special_ids)
                history.extend([dict(role='user',content=user),dict(role='assistant',content=reply['text'])])
                tail = e.tok.encode('<|im_end|>\n', add_special_tokens=False)
                _, cache = e.feed(tail, cache)
                consumed += tail
                assert consumed == e.serialize(history), 'Generated IDs differ from text retokenization.'
            append(args.out/'histories.jsonl', dict(case_id=source['case_id'], history=history, history_ids=consumed))
            for query_id, query in query_items(source):
                canonical = e.serialize(history)
                native = e.serialize(history, query)
                assert native[:len(canonical)] == canonical and consumed == canonical
                for method in cfg['methods']:
                    e.calls = []
                    ids = e.encode(history, query, pasted=method=='CPasted')['input_ids']
                    if method == 'AOnline':
                        branch = copy.deepcopy(cache)
                        logits, branch = e.feed(ids[len(consumed):], branch)
                    else:
                        logits, branch = e.feed(ids)
                    reply, branch, _ = e.decode(logits, branch, ids.copy(), args.cap)
                    append(args.out/'outputs.jsonl', dict(case_id=source['case_id'], query=query_id, query_text=query, method=method, input_ids=ids, calls=e.calls.copy(), **reply))
                    del branch
            del cache
        (args.out/'status.json').write_text(json.dumps(dict(phase='completed', reminder='Score new replies separately; do not append to the archived tables.'))+'\n')
    except Exception:
        (args.out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
