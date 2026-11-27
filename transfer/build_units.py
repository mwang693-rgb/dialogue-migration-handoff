"""Build the frozen native/pasted prompt pairs for format-direction transfer v1 (CPU only).

Each unit is one (history, final request) pair from an earlier exploratory run in the
sibling repository. B = native role turns, C = the same history pasted into one user
message. Prompt IDs come from the old run's saved input IDs when present; otherwise
they are rebuilt with the old run's own serializer and must match the saved input
token count. Groups are fixed here by task type, before any new generation.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, rows, sha, write

SYSTEM = {'role': 'system', 'content': 'You are a helpful assistant.'}
PILOT_SEPARATOR = '\n\nLatest user message:\n'
OPEN_TEXT_QUERY = ('Continue our conversation by proposing the most useful next step, keeping earlier decisions and '
                   'preferences in mind. Do not summarize the transcript.')

# pool: (group, B method, C method, where the IDs live, fields that identify a unit)
SAVED = {
    'qa_rewrite_discovery_v1': ('short', 'N', 'P', 'input_ids', ('case_id',)),
    'structured_qa_discovery_v1': ('short', 'N', 'P', 'input_ids', ('case_id',)),
    'retention_discovery_v1': ('short', 'N', 'P', 'input_ids', ('case_id',)),
    'mti_dynamic_discovery_v1': ('short', 'N', 'P', 'input_ids', ('case_id',)),
    'migration_exploration_baseline_v1': ('short', 'N', 'P', 'input.input_ids', ('case_id', 'query')),
    'approved_plan_dev_v1': ('short', 'N', 'P', 'input_token_ids', ('case_id', 'query')),
    'executable_result_dev_v1': ('short', 'NFull', 'PFull', 'input_ids', ('case_id', 'query')),
    'frozen_operation_v1': ('short', 'NFull', 'PFull', 'input_ids', ('case_id', 'query')),
    'plan_practice_v1': ('short', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'policy_practice_v2': ('short', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'policy_threeway_v1': ('short', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'style_migration_v1': ('open', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'existing_continuity_v3': ('open', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'mteval_continuity_v1': ('near', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query')),
    'mteval_tworesponse_extension_v2': ('near', 'BReplay', 'CPasted', 'input.input_ids', ('case_id', 'query_index')),
    'mti_diagnostic_v1': ('external', 'GoldN', 'GoldP', 'input_ids', ('case_id',)),
}
# pools whose prompts are rebuilt with scripts/run_pilot.py:messages (format/method N vs P)
REBUILT = ['dataset_study_v1', 'dataset_expansion_v1', 'open_text_fidelity_v1', 'active_task_dev_v2',
           'frozen_update_stress_v1', 'artifact_protocol_v2', 'artifact_expansion_v1']
DATASET_STUDY_GROUP = {'self_short': 'short', 'lost_math': 'open', 'multiif': 'open', 'wild_regenerated': 'open',
                       'wild_original': 'external'}
STRUCTURED = {'frozen_update_stress_v1', 'artifact_protocol_v2', 'artifact_expansion_v1'}  # first request per history only
GROUP_ORDER = ['short', 'open', 'structured', 'near', 'external']


def get(row, path):
    for k in path.split('.'):
        row = row[k]
    return row


def pilot_messages(history, query, pasted):
    """Copy of dialogue-migration-recovery/scripts/run_pilot.py:messages (hint=False)."""
    if not pasted:
        return [SYSTEM] + history + [{'role': 'user', 'content': query}]
    text = '\n\n'.join(m['role'].capitalize() + ':\n' + m['content'] for m in history)
    return [SYSTEM, {'role': 'user', 'content': text + PILOT_SEPARATOR + query}]


def encode(tok, msgs):
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)['input_ids']


def h(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class Checker:
    def __init__(self, tok):
        self.tok = tok
        self.im_start = tok.convert_tokens_to_ids('<|im_start|>')
        self.head = encode(tok, [SYSTEM])[:-3]  # <|im_start|>system ... <|im_end|>\n
        self.tail = tok('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']

    def history_key(self, b):
        """Fingerprint of the native history: B IDs before the final user turn."""
        starts = [i for i, t in enumerate(b) if t == self.im_start]
        return h(b[:starts[-2]])

    def problems(self, b, c):
        out = []
        for name, ids in (('B', b), ('C', c)):
            if ids[:len(self.head)] != self.head:
                out.append(f'{name}_system_prompt_differs')
            if ids[-len(self.tail):] != self.tail:
                out.append(f'{name}_does_not_end_with_assistant_header')
        if c.count(self.im_start) != 3:
            out.append('C_is_not_single_user_message')
        return out


def saved_units(results, pool, spec):
    group, mb, mc, field, key = spec
    by = defaultdict(dict)
    for r in rows(results / pool / 'outputs.jsonl'):
        if r.get('method') in (mb, mc):
            k = tuple(json.dumps(r.get(f)) for f in key)
            assert r['method'] not in by[k], (pool, k, r['method'])
            by[k][r['method']] = r
    for k, pair in by.items():
        assert set(pair) == {mb, mc}, (pool, k)
        b, c = get(pair[mb], field), get(pair[mc], field)
        yield dict(pool=pool, group=group, key=[json.loads(x) for x in k], id_source='saved_input_ids', B=b, C=c,
                   origin='external' if group == 'external' else 'qwen')


def rebuilt_units(results, pool, tok):
    hist = {r['case_id']: r for r in rows(results / pool / 'histories.jsonl')}
    cases = {r['case_id']: r for r in rows(results / pool / 'cases.jsonl')} if (results / pool / 'cases.jsonl').exists() else {}
    by = defaultdict(dict)
    for r in rows(results / pool / 'outputs.jsonl'):
        form = r.get('format') or r.get('method')
        if form in ('N', 'P'):
            k = (r['case_id'], r.get('query_id'))
            assert form not in by[k], (pool, k)
            by[k][form] = r
    first_query = {c: r['queries'][0]['query_id'] for c, r in hist.items() if 'queries' in r}
    for (case, qid), pair in by.items():
        assert set(pair) == {'N', 'P'}, (pool, case, qid)
        if pool in STRUCTURED and qid != first_query[case]:
            continue
        hrow = hist[case]
        if pool == 'open_text_fidelity_v1':
            query = OPEN_TEXT_QUERY
        elif isinstance(pair['N'].get('query'), str):
            query = pair['N']['query']
        else:
            source = hrow if 'queries' in hrow else cases[case]
            query = next(q['text'] for q in source['queries'] if q['query_id'] == qid)
        if pool == 'dataset_study_v1':
            group = DATASET_STUDY_GROUP[pair['N']['source']]
        elif pool in STRUCTURED:
            group = 'structured'
        elif pool == 'active_task_dev_v2':
            group = 'short'
        else:
            group = 'open'
        b = encode(tok, pilot_messages(hrow['history'], query, False))
        c = encode(tok, pilot_messages(hrow['history'], query, True))
        counts = dict(B=[len(b), pair['N']['input_tokens']], C=[len(c), pair['P']['input_tokens']])
        unit = dict(pool=pool, group=group, key=[case, qid], id_source='rebuilt_run_pilot_serializer', B=b, C=c,
                    origin='external' if group == 'external' else 'qwen', rebuilt_token_counts=counts)
        if counts['B'][0] != counts['B'][1] or counts['C'][0] != counts['C'][1]:
            unit['exclude'] = 'rebuilt_token_count_differs_from_saved'
        yield unit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery', type=Path, required=True, help='dialogue-migration-recovery folder')
    p.add_argument('--v21', type=Path, required=True, help='extracted confirmation-v2_1 run folder')
    p.add_argument('--directions', type=Path, required=True, help='intervention v1 directions.pt')
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    chk = Checker(tok)
    results = a.recovery / 'results'
    assert OPEN_TEXT_QUERY in (a.recovery / 'scripts/run_open_text_fidelity_v1.py').read_text(encoding='utf-8')
    assert PILOT_SEPARATOR.encode('unicode_escape').decode() in (a.recovery / 'scripts/run_pilot.py').read_text(encoding='utf-8')

    v21_keys = {chk.history_key(r['input_ids']) for r in rows(a.v21 / 'outputs.jsonl') if r['condition'] == 'B'}
    raw = [u for pool, spec in SAVED.items() for u in saved_units(results, pool, spec)]
    raw += [u for pool in REBUILT for u in rebuilt_units(results, pool, tok)]

    units, excluded, seen = [], [], set()
    for u in raw:
        u['history_key'] = chk.history_key(u['B'])
        u['problems'] = chk.problems(u['B'], u['C'])
        pair = (h(u['B']), h(u['C']))
        reason = u.get('exclude') or ('duplicate_prompt_pair' if pair in seen else None) \
            or ('history_in_v21' if u['history_key'] in v21_keys else None) or (u['problems'][0] if u['problems'] else None)
        if reason:
            excluded.append(dict(pool=u['pool'], key=u['key'], reason=reason))
            continue
        seen.add(pair)
        units.append(u)
    units.sort(key=lambda u: (GROUP_ORDER.index(u['group']), u['pool'], json.dumps(u['key'])))
    for i, u in enumerate(units):
        u['unit_id'] = f"{i:03d}-{u['pool']}"

    d = torch.load(a.directions, map_location='cpu', weights_only=True)
    mean = d['full_sample_mean'].float()
    g = torch.Generator().manual_seed(853)
    rand = torch.randn(len(units), mean.numel(), generator=g, dtype=torch.float64)
    rand = (rand / rand.norm(dim=1, keepdim=True) * mean.double().norm()).float()

    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'units.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for u in units:
            f.write(json.dumps({k: u[k] for k in ('unit_id', 'pool', 'group', 'origin', 'key', 'history_key', 'id_source', 'B', 'C')
                                + (('rebuilt_token_counts',) if 'rebuilt_token_counts' in u else ())}, ensure_ascii=False) + '\n')
    torch.save(dict(schema_version=1, layer=d['layer'], mean=mean, random=rand, random_seed=853,
                    unit_ids=[u['unit_id'] for u in units], directions_sha256=sha(a.directions)), a.out / 'vectors.pt')
    cnt = Counter((u['group'], u['pool']) for u in units)
    manifest = dict(
        units_sha256=sha(a.out / 'units.jsonl'), vectors_sha256=sha(a.out / 'vectors.pt'), source_directions_sha256=sha(a.directions),
        mean_norm=float(mean.norm()), n_units=len(units),
        n_histories={g_: len({u['history_key'] for u in units if u['group'] == g_}) for g_ in GROUP_ORDER},
        units_by_group={g_: sum(u['group'] == g_ for u in units) for g_ in GROUP_ORDER},
        units_by_pool={f'{g_}/{p_}': n for (g_, p_), n in sorted(cnt.items(), key=lambda kv: (GROUP_ORDER.index(kv[0][0]), kv[0][1]))},
        id_source=dict(Counter(u['id_source'] for u in units)),
        prompt_tokens={x: dict(min=min(len(u[x]) for u in units), max=max(len(u[x]) for u in units)) for x in ('B', 'C')},
        excluded=dict(Counter(f"{e['pool']}:{e['reason']}" for e in excluded)),
        v21_history_overlap=sum(e['reason'] == 'history_in_v21' for e in excluded),
        source_sha256={f'{pool}/outputs.jsonl': sha(results / pool / 'outputs.jsonl') for pool in list(SAVED) + REBUILT})
    write(a.out / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_sha256'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
