"""Extract the 81 open-group transfer units as text (user turns, final request, serializer) so
that each new model can regenerate its own assistant turns. CPU only.

Every unit is checked: re-serializing its original (Qwen-written) history with the Qwen
tokenizer must reproduce the transfer v1 B and C prompt IDs exactly.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'replication'))
from confirmation.common import rows, sha, write
from engine import messages

OPEN_TEXT_QUERY = ('Continue our conversation by proposing the most useful next step, keeping earlier decisions and '
                   'preferences in mind. Do not summarize the transcript.')
PILOT = {'dataset_expansion_v1', 'dataset_study_v1', 'open_text_fidelity_v1'}
CORE = {'style_migration_v1', 'existing_continuity_v3'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--transfer-inputs', type=Path, required=True)
    p.add_argument('--qwen-tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.qwen_tokenizer), local_files_only=True)
    enc = lambda msgs: tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)['input_ids']
    res = a.recovery / 'results'
    units = [u for u in rows(a.transfer_inputs / 'units.jsonl') if u['group'] == 'open']
    hist, outs = {}, {}
    out = []
    for u in units:
        pool, key = u['pool'], u['key']
        if pool not in hist:
            hist[pool] = {r['case_id']: r for r in rows(res / pool / 'histories.jsonl')}
            outs[pool] = rows(res / pool / 'outputs.jsonl')
        h = hist[pool][key[0]]['history']
        if pool == 'open_text_fidelity_v1':
            query, serializer = OPEN_TEXT_QUERY, 'pilot'
        elif pool in PILOT:
            r = next(o for o in outs[pool] if o['case_id'] == key[0] and o.get('query_id') == key[1] and o.get('format') == 'N')
            query, serializer = r['query'], 'pilot'
        else:
            r = next(o for o in outs[pool] if o['case_id'] == key[0] and o.get('method') == 'BReplay' and json.dumps(o.get('query')) == json.dumps(key[1]))
            query, serializer = r['input']['query_text'], 'core'
        assert enc(messages(h, query, 'B', True, serializer)) == u['B'], (u['unit_id'], 'B')
        assert enc(messages(h, query, 'C', True, serializer)) == u['C'], (u['unit_id'], 'C')
        users = [m['content'] for m in h if m['role'] == 'user']
        assert [m['role'] for m in h] == ['user', 'assistant'] * len(users)
        out.append(dict(unit_id=u['unit_id'], pool=pool, key=key, group='open', serializer=serializer, user_turns=users, query=query,
                        history_key=hashlib.sha256(json.dumps(users, ensure_ascii=False).encode()).hexdigest()[:16],
                        qwen_history_key=u['history_key']))
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'open_units.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    m = dict(open_units_sha256=sha(a.out / 'open_units.jsonl'), n_units=len(out), n_histories=len({r['history_key'] for r in out}),
             by_pool=dict(Counter(r['pool'] for r in out)), turns=dict(Counter(len(r['user_turns']) for r in out)),
             verified_against_transfer_v1_qwen_ids=True)
    write(a.out / 'manifest.json', m)
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
