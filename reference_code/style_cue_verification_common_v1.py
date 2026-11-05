"""Frozen text manipulations for style calibration and four-condition ablation."""
from pathlib import Path
import copy
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/style_migration_v1'
METHODS = ['BKeep', 'CKeep', 'BRemove', 'CRemove']
CONTRACTIONS = [
    ('cannot', "can't"), ('do not', "don't"), ('does not', "doesn't"),
    ('did not', "didn't"), ('is not', "isn't"), ('are not', "aren't"),
    ('was not', "wasn't"), ('were not', "weren't"), ('will not', "won't"),
    ('would not', "wouldn't"), ('should not', "shouldn't"), ('could not', "couldn't"),
    ('have not', "haven't"), ('has not', "hasn't"), ('had not', "hadn't"),
    ('I am', "I'm"), ('you are', "you're"), ('we are', "we're"),
    ('they are', "they're"), ('I have', "I've"), ('we have', "we've"),
    ('they have', "they've"), ('you have', "you've"), ('I will', "I'll"),
    ('you will', "you'll"), ('we will', "we'll"), ('they will', "they'll"),
    ('he will', "he'll"), ('she will', "she'll"), ('it will', "it'll"),
]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def plaintext(s):
    s = re.sub(r'(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', s)
    return ' '.join(s.replace('**', '').replace('__', '').replace('`', '').split()[:120])

def casing(source, replacement):
    if source.isupper() and len(source) > 1: return replacement.upper()
    if source[0].isupper(): return replacement[0].upper() + replacement[1:]
    return replacement

def expanded(s):
    for full, short in CONTRACTIONS:
        pattern = r'\b' + re.escape(short).replace("'", "['’]") + r'\b'
        s = re.sub(pattern, lambda m: casing(m.group(), full), s, flags=re.I)
    return s

def contracted(s):
    edits = 0
    for full, short in CONTRACTIONS:
        s, n = re.subn(r'\b' + re.escape(full) + r'\b', lambda m: casing(m.group(), short), s, flags=re.I)
        edits += n
    return s, edits

def calibration_materials():
    valid = {h['case_id'] for h in rows(BASE / 'histories.jsonl')}
    pairs, skipped = [], []
    for r in rows(BASE / 'history_outputs.jsonl'):
        if r['case_id'] not in valid: continue
        original = plaintext(r['text'])
        full = expanded(original)
        short, edits = contracted(full)
        key = dict(case_id=r['case_id'], turn=r['turn'], source_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest())
        if not edits:
            skipped.append(dict(**key, reason='No eligible contraction pattern; exclusion independent of encoder scores'))
            continue
        assert full != short
        pairs.append(dict(**key, flow=r['flow'], style=r['style'], expanded=full, contracted=short,
                          layout='**' + full + '**', edits=edits))
    assert len(pairs) >= 2
    for i, p in enumerate(pairs):
        candidates = [j for j, q in enumerate(pairs) if q['flow'] != p['flow']]
        assert candidates
        p['other_content_index'] = candidates[i % len(candidates)]
    return pairs, skipped

def history_variant(h, keep):
    history = copy.deepcopy(h['history'])
    clause = 'Across this conversation, keep the following writing style unless I explicitly ask to change it: ' + h['style_instruction'] + ' '
    assert history[0]['role'] == 'user' and history[0]['content'].startswith(clause)
    if not keep: history[0]['content'] = history[0]['content'][len(clause):]
    assert history[1:] == h['history'][1:]
    assert all(a == b for a, b in zip(history, h['history']) if a['role'] == 'assistant')
    return history, clause

def design():
    histories = rows(BASE / 'histories.jsonl')
    assert len(histories) == 11 and len({h['case_id'] for h in histories}) == 11
    result = []
    for h in histories:
        for query, text in enumerate(h['queries']):
            for method in METHODS:
                history, clause = history_variant(h, method.endswith('Keep'))
                result.append(dict(case_id=h['case_id'], flow=h['flow'], style=h['style'], query=query,
                                   query_text=text, method=method, pasted=method.startswith('C'),
                                   keep_style_instruction=method.endswith('Keep'), history=history,
                                   removed_clause='' if method.endswith('Keep') else clause))
    assert len(result) == 88
    return result

if __name__ == '__main__':
    pairs, skipped = calibration_materials()
    print(json.dumps(dict(design_inputs=len(design()), calibration_pairs=len(pairs),
                          calibration_skipped=len(skipped), calibration_texts=3*len(pairs))))
