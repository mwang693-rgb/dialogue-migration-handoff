import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')
        f.flush()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split())


def validate_cases(cases, cfg, confirmation=False):
    assert len({c['case_id'] for c in cases}) == len(cases), 'Duplicate case IDs'
    if confirmation:
        assert len(cases) == cfg['target_sources']
    seen = set()
    for c in cases:
        assert c['source_id'] and c['case_id']
        assert re.fullmatch(r'[A-Za-z0-9_-]+',c['case_id']), 'Use a safe case filename'
        assert len(c['history_questions']) == cfg['history_turns']
        assert isinstance(c['question'], str) and c['question'].strip()
        assert c.get('provenance'), 'Record source or construction recipe'
        for q in c['history_questions'] + [c['question']]:
            n = normalize(q)
            assert n and n not in seen, 'Repeated question: group or revise before freezing'
            seen.add(n)
    # One case per source is the primary protocol; paraphrases are not new sources.
    assert len({c['source_id'] for c in cases}) == len(cases)


def split_sources(source_ids, seed=453, fractions=(0.7, 0.15, 0.15)):
    ids = sorted(set(source_ids))
    assert len(ids) >= 12, 'At least twelve sources required for train/validation/test'
    random.Random(seed).shuffle(ids)
    train = max(1, int(len(ids)*fractions[0]))
    val = max(1, int(len(ids)*fractions[1]))
    assert train+val < len(ids)
    return {s: 'train' if i < train else 'validation' if i < train+val else 'test' for i,s in enumerate(ids)}


def check_rule(text, request_type, truncated=False, word_limit=200):
    parts = [s.strip() for s in text.split('******') if s.strip()]
    words = len(text.split())
    if request_type == 'ordinary':
        form = len(parts) == 2 and parts[0] != parts[1] and text.count('******') == 1
    elif request_type == 'change':
        form = bool(text.strip()) and '******' not in text
    else:
        raise ValueError(request_type)
    return dict(form_pass=form, words=words, length_pass=words <= word_limit,
                complete=not truncated, passed=bool(form and words <= word_limit and not truncated),
                definition='Surface response-form + word limit + natural stop; not factual quality or semantic distinctness.')


def sample_audit_sources(source_ids, seed=453, count=12):
    # Selection depends only on the source IDs and seed, never output/condition.
    return random.Random(seed).sample(sorted(set(source_ids)), min(count, len(set(source_ids))))


def request_text(question, request_type, condition, cfg):
    assert request_type in ['ordinary','change'] and condition in cfg['conditions']
    text=question+ ('\n\n'+cfg['override'] if request_type=='change' else '')
    if condition=='C_reminder':
        text+='\n\nFor this reply, follow this response requirement: '+(cfg['ordinary_rule'] if request_type=='ordinary' else cfg['override'])
    return text
