"""CPU-only verification of selected, already completed experiments (stdlib)."""
import ast
from collections import defaultdict
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parent
CONTRACTIONS = ["can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't", "haven't", "hasn't", "hadn't", "I'm", "you're", "we're", "they're", "I've", "we've", "they've", "you've", "I'll", "you'll", "we'll", "they'll", "he'll", "she'll", "it'll"]


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in (ROOT / path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def unwrap(text):
    s = text.strip()
    if s.startswith('```'):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        s = '\n'.join(lines).strip()
    return s


def portable_ast_signature(dump):
    """Compare frozen dumps across Python 3.11/3.12 without evaluating them.

    Python 3.12 added empty type_params to FunctionDef/ClassDef. Remove only
    that empty schema field in the parsed dump expression, never string values
    or a nonempty type-parameter declaration. No generated code is executed.
    """
    expression = ast.parse(dump, mode='eval')
    for node in ast.walk(expression):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'FunctionDef', 'AsyncFunctionDef', 'ClassDef'}:
            node.keywords = [k for k in node.keywords if not (k.arg == 'type_params' and isinstance(k.value, ast.List) and not k.value.elts)]
    return ast.dump(expression, include_attributes=False)


def check_edit(text, family, q):
    """Match original joint/retention definitions, without executing model code."""
    try:
        if family == 'code':
            tree = ast.parse(unwrap(text))
            strict = portable_ast_signature(ast.dump(tree, include_attributes=False)) == portable_ast_signature(q['expected_ast'])
            passed = strict
            if q['query_id'] == 'set_docstring':
                expected = ast.parse(q['expected_code'])
                for t in (tree, expected):
                    for n in t.body:
                        if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr):
                            v = n.body[0].value
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                v.value = inspect.cleandoc(v.value)
                passed = ast.dump(tree, include_attributes=False) == ast.dump(expected, include_attributes=False)
            return dict(all_pass=passed, strict_ast_pass=strict, valid_syntax=True)
        actual = json.loads(unwrap(text))
        key = 'pages' if family == 'book' else 'items'
        got = actual.get(key, []) if isinstance(actual, dict) else []
        schema = isinstance(actual, dict) and set(actual) == {key} and isinstance(got, list)
        if family == 'book':
            schema = schema and all(isinstance(x, str) for x in got)
        else:
            schema = schema and all(isinstance(x, dict) and set(x) == {'id', 'location', 'budget', 'status'} and type(x['budget']) is int and all(isinstance(x[k], str) for k in ['id', 'location', 'status']) for x in got)
        retained = sum(x in got for i, x in enumerate(q['original'][key]) if i not in q['changed_indices'])
        return dict(all_pass=actual == q['expected'], valid_schema=bool(schema), retained_unchanged_items=retained, expected_retained=q['expected_retained'])
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return dict(all_pass=False)


def plaintext_first120(text):
    text = re.sub(r'(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', text)
    return ' '.join(text.replace('**', '').replace('__', '').replace('`', '').split()[:120])


def features(text):
    count = sum(len(re.findall(r'\b' + re.escape(s).replace("'", "['’]") + r'\b', text, re.I)) for s in CONTRACTIONS)
    words = len(re.findall(r'\w+', text))
    return dict(count=count, words=words, per100=100 * count / max(1, words))


def style_score(text):
    return dict(full=features(text), first120=features(plaintext_first120(text)))


def source_checker_classes():
    source = (ROOT/'third_party/mt_eval/global_inst.py').read_text(encoding='utf-8-sig')
    names = ['Instruction', 'JsonFormat', 'BulletList', 'TwoResponses', 'EndPhrase']
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == len(names)
    env = dict(json=json, re=re, List=list, Dict=dict, Any=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'mt_eval_original_classes', 'exec'), env)
    return env


def check_strategy(text, history):
    mapping = {'format:json_format': 'JsonFormat', 'format:bullet_list': 'BulletList', 'combination:two_responses': 'TwoResponses', 'startend:end_phrase': 'EndPhrase'}
    return bool(source_checker_classes()[mapping[history['inst_name']]](**history['inst_args']).check_following(text))


def unique(records, fields):
    keys = [tuple(r[f] for f in fields) for r in records]
    assert len(keys) == len(set(keys)), fields


def analyze_editing():
    histories = rows('experiments/01_editing/histories.jsonl')
    outputs = rows('experiments/01_editing/outputs.jsonl')
    assert len(histories) == 12 and len(outputs) == 144
    unique(outputs, ['case_id', 'query_id', 'method'])
    tasks = {(h['case_id'], q['query_id']): q for h in histories for q in h['queries']}
    native = {(r['case_id'], r['query_id']): r['text'] for r in outputs if r['method'] == 'N'}
    metrics = {}
    for method in ['N', 'P', 'Hint', 'D', 'RandomD', 'ReverseD']:
        rs = [r for r in outputs if r['method'] == method]
        assert len(rs) == 24
        checks = [check_edit(r['text'], r['family'], tasks[r['case_id'], r['query_id']]) for r in rs]
        for r, c in zip(rs, checks):
            assert c['all_pass'] == r['checker']['all_pass'], (r['case_id'], r['query_id'], method)
            if 'retained_unchanged_items' in c:
                assert c['retained_unchanged_items'] == r['checker']['retained_unchanged_items']
        kept = sum(c.get('retained_unchanged_items', 0) for c in checks)
        total = sum(tasks[r['case_id'], r['query_id']].get('expected_retained', 0) for r in rs)
        metrics[method] = dict(n=24, joint_correct=sum(c['all_pass'] for c in checks), exact_native_text=sum(r['text'] == native[r['case_id'], r['query_id']] for r in rs), retained_items=kept, expected_retained_items=total, retention_note='Book/page and table/row membership proxy; not order or field-level preservation. Code uses AST joint check separately.', capped=sum(r['truncated'] for r in rs), family_correct={fam: sum(c['all_pass'] for r, c in zip(rs, checks) if r['family'] == fam) for fam in ['book', 'code', 'table']})
    dense = rows('experiments/01_editing/dense_distributions.jsonl')
    assert len(dense) == 60
    unique(dense, ['case_id', 'query_id', 'reference', 'method'])
    for r in dense:
        assert len(r['reference_ids']) == len(r['kl_by_position'])
        assert abs(mean(r['kl_by_position']) - r['dense_mean_kl']) < 1e-10
        assert abs(mean(r['js_by_position']) - r['dense_mean_js']) < 1e-10
        assert not r['reference_capped']
    examples = [{k: r[k] for k in ['case_id', 'query_id', 'reference', 'dense_mean_kl', 'old_sparse_mean_recomputed64', 'first_native_paste_free_divergence', 'kl_at_first_free_divergence']} for r in dense if r['reference'] == 'N' and r['method'] == 'P']
    return dict(histories=12, queries=24, outputs=144, metrics=metrics, dense_records=60, dense_examples=examples)


def analyze_strategy():
    hs = {h['case_id']: h for h in rows('experiments/02_strategy/histories.jsonl')}
    main = rows('experiments/02_strategy/outputs.jsonl')
    ext = rows('experiments/02_strategy/extension_outputs.jsonl')
    assert len(hs) == 8 and len(main) == 80 and len(ext) == 40
    unique(main, ['case_id', 'query', 'method'])
    unique(ext, ['case_id', 'query_index', 'method'])
    for r in main + ext:
        assert check_strategy(r['text'], hs[r['case_id']]) == r['source_pass']
        assert r['termination'] == 'assistant_eos' and not r['truncated']
    grouped = {}
    for category in sorted({h['inst_name'] for h in hs.values()}):
        grouped[category] = {}
        for method in sorted({r['method'] for r in main}):
            rs = [r for r in main if r['inst_name'] == category and r['method'] == method]
            grouped[category][method] = dict(n=len(rs), passed=sum(r['source_pass'] for r in rs))
    extension = {method: dict(n=sum(r['method'] == method for r in ext), passed=sum(r['source_pass'] for r in ext if r['method'] == method)) for method in sorted({r['method'] for r in ext})}
    gap = []
    for r in main:
        if r['method'] != 'CPasted' or r['source_pass']:
            continue
        peers = {x['method']: x for x in main if x['case_id'] == r['case_id'] and x['query'] == r['query']}
        if peers['AOnline']['source_pass'] and peers['BReplay']['source_pass']:
            gap.append(dict(case_id=r['case_id'], query=r['query'], inst_name=r['inst_name'], source_history_established=hs[r['case_id']]['established'], outputs={m: peers[m]['text'] for m in peers}))
    return dict(histories=8, established=sum(h['established'] for h in hs.values()), final_outputs=80, groups=grouped, extension=dict(histories=2, outputs=40, methods=extension), first_batch_gaps=gap)


def summarize_style(records, old=False):
    result = {}
    for window in ['full', 'first120']:
        normal = [r for r in records if r['query'] != 'override_q1']
        methods = sorted({r['method'] for r in normal})
        counts = {m: sum(style_score(r['text'])[window]['count'] for r in normal if r['method'] == m) for m in methods}
        per_source = []
        for cid in sorted({r['case_id'] for r in normal}):
            rs = [r for r in normal if r['case_id'] == cid]
            rates = {m: mean(style_score(r['text'])[window]['per100'] for r in rs if r['method'] == m) for m in methods}
            per_source.append(dict(case_id=cid, rates=rates, c_minus_b=rates['CKeep']-rates['BKeep']))
        result[window] = dict(counts=counts, source_rows=per_source, mean_c_minus_b=mean(s['c_minus_b'] for s in per_source), positive_sources=sum(s['c_minus_b'] > 0 for s in per_source), mean_rates={m: mean(s['rates'][m] for s in per_source) for m in methods})
    return result


def analyze_style():
    old = rows('experiments/03_style/old_outputs.jsonl')
    new = rows('experiments/03_style/outputs.jsonl')
    assert len(old) == 48 and len(new) == 52
    for rs in [old, new]:
        unique(rs, ['case_id', 'query', 'method'])
        assert all(not r['truncated'] and r['termination'] == 'assistant_eos' for r in rs)
    for r in new:
        assert style_score(r['text']) == r['score']
    saved = load('experiments/03_style/original_analysis.json')
    current = summarize_style(new)
    assert current['full']['positive_sources'] == saved['groups']['full']['positive_C_minus_B'] == 2
    assert current['full']['counts'] == saved['groups']['full']['total_counts']
    overrides = {m: dict(n=4, zero_short_forms=sum(style_score(r['text'])['full']['count'] == 0 for r in new if r['query'] == 'override_q1' and r['method'] == m)) for m in ['AOnline', 'BKeep', 'CKeep']}
    return dict(old_histories=6, old_outputs=48, old=summarize_style(old, True), new_histories=4, new_outputs=52, new=current, override=overrides, frozen_gate_pass=saved['gate_pass'], gate_reasons=saved['gate_reasons'], steering_outputs=0)


def analyze_all():
    return dict(editing=analyze_editing(), strategy=analyze_strategy(), style=analyze_style(), note='Recomputed from the selected recorded outputs, not new inference. Source histories, not outputs, are sampling units.')
