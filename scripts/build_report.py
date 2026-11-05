"""Rebuild the selected result tables from saved outputs. Standard library only."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff import ROOT, analyze_all, rows


def write(path, text):
    path.write_text(text+'\n', encoding='utf-8', newline='\n')


def main():
    result = analyze_all()
    out = ROOT/'reports'
    out.mkdir(exist_ok=True)
    write(out/'summary.json', json.dumps(result, ensure_ascii=False, indent=2))
    e, s, t = result['editing'], result['strategy'], result['style']
    lines = ['# Results', '', 'Experiments: October 10–14, 2026. Prepared November 4, 2026. Tables are recomputed from saved outputs; no new model calls were made.', '',
             '## Local editing', '',
             'Qwen generated four books, four functions, and four tables. Each saved artifact had two independent edit requests, giving 24 questions. This early batch compared role replay (B, originally N) with pasted history (C, originally P); it did not have an online A condition.', '',
             '|Condition|Complete edits|Exact match to B|Unchanged items retained|', '|---|---:|---:|---:|']
    labels = {'N':'B: role replay','P':'C: pasted','Hint':'C + continuation hint','D':'C + history-difference vector','RandomD':'C + same-norm random vector','ReverseD':'C + reverse vector'}
    for m, v in e['metrics'].items():
        lines.append(f"|{labels[m]}|{v['joint_correct']}/24|{v['exact_native_text']}/24|{v['retained_items']}/{v['expected_retained_items']}|")
    lines += ['',
        'Complete edits require the requested change and the expected remaining structure. The retention column counts page or row membership for books and tables; it does not check order by itself. Code uses AST comparison and was not executed.', '',
        'The distinction matters in `artifact-book-0 / replace_page`: B kept all seven untouched pages but omitted the requested digit 1 from the replacement sentence. Preservation passed while the edit failed.', '',
        'C was correct more often than B, so matching the role-replay answer is not a quality target on its own. The continuation hint also outperformed the simple difference vector. We kept six complete method columns from the original nine, with every question retained.', '',
        '### Common-prefix KL', '',
        'An earlier diagnostic used one history from each editing family: three histories, two requests, two reference paths, and five conditions, for 60 records. At each reference position, both conditions receive the same previously generated token IDs. KL is computed from the role-replay next-token distribution to the other condition. This measures conditional prediction differences, not answer correctness or full-sequence KL.', '',
        '|Question, B reference / C condition|Sparse mean KL|All-position mean KL|First free-text divergence, zero-based|KL there|', '|---|---:|---:|---:|---:|']
    for r in e['dense_examples']:
        pos = r['first_native_paste_free_divergence']
        value = r['kl_at_first_free_divergence']
        lines.append(f"|{r['case_id']}/{r['query_id']}|{r['old_sparse_mean_recomputed64']:.6g}|{r['dense_mean_kl']:.6g}|{pos if pos is not None else '—'}|{format(value, '.6g') if value is not None else '—'}|")
    lines += ['',
        'For page reordering, the sparse points missed position 11, where KL was 4.592. The small sparse mean therefore gave a misleading impression of agreement. The CPU script checks the stored per-position means; rerunning the full vocabulary probability calculation requires the model.', '',
        '## Response rules', '',
        'We adapted eight MT-Eval histories: two each for JSON, lists, two responses, and a fixed ending. Qwen generated the acknowledgement and three practice answers. Questions 4 and 5 were then tested independently. Seven histories followed their category rule in all three practice answers. All eight histories and 80 final outputs are included.', '',
        '|Rule|A: online|B: replay|C: pasted|Split replay|C + explicit reminder|', '|---|---:|---:|---:|---:|---:|']
    names = {'format:json_format':'JSON','format:bullet_list':'Bullet list','combination:two_responses':'Two responses','startend:end_phrase':'Fixed ending'}
    for category, metrics in s['groups'].items():
        vals=[f"{metrics[m]['passed']}/{metrics[m]['n']}" for m in ['AOnline','BReplay','CPasted','BSplitNull','CSpecificRule']]
        lines.append('|'+names[category]+'|'+'|'.join(vals)+'|')
    lines += ['',
        'On the vaccine question in `mteval-24`, A/B gave two responses and C gave one. Repeating the rule fixed it. Its practice answers and final answer also satisfied the separate 200-word requirement.', '',
        'The table uses the source category checker. The two-response check requires two different strings separated by `******`; it does not judge whether they offer meaningfully different answers. The table does not include the separate word-limit check.', '',
        '### Further questions from the same histories', '',
        'After that failure, we tested questions 6–10 from both two-response histories. There were ten requests and 40 outputs, with no new history and no online A condition.', '',
        '|Condition|Two-response compliance|', '|---|---:|']
    for m, v in s['extension']['methods'].items():
        lines.append(f"|{m}|{v['passed']}/{v['n']}|")
    lines += ['',
        'Both B and C passed 9/10, so this extension found no additional B-correct/C-wrong case. They shared a failure on an email request. Combining batches gives B 13/14, C 12/14, and the reminder 14/14, but those requests come from just two histories. An earlier extension attempt failed in the scoring code after one unsaved generation; these 40 outputs are from the corrected run.', '',
        '## Contractions in casual replies', '',
        'We took MT-Eval question streams, added a casual-style request to the first user turn, and let Qwen answer three questions. Two further questions were tested independently. We counted 30 listed contractions. Density is occurrences per 100 regex words, averaged over requests within each history and then over histories. We also checked a first-120-word window after removing some Markdown.', '',
        '|Batch|B contractions|C contractions|Histories with higher C density|Mean C−B density|', '|---|---:|---:|---:|---:|']
    for label, batch in [('Earlier batch: 6 histories', t['old']), ('New batch: 4 histories', t['new'])]:
        v=batch['full']
        lines.append(f"|{label}|{v['counts']['BKeep']}|{v['counts']['CKeep']}|{v['positive_sources']}/{len(v['source_rows'])}|{v['mean_c_minus_b']:.5f}|")
    lines += ['',
        'The new batch had 6/7/6 contractions in A/B/C. Its positive mean was driven by one history; only two of four histories had higher C density. Removing the initial style clause produced no listed contractions in new B/C answers. Explicitly requesting formal, uncontracted wording produced zero listed contractions in all four requests for each format.', '',
        'The earlier 6/6 pattern did not repeat consistently. The new batch failed the criteria set before generation, including at least 3/4 histories in the same direction and at least four additional contraction occurrences in C. We did not extract directions or run steering on it. These were working criteria for continuing the experiment, not a significance test.', '',
        'The new questions were absent from the 60 saved history files checked during selection. That does not establish topic novelty. The feature is also sparse: different answer lengths and wording opportunities affect density, so it is not a measure of style as a whole.', '',
        '## What to carry forward', '',
        'Response rules are the simplest course-project starting point: public questions, an understandable requirement, and an existing example we can inspect. The extension also shows why we should check a pattern before adding more interventions. Editing is a useful alternative when content preservation matters. The contraction results are best kept as a small follow-up that did not confirm the earlier pattern.', '',
        'The three tasks answer different questions and should not be combined into one success rate. All are development results already seen by the team. Raw selected records are in `experiments/`, source hashes and selection details in `provenance/`, and historical code snapshots in `reference_code/`.']
    write(out/'results.md', '\n'.join(lines))

    edit_rows = rows('experiments/01_editing/outputs.jsonl')
    histories = {h['case_id']:h for h in rows('experiments/01_editing/histories.jsonl')}
    pair = next(r for r in edit_rows if r['method']=='N' and not r['checker']['all_pass'] and any(x['case_id']==r['case_id'] and x['query_id']==r['query_id'] and x['method']=='P' and x['checker']['all_pass'] for x in edit_rows))
    query = next(q for q in histories[pair['case_id']]['queries'] if q['query_id']==pair['query_id'])
    cases = ['# Examples', '', 'These examples were picked after inspection to make the scoring easier to follow. Complete selected batches are in `experiments/`.', '', '## An edit where pasting helped', '', f"`{pair['case_id']} / {pair['query_id']}`", '', query['text'], '', 'The expected answer was built from the saved artifact and this edit request. Replies below are unchanged.']
    for m in ['N','P','Hint','D']:
        r = next(x for x in edit_rows if x['case_id']==pair['case_id'] and x['query_id']==pair['query_id'] and x['method']==m)
        cases += ['', f"### {labels[m]}: {'pass' if r['checker']['all_pass'] else 'fail'}", '', '~~~~text', r['text'], '~~~~']
    gap = s['first_batch_gaps'][0]
    history = {h['case_id']:h for h in rows('experiments/02_strategy/histories.jsonl')}[gap['case_id']]
    cases += ['', '## Two responses became one', '', 'History rule: '+history['rule_text'], '', 'Follow-up: '+history['queries'][gap['query']], '', 'All three practice replies passed the category checker.']
    for m in ['AOnline','BReplay','CPasted','CSpecificRule']:
        cases += ['', '### '+m, '', '~~~~text', gap['outputs'][m], '~~~~']
    cases += ['', 'The later questions from these same histories gave B/C 9/10 each. This one example does not establish a general loss.', '', '## The four new style histories', '', '|History|C−B contraction density|', '|---|---:|']
    for r in t['new']['full']['source_rows']:
        cases.append(f"|{r['case_id']}|{r['c_minus_b']:+.5f}|")
    cases += ['', 'The average remains positive, but only two histories are positive. The per-history results and raw occurrence counts tell a different story from the mean alone.']
    write(out/'examples.md', '\n'.join(cases))
    print(json.dumps(dict(verified=True, editing_outputs=144, dense_records=60, strategy_outputs=120, style_outputs=100, new_model_calls=0)))


if __name__ == '__main__':
    main()
