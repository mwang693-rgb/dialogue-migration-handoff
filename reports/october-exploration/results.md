# Results

Experiments: October 10–14, 2026. Prepared November 4, 2026. Tables are recomputed from saved outputs; no new model calls were made.

## Local editing

Qwen generated four books, four functions, and four tables. Each saved artifact had two independent edit requests, giving 24 questions. This early batch compared role replay (B, originally N) with pasted history (C, originally P); it did not have an online A condition.

|Condition|Complete edits|Exact match to B|Unchanged items retained|
|---|---:|---:|---:|
|B: role replay|15/24|24/24|96/100|
|C: pasted|17/24|18/24|97/100|
|C + continuation hint|19/24|17/24|97/100|
|C + history-difference vector|11/24|13/24|90/100|
|C + same-norm random vector|15/24|16/24|98/100|
|C + reverse vector|18/24|18/24|99/100|

Complete edits require the requested change and the expected remaining structure. The retention column counts page or row membership for books and tables; it does not check order by itself. Code uses AST comparison and was not executed.

The distinction matters in `artifact-book-0 / replace_page`: B kept all seven untouched pages but omitted the requested digit 1 from the replacement sentence. Preservation passed while the edit failed.

C was correct more often than B, so matching the role-replay answer is not a quality target on its own. The continuation hint also outperformed the simple difference vector. We kept six complete method columns from the original nine, with every question retained.

### Common-prefix KL

An earlier diagnostic used one history from each editing family: three histories, two requests, two reference paths, and five conditions, for 60 records. At each reference position, both conditions receive the same previously generated token IDs. KL is computed from the role-replay next-token distribution to the other condition. This measures conditional prediction differences, not answer correctness or full-sequence KL.

|Question, B reference / C condition|Sparse mean KL|All-position mean KL|First free-text divergence, zero-based|KL there|
|---|---:|---:|---:|---:|
|artifact-book-0/replace_page|4.86267e-08|0.000974771|—|—|
|artifact-book-0/reorder_pages|1.46912e-08|0.0410309|11|4.59202|
|artifact-code-0/rename_function|0.000112979|9.03374e-06|—|—|
|artifact-code-0/set_docstring|0.000242256|0.000560896|—|—|
|artifact-table-0/delete_row|9.07993e-08|0.000510925|—|—|
|artifact-table-0/update_budget|2.5394e-07|0.000160608|188|0.031259|

For page reordering, the sparse points missed position 11, where KL was 4.592. The small sparse mean therefore gave a misleading impression of agreement. The CPU script checks the stored per-position means; rerunning the full vocabulary probability calculation requires the model.

## Response rules

We adapted eight MT-Eval histories: two each for JSON, lists, two responses, and a fixed ending. Qwen generated the acknowledgement and three practice answers. Questions 4 and 5 were then tested independently. Seven histories followed their category rule in all three practice answers. All eight histories and 80 final outputs are included.

|Rule|A: online|B: replay|C: pasted|Split replay|C + explicit reminder|
|---|---:|---:|---:|---:|---:|
|Two responses|4/4|4/4|3/4|4/4|4/4|
|Bullet list|4/4|4/4|4/4|4/4|4/4|
|JSON|4/4|4/4|4/4|4/4|4/4|
|Fixed ending|3/4|4/4|4/4|4/4|4/4|

On the vaccine question in `mteval-24`, A/B gave two responses and C gave one. Repeating the rule fixed it. Its practice answers and final answer also satisfied the separate 200-word requirement.

The table uses the source category checker. The two-response check requires two different strings separated by `******`; it does not judge whether they offer meaningfully different answers. The table does not include the separate word-limit check.

### Further questions from the same histories

After that failure, we tested questions 6–10 from both two-response histories. There were ten requests and 40 outputs, with no new history and no online A condition.

|Condition|Two-response compliance|
|---|---:|
|BReplay|9/10|
|BSplitNull|9/10|
|CPasted|9/10|
|CSpecificRule|10/10|

Both B and C passed 9/10, so this extension found no additional B-correct/C-wrong case. They shared a failure on an email request. Combining batches gives B 13/14, C 12/14, and the reminder 14/14, but those requests come from just two histories. An earlier extension attempt failed in the scoring code after one unsaved generation; these 40 outputs are from the corrected run.

## Contractions in casual replies

We took MT-Eval question streams, added a casual-style request to the first user turn, and let Qwen answer three questions. Two further questions were tested independently. We counted 30 listed contractions. Density is occurrences per 100 regex words, averaged over requests within each history and then over histories. We also checked a first-120-word window after removing some Markdown.

|Batch|B contractions|C contractions|Histories with higher C density|Mean C−B density|
|---|---:|---:|---:|---:|
|Earlier batch: 6 histories|9|20|6/6|0.61386|
|New batch: 4 histories|7|6|2/4|0.40973|

The new batch had 6/7/6 contractions in A/B/C. Its positive mean was driven by one history; only two of four histories had higher C density. Removing the initial style clause produced no listed contractions in new B/C answers. Explicitly requesting formal, uncontracted wording produced zero listed contractions in all four requests for each format.

The earlier 6/6 pattern did not repeat consistently. The new batch failed the criteria set before generation, including at least 3/4 histories in the same direction and at least four additional contraction occurrences in C. We did not extract directions or run steering on it. These were working criteria for continuing the experiment, not a significance test.

The new questions were absent from the 60 saved history files checked during selection. That does not establish topic novelty. The feature is also sparse: different answer lengths and wording opportunities affect density, so it is not a measure of style as a whole.

## What to carry forward

Response rules are the simplest course-project starting point: public questions, an understandable requirement, and an existing example we can inspect. The extension also shows why we should check a pattern before adding more interventions. Editing is a useful alternative when content preservation matters. The contraction results are best kept as a small follow-up that did not confirm the earlier pattern.

The three tasks answer different questions and should not be combined into one success rate. All are development results already seen by the team. Raw selected records are in `experiments/`, source hashes and selection details in `provenance/`, and historical code snapshots in `reference_code/`.
