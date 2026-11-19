# Response-rule confirmation: version 2

Prepared November 19, 2026, before new confirmation generation. Version 1
remains in Git history. Steven authorized implementation and generation;
this does not claim that every teammate has signed off.

## 1. Question and primary contrast

Does flattening an established conversation reduce compliance with its response
rule? Primary contrast: mean paired binary compliance B minus C on ordinary
requests. A is an online-computation reference, not a quality gold standard.
Previous post-outcome-selected exploratory batches remain separate.

## 2. Model and decoding

Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`,
BF16, one RTX 5090, greedy argmax, no repetition penalty, seed 453.
System: `You are a helpful assistant.` Cap: 1024 new / 8192 input tokens.
Verify weight SHA256 before inference. Save library versions, actual input IDs,
generated IDs, termination and code hashes.

## 3. Rules, question source and establishment

Four MT-Eval rules: two responses separated by `******`, valid JSON, bullet
list, fixed ending `Is there anything else I can help with?`.
Every practice turn requests at most 200 words; length is secondary.

Version 2 changes the question source to Databricks Dolly 15k, revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. MT-Eval's 220 standalone
instruction questions cannot supply 480 disjoint new questions for the minimum
design, let alone reserves. Dolly reference answers are never supplied to Qwen
or used in scoring. This is an MT-Eval rule/checker adaptation on constructed
conversations, not an official MT-Eval aggregate score.

The frozen builder selects empty-context open_qa/brainstorming/creative_writing
questions, 8--70 words, excluding specified competing-format terms, exported
development-text matches, exact duplicates and token-set Jaccard above .85.
Seed 453 selects 960 questions; four consecutive questions form a bundle;
rules are round-robin. Source row IDs, dataset revision/hash and CC BY-SA 3.0
notice are in `data/confirmation_v2/`.

Qwen answers three practice turns online with a retained cache; there is no
acknowledgment-only turn. Admit only histories whose three replies pass the
category checker and end naturally without internal special tokens. Reject
before either test question; preserve every attempt/rejection. Process the
fixed candidate order until 30 histories per rule are accepted, with a fixed
reserve of 60 candidates per rule. Never select on N/P test outcomes or add
post-outcome candidates. Reserve exhaustion is a feasibility failure.

## 4. Conditions

|Condition|History input|
|---|---|
|A online|Retained history KV cache plus current request|
|B role replay|Same text as alternating user/assistant messages|
|C pasted|Same text with written role labels inside one user message|
|C + reminder|C with current applicable requirement explicitly repeated|

A/B full serialized IDs must match. Their computation is chunked differently;
BF16 can change greedy outputs. Report disagreement rather than attributing
it to authorship. B/C differ in serialization, role boundaries and length;
this does not isolate a hidden sense of ownership. Both requests independently
branch from the same frozen history.

## 5. Scoring, interval and review

Primary ordinary outcome: MT-Eval category checker passes AND natural assistant
EOS before the cap. Also report category-only, word count, truncation and
near-identical-two-response flags. This measures response form, not factual
quality, helpfulness or semantic distinctness.

Estimate the mean of 120 paired B-minus-C binary differences. Use 2000
source-history bootstrap resamples with seed 453 and linearly interpolated
2.5/97.5 percentiles. Report all conditions and rule strata; strata are
descriptive. Ordinary and changed-rule outcomes are never pooled.

Manual review: target 96 answers (three per rule x condition x request-type
cell), seed 453; add every truncated/empty or degenerate-two-response row,
so the final count can be larger. Two reviewers independently apply the
explicit current requirement with condition labels hidden. Report exact
agreement counts/denominators, missing reviews and both disagreement directions,
for reviewer/reviewer and each reviewer/checker. Manual semantic judgments do
not replace the primary surface checker. Review is a team CPU task and does
not block generation. No human review is claimed completed.

New dense KL is not required in this batch. A/B and B/C exact-text agreement
are secondary. This removes version 1's unimplemented dense-KL promise.

## 6. Rule-change request: one per history

Use the same held-out question twice, ordinary and changed-rule. Append:

|History rule|Exact changed-rule instruction|
|---|---|
|Two responses|For this reply only, give one response. Do not use the ****** separator or give a second version.|
|JSON|For this reply only, answer in plain prose, not JSON. Do not use a code block.|
|Bullet list|For this reply only, answer in one plain paragraph. Do not use a bulleted or numbered list.|
|Ending|For this reply only, omit the usual ending phrase "Is there anything else I can help with?". End with the answer itself.|

Append `Keep this reply at most 200 words.` to each. On change requests,
C + reminder repeats the NEW instruction. Change checks respectively require
nonempty text without separator; non-JSON text without a code fence; no
bullet/numbered lines or blank-paragraph separators; omission of exact ending.
Also require natural stop. These surface proxies have the review limitations
above and are reported separately from ordinary compliance.

## 7. Fixed sample sizes and budget

|Item|Planned|
|---|---:|
|Accepted histories|120 (30 per rule)|
|Candidate maximum|240 (60 per rule)|
|Requests per accepted history|2 (one ordinary, one change)|
|Conditions per request|4|
|Final answers|960|
|Changed-rule answers|480|
|Minimum practice replies for 120 immediately accepted histories|360|
|Maximum attempted practice replies|720|
|Maximum total free generations|1680|
|Raw activation records across both request types|3840|

At n=120, a worst-case normal-approximation standard error for the mean of
paired differences in [-1,1] is 1/sqrt(120), giving an approximate 95% half-width
.179. This is conservative planning, not a guaranteed detectable effect or
an equivalence margin. Precision depends on discordant pairs and admission.
Synthetic tensors test software; their intervals do not establish real power.

## 8. Continue/stop rule

Explore recovery only if the ordinary B-minus-C 95% interval has a POSITIVE
lower endpoint. A negative interval supports the opposite direction. An
interval containing zero is inconclusive; report its bounds, not 'no effect'.
No post-outcome sample extension. Probe analysis proceeds under any outcome.

Intervention outputs require a separate registration: reminder, history-
difference/probe directions, same-norm random, wrong-history and reverse
controls, fixed layer 16 / alpha 1. This runner does not automatically launch
interventions or claim that a classification direction is causal.

## 9. Raw activation export (required)

Capture B and C separately at the last prompt token BEFORE answering. Layers
0,4,8,12,16,20,24,28; 0 is token embedding output, k>0 is block k output before
model-level final normalization. CPU float32, one-dimensional length d:

```python
{"records": [{"case_id": str, "condition": "nat" | "flat",
              "layer": int, "vector": tensor_of_shape_d,
              "n_tokens": int, ...}]}
```

`nat` is B, not A. Include source/query IDs, request type, position and terminal
token ID. Pack ordinary pairs into `activations.pt` (1920 records) and change
pairs into `activations_change.pt` (1920), with `outputs.jsonl`. Thus each file
has one nat/flat pair per history/layer. Raw shards retain both request types.
The last prompt token is identical in B/C: layer-0 vectors MUST match. Layer 0
checks token identity; it does not exclude surface-format cues at later layers.

## 10. CPU probe plan and remaining implementation

Source-grouped five-fold outer CV; select hyperparameters AND layer on an
inner validation slice, not pooled outer-test accuracy. Report the full layer
curve separately from the nested selected-layer estimate. Seeds 0/1/2;
lr .001/.01; L2 .0001/.001/.01/.1; Adam/BCE, 200 epochs max, patience 10.
Standardize on training data only; token-length baseline uses the same folds.
Average seed correctness within source; bootstrap sources, not seeds.

Empirical H0: 199 independent source-wise nat/flat label swaps, consistent
across layers, repeating selection/evaluation. p=(1+null>=observed)/200.
Raw directions undo standardization: w_raw = D^-1 w, zero-filling dropped
columns. PCA is descriptive and cannot select the reported layer.

The inherited `analysis/probe.py` is retained for review. Its current CV path
selects a layer on pooled test accuracy and its direction does not undo
standardization. DO NOT use that summary or direction as the confirmatory
result. CPU corrections and the empirical null remain Sishen/Lang handoff
work, following this fixed plan; raw exports avoid a second GPU run. No probe
result is asserted now. Disclose later implementation fixes in the report.

## 11. Execution record

`confirmation/freeze.py` records protocol, candidates and source-code hashes
before confirmation. Commit registration before inference. Two OLD development
cases can test CUDA/export first, never counted as confirmation. Do not change
code/config inside a run. Preserve failures; implementation fixes require a
new registration/run directory. This protocol fills version 1's open items;
any later scientific change is a disclosed amendment.
