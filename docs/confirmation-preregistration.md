# Response-rule batch: registration draft

November 19, 2026. **Draft, not yet frozen.** The shared repository contained no separate preregistration when work began. This document fills in the proposed counts and export interface; the team still needs to review the case list and measurement decisions. No confirmation outputs have been generated.

## 1. Question

Does pasting a conversation reduce compliance with a response rule already used in that conversation? Can a reminder recover compliance? A layer-wise format probe will use the same batch, whether or not a behavioral gap appears.

## 2. Model

Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16, eval mode, greedy decoding, repetition penalty 1.0. Local model files are hashed before generation; the team must check that they correspond to the intended snapshot. Use one fixed system message: `You are a helpful assistant.` Set a 512-token limit for each history and final reply. Record every truncation; a truncated final reply fails the complete-response metric.

## 3. Cases

The proposed budget is 120 source histories. Each has three history questions and one subsequent question, all chosen before generation. The final question has two request variants, tested independently. Qwen generates all history assistant turns. Public reference assistant replies are not supplied as history.

**The new source list has not been supplied or selected yet.** Do not refill this batch with the eight old MT-Eval histories, their remaining questions, paraphrases counted as independent sources, or examples selected because B previously succeeded and C failed. `cases.jsonl` must identify source/provenance and any shared-source or near-duplicate grouping. This draft assumes one case per source. If material cannot meet that assumption, revise counts and split logic before registration.

## 4. Four conditions

- A: continue from the real history cache.
- B: replay the same history with user/assistant roles.
- C: put the history and current request inside one user message, using the existing pasted wrapper.
- C + reminder: use C, then explicitly repeat the requirement that applies to this reply. On a change request, repeat the **new** one-response requirement, not the old two-response rule.

All final branches start from independent copies of the same frozen history. Check A/B input IDs for exact equality. B/C is the format comparison; A/B is a compute-route control.

## 5. Primary measure

The initial rule is:

> For this conversation, give two different responses separated by exactly six asterisks: ******. Keep the entire reply at most 200 words.

The code uses `ordinary_rule` in `confirmation/protocol.json` verbatim, preceded by `For this conversation, ` on the first history user turn. An ordinary reply passes when it has exactly one `******` separator, two nonempty unequal strings, at most 200 whitespace-delimited words, and a natural stop. A change reply passes when nonempty, has no `******`, respects the same word limit, and stops naturally.

This is a surface-form checker. It does not establish factual correctness or semantic distinctness. Preserve form, length, and completeness as separate fields. Main behavior estimates include all valid histories; report established-history results separately rather than silently excluding failures.

Two reviewers will independently inspect 12 source histories, both request variants, and all four conditions: **96 final replies each**. Select sources using seed 453 before reading their outputs, and hide condition labels. Reviewers mark pass/fail/unclear against the actual request. Report raw inter-reviewer agreement, clear-label agreement with its coverage, and checker agreement after adjudication. Unresolved items keep the audit incomplete. Lang should review this protocol before freezing.

## 6. Change-rule requests

Each history has **one ordinary request and one change request**, sharing the same question body. The ordinary variant adds no new style or format instruction. Append this exact text to make the change variant:

> For this reply only, replace our two-response rule: give one response, without the ****** separator. Keep the entire reply at most 200 words.

Do not feed one variant's answer to the other. The change subset prevents us from calling rigid adherence to an obsolete rule successful continuity.

## 7. Sample counts

|Item|Planned count|
|---|---:|
|Source histories|120|
|Generated history replies per source|3|
|Test requests per history|2: one ordinary, one change|
|Conditions per request|4|
|Final replies|960|
|Final replies to change requests|480|
|History-generation replies|360|
|Total free generations before any conditional intervention|1,320|
|Saved activation vectors, B/C only|3,840|

These are a proposed course-project budget, not a power calculation. At least 96 valid histories are required for the planned behavior gate. Invalid histories are logged with their reason and not replaced after seeing final outcomes. CPU probe splits use source IDs, never activation rows or question variants.

## 8. Continue/stop decision

The following are proposed working thresholds for Lang to review, not established statistical conventions. Run no activation intervention unless all hold:

1. At least 80% of planned histories are valid, and at least 80% of valid histories follow the joint rule in all three history replies.
2. On ordinary requests, B−C compliance is at least 10 percentage points and the paired source-bootstrap 95% interval has a lower endpoint above zero.
3. A/B agree on binary behavior scores on at least 95% of matched requests.
4. C passes at least 80% of change requests.
5. The human audit is complete and checker agreement with adjudicated labels is at least 90%.

The 10-point threshold avoids expanding a tiny observed difference into a method project; the other thresholds screen history validity, recomputation problems, and checker errors. They do not turn this budget into a population-wide significance claim. If the gate fails, report the behavior result and still train the format probe. Do not search layers, change metrics, or resample cases to obtain a pass.

An activation comparison would be registered separately before intervention outputs: C, explicit reminder, history-difference direction, same-norm random, and reverse. The reminder already has a baseline result in this batch. Do not automatically treat a probe's normal vector as the best intervention vector.

## 9. Activation export

Save a `.pt` shard per valid history. The container is `{"schema_version": 1, "is_synthetic": false, "hidden_size": d, "records": [...]}`. Each record has:

```python
{
    "source_id": str,
    "case_id": str,
    "query_id": "ordinary" or "change",
    "request_type": "ordinary" or "change",
    "condition": "nat" or "flat",  # B and C, respectively
    "run_condition": "B" or "C",
    "layer": int,
    "vector": torch.Tensor,  # CPU float32, shape [d], not a difference
    "n_tokens": int,         # full prompt length
    "position": int,         # n_tokens - 1
    "terminal_token_id": int,
}
```

For Qwen2.5-7B, save layers **0, 4, 8, 12, 16, 20, 24, 28**. Layer 0 is the token-embedding output; layers 1–28 are post-block outputs before the final norm. Capture the last prompt position before generating any answer token. This is the same functional readout position used for the difference direction, but the actual question/cue must also be recorded when comparing directions.

Both request variants are exported. The primary binary probe uses **ordinary B/C only**, one pair per source. Keep change-request vectors for a separate analysis. A and C + reminder have recorded replies but do not add examples to the binary nat/flat probe.

At this readout position, B/C normally end in the **same token**, so their token-embedding vectors should be identical even though the preceding sequences differ. Layer 0 must be included and checked, but chance performance there does **not** eliminate role-token or length cues at later layers. Use the saved prompt lengths for the separate length baseline.

The interface test uses d=3,584 and eight layers with synthetic tensors. A real CUDA smoke run is still required before full generation. Hand over `activations/*.pt`, `outputs.jsonl`, histories, cases, protocol, registration, and validation output. A difference tensor alone is insufficient.

## Before generation

Review the proposed sample counts and scoring with the team, provide the new case list, complete a real smoke run using old development cases, and commit the reviewed protocol/code/case hashes. Then use `confirmation/freeze.py` to register the actual inputs. The file being present in Git is not by itself evidence that the case list was preregistered.
