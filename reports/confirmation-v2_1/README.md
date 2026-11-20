# Response-rule confirmation results

Run: confirmation-v2_1. Checkpoint: Qwen2.5-7B-Instruct, BF16, one RTX 5090.
Protocol and generation code were frozen before the replacement run. The technical rerun amendment is recorded in [PREREGISTRATION.md](../../PREREGISTRATION.md).

## Main finding

The ordinary-request result did not support the preregistered hypothesis that pasting reduces established-rule compliance. B passed 112/120 and C passed 115/120. The paired B-minus-C difference was -2.5 percentage points; the source-bootstrap 95% interval was [-5.83, 0.00] points. This is not evidence for a positive recovery target, and it does not establish equivalence or general superiority of C. No activation intervention was run.

|Condition|Ordinary requests|Changed-rule requests|
|---|---:|---:|
|A: online|112/120|118/120|
|B: role replay|112/120|117/120|
|C: pasted|115/120|112/120|
|C + reminder|113/120|106/120|

Changed-rule requests are a separate outcome. Compliance means following the new instruction, not persisting with the old rule. These counts are descriptive and do not replace the ordinary-request primary comparison.

## Data and validation

- 120 accepted target-model histories, 30 per rule; three practice replies per accepted history.
- 132 candidates attempted; 120 accepted, 3 rejected for token replay mismatch, and 9 for rule establishment. The other 108 reserve candidates were unused. There were 381 practice replies.
- 960 final answers and 3840 raw activation records across eight layers (0,4,8,12,16,20,24,28).
- 0 final answers reached the output cap. Generation time: 31.45 minutes.
- Archive SHA256, each handoff file hash, tensor schema, layer completeness, prompt correspondence and the behavior summary were verified locally.
- A/B compliance agreement across both request types: 98.75%. Identical prompt IDs do not guarantee bitwise-identical BF16 chunked/full computation.

## Text equality (secondary)

|Request|A/B exact text|B/C exact text|
|---|---:|---:|
|ordinary|67/120|6/120|
|change|25/120|1/120|

Text equality is not correctness. No new KL or semantic quality result is claimed.

## Scope and remaining work

This is a constructed task using public Dolly questions, Qwen-generated history replies, and four MT-Eval rule checkers. It is not an official benchmark aggregate. Only established, losslessly replayable histories were admitted; conditioning on admission limits generalization. Reference dataset assistant answers were not used.

The first attempt stopped while constructing candidate 2 after completing one source. The same text had different generated and re-encoded BPE segmentations. Its complete output/code/config was preserved. The replacement rejects such histories before test requests and records both token streams. The candidate pool, quotas and outcome definitions were unchanged. One source had prior outputs, so the replacement is not described as wholly unseen.

Human review, semantic checks and probe fitting remain unfinished. The inherited probe code has the selection/weight-conversion issues documented in the preregistration; its existing summary must not be used as confirmatory analysis. The negative behavioral gate does not prevent a properly evaluated source-grouped probe.

## Download the data

[GitHub Release: confirmation-v2.1](https://github.com/mwang693-rgb/dialogue-migration-handoff/releases/tag/confirmation-v2.1)

- `activations.pt`: ordinary-request B/C pairs, 1920 records.
- `activations_change.pt`: changed-rule B/C pairs, 1920 records.
- `outputs.jsonl`: all four conditions, 960 answers.
- `confirmation-v2_1.tar.gz`: complete run, both activation exports, histories, frozen source code, configuration, registration and dataset attribution.
- `SHA256SUMS.txt`: checksums for the downloadable files.
- `HANDOFF.md`: loading instructions and remaining CPU work.

The standalone activation/output files are also contained in the complete archive; download either the three working files or the archive. Source conversations are the unit of splitting and uncertainty, not output rows. Manual ratings and probe fitting remain pending.
