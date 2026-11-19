# Conversation format experiments

We tested whether copying a conversation into one user message changes the next reply. The setup is settled; we are still choosing the task and the outcome measure for the course project.

This repository collects three small experiments from October 10–14, 2026. Start with the [results](reports/results.md), or read the [examples](reports/examples.md) to see the actual replies. The [experiment notes](docs/experiment-notes.md) cover the other things we tried. A [project outline](docs/project-outline.md) is included for the team to work from.

The next-batch code is under [confirmation/](confirmation/README.md). It adds the four-condition run and separate nat/flat activation exports. Its protocol is a draft pending the new source list and team review; no new model results are included.

|Experiment|What we measured|What we found|
|---|---|---|
|Editing books, code, and tables|Requested edits and preservation of unchanged content|Out of 24 requests: replay passed 15, pasted history 17, a continuation hint 19, and a history-difference vector 11.|
|MT-Eval response rules|Whether a rule used in earlier replies continued to hold|Two-response answers scored 4/4 online and in replay, versus 3/4 when pasted. Further questions scored 9/10 in both replay and pasted conditions.|
|Contractions in casual replies|Occurrences and density of 30 contraction forms|Pasted replies had higher density in all six old histories, but only two of four new histories. We stopped before steering on the new batch.|

## The comparison

Qwen generates the history one turn at a time. We save that history and test each follow-up independently:

- **A — online:** continue with the cache from actual generation.
- **B — role replay:** feed the same text again as alternating user and assistant messages.
- **C — pasted:** put the same text, with written role labels, inside one user message.

B/C is the main format comparison. A/B checks what changes when the history is recomputed. Some earlier runs only have B/C, labeled N/P in their saved files. In particular, the editing batch and the later response-rule extension have no online A results.

The model was Qwen2.5-7B-Instruct, using BF16 and greedy decoding on one RTX 5090. Public datasets supplied user questions and rules; Qwen generated the assistant history.

## Rebuild the tables

Python 3.10+ is enough. No model, API key, or extra package is needed.

```bash
python scripts/build_report.py
python -m unittest discover -s tests -v
python scripts/verify_release.py
```

This checks the saved answers and regenerates `reports/results.md`, `reports/examples.md`, and `reports/summary.json`. It also checks means from the saved per-token KL records. Recomputing the underlying vocabulary distributions would require the model.

## Run a new batch

The optional runner regenerates histories, then compares A/B/C. It writes to a new directory so the saved experiments stay separate.

```bash
pip install -r requirements-gpu.txt
python scripts/run_new_threeway.py \
  --experiment strategy \
  --model /path/to/local/Qwen2.5-7B-Instruct \
  --max-cases 1 \
  --out runs/strategy-new
```

`--experiment style` is also supported. It needs a local model and CUDA. The new wrapper has passed syntax and CLI checks, but we have not run it on a GPU for this release. Its history generation uses the original `OnlineEngine` code. New histories can differ from the saved ones and should be scored separately.

## Files

- `experiments/`: selected histories, answers, token IDs, scores, and original configs.
- `reference_code/`: the original experiment scripts and their local dependencies. They still refer to the old workspace layout; use them to inspect the methods rather than running their `main()` functions directly.
- `handoff.py`: CPU scoring and checks used by the report builder.
- `provenance/` and `selection.json`: source hashes and the selection made for this repository.
- `third_party/mt_eval/`: the source checker, its license, and attribution.

There are 364 final answers and 60 distribution records. These are repeated measurements on a much smaller set of histories, not 424 independent samples. We chose these batches after seeing the results because they were easier to explain; the other exploratory runs are summarized in the notes.

The original replies and scores are unchanged. AST comparisons accommodate the empty `type_params` field added in Python 3.12 and explicitly retain empty fields when dumping ASTs on Python 3.13. Nonempty type parameters and code strings still count. Third-party material retains its original license; see [sources](THIRD_PARTY_NOTICES.md).
