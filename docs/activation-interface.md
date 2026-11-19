# Activation handoff

`nat` is B role replay; `flat` is C pasted. A and C+reminder are response
conditions only. Capture the last prompt token, before generating the answer,
at layers 0,4,8,12,16,20,24,28. Layer 0 is embedding output; layer k is block k
output. CPU float32 vectors have shape [3584], with n_tokens and position.
Both requests total 3840 vectors for 120 accepted histories (~52.5 MiB payload).

The validator checks unique case/query/condition/layer keys, completeness,
shape, finite values, CPU dtype, prompt/token correspondence, A/B input IDs,
and identical B/C last-token embeddings. A high later-layer probe score can
still reflect superficial serialization cues. This is not an authorship test.

## Development CUDA test

```bash
python confirmation/prepare_development_cases.py \
  --out runs/dev-cases.jsonl
python confirmation/run_batch.py \
  --development-smoke \
  --cases runs/dev-cases.jsonl \
  --model /path/to/pinned/model \
  --snapshot-manifest artifacts/model_snapshot.json \
  --out runs/cuda-smoke
```

The manifest must contain `all_weights_sha256_verified: true`, `revision` and
`snapshot`. Smoke outputs are tagged development and cannot be summarized as
confirmation. Synthetic interface files are likewise clearly marked.

## Freeze and run

```bash
python confirmation/freeze.py \
  --cases data/confirmation_v2/candidates.jsonl \
  --review-note "Generation authorized by Steven; v2 choices fixed before generation" \
  --out data/confirmation_v2/registration.json
# Commit registration and protocol before starting new inference.
python confirmation/run_batch.py \
  --cases data/confirmation_v2/candidates.jsonl \
  --registration data/confirmation_v2/registration.json \
  --model /path/to/pinned/model \
  --snapshot-manifest artifacts/model_snapshot.json \
  --out runs/confirmation-v2
python confirmation/score.py \
  --run runs/confirmation-v2
python confirmation/pack.py \
  --run runs/confirmation-v2 \
  --out runs/sishen-handoff-v2
```

Packing produces `activations.pt` for ordinary requests and
`activations_change.pt` for change requests. Each has exactly one nat/flat pair
per history/layer, compatible with a binary source-grouped probe:

```python
import torch
blob = torch.load('activations.pt', map_location='cpu', weights_only=True)
assert not blob['is_synthetic']
records = blob['records']
```

Never concatenate the two files and treat duplicated case IDs as new sources.
Raw shards retain request_type/query_id for provenance. CPU corrections to the
inherited probe, empirical-null implementation and human review remain team
handoff tasks under the fixed analysis plan in PREREGISTRATION.md.
