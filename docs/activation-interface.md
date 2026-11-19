# Files for Sishen

The new runner is separate from the old results. It has four response conditions, but exports B/C activations under the labels your probe expects:

|Response condition|Activation label|
|---|---|
|A, online|Not part of the binary probe|
|B, role replay|`nat`|
|C, pasted|`flat`|
|C + reminder|Not part of the binary probe|

For a 120-history batch with an ordinary and a change request per history, the output plan is 960 final replies and 3,840 vectors. Every vector has shape `[3584]` on Qwen2.5-7B; eight layers are stored together. The 120 shards contain about 52.5 MiB of vector payload, plus tensor metadata.

```python
import torch
from pathlib import Path

records = []
for path in sorted(Path("run/activations").glob("*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert not payload["is_synthetic"]
    records.extend(payload["records"])

ordinary = [r for r in records if r["request_type"] == "ordinary"]
# Split by source_id before fitting scalers or probes.
# Use (case_id, query_id, condition, layer), not case_id alone, as the row key.
```

The export preserves the raw nat/flat vectors, including layer 0 and `n_tokens`. The validator checks completeness, unique keys, shape, finite values, CPU float32, and correspondence to the saved prompts. Files made by the interface test have `is_synthetic=True`; they must not be treated as experimental results.

## CPU interface check

In an environment with PyTorch:

```bash
python confirmation/make_fake_data.py \
  --out runs/interface-smoke \
  --sources 16 \
  --hidden-size 3584

python confirmation/score.py --validate-old
```

This creates real `.pt` files and mock `outputs.jsonl`, then validates the contract. It does not load a language model or use the GPU.

## Real run, after protocol review and a development smoke test

First prepare two **old** cases and smoke-test the actual exporter on CUDA:

```bash
python confirmation/prepare_development_cases.py \
  --out runs/development-cases.jsonl
python confirmation/run_batch.py \
  --development-smoke \
  --cases runs/development-cases.jsonl \
  --model /path/to/pinned/Qwen2.5-7B-Instruct \
  --out runs/cuda-smoke
```

These files are marked `development_smoke` and are rejected by the confirmation behavior summary. After the team reviews the real source list and protocol, change protocol `status` to `reviewed`, commit, then register:

```bash
python confirmation/freeze.py \
  --cases /path/to/new-cases.jsonl \
  --review-note "Reviewed protocol and source list with the team" \
  --out runs/registration.json

python confirmation/run_batch.py \
  --cases /path/to/new-cases.jsonl \
  --registration runs/registration.json \
  --model /path/to/pinned/Qwen2.5-7B-Instruct \
  --out runs/confirmation-v1
```

Case records contain `source_id`, `case_id`, three `history_questions`, one final `question`, and `provenance`. No reference assistant answers are needed. There is deliberately no automatic reuse of the old eight histories as new confirmation data. Real generation has not been run with this new exporter yet.

To merge validated shards for Sishen:

```bash
python confirmation/pack.py \
  --run runs/confirmation-v1 \
  --out runs/sishen-handoff
```

This creates one `activations.pt`, copies `outputs.jsonl` and metadata, and writes file hashes. Packing fake interface files requires the explicit `--allow-synthetic` flag.
