# transfer/: format-direction transfer v1

Question: does the intervention-v1 length pattern hold on 375 native/pasted pairs from earlier, unrelated experiments? The vector is fixed and not refit. See the [result](../reports/transfer-v1/README.md) and the [registration write-up](../docs/registrations/transfer-v1.md).

**Registered files.** Hashes are in `data/transfer_v1/registration.json`, so do not edit these files.

| File | Role |
|---|---|
| `protocol.json` | Groups, conditions, P1 (length) and P2 (KL specificity), and the decision rules. |
| `build_units.py` | Builds `units.jsonl` and `vectors.pt`. It takes saved prompt IDs from the sibling `dialogue-migration-recovery` runs, or rebuilds prompts with each run's own serializer. CPU plus tokenizer. |
| `freeze.py` | Writes the registration. |
| `run.py` | GPU run. It reuses `intervention/steer.py`. |
| `summarize.py` | History-cluster bootstrap (`cluster_boot`), the statistics (`stats`) and the decisions. Later studies reuse `stats`. |

The frozen inputs (`units.jsonl`, `vectors.pt`) are not in git. Their hashes and counts are in `data/transfer_v1/manifest.json`.

```bash
python transfer/build_units.py --recovery <dialogue-migration-recovery> --v21 <v2.1 run> --directions <directions.pt> --tokenizer <Qwen tokenizer> --out <inputs>
python transfer/run.py --inputs <inputs> --registration data/transfer_v1/registration.json --model <Qwen2.5-7B-Instruct> --out runs/run-transfer-v1
python transfer/summarize.py --run runs/run-transfer-v1
```

`--smoke N` on `run.py` runs a tiny-model CPU check without a registration.
