# intervention/: format-direction intervention v1

Question: is the shared native-minus-pasted direction at layer 16 used by Qwen when it writes the answer, or is it inert? See the [result](../reports/intervention-v1/README.md) and the [registration write-up](../docs/registrations/intervention-v1.md).

**Registered files.** The SHA256 of each file is recorded in `data/intervention_v1/registration.json`, so do not edit them. A new design gets a new folder.

| File | Role |
|---|---|
| `protocol.json` | The design: site, conditions, outcomes and decision rule. |
| `directions.py` | Builds the cross-fitted mean and random vectors (`directions.pt`) from the v2.1 activations. CPU. |
| `freeze.py` | Writes the registration (hashes of protocol, vectors, inputs and code). |
| `steer.py` | Shared GPU helpers: the `Steer` hook (adds a vector at one block output), greedy `generate`, `teacher_forced_logprobs` and `kl_summary`. Later studies reuse it. |
| `run.py` | GPU run of every condition. It checks the registration hashes before it starts. |
| `summarize.py` | CPU summary that applies the registered decision rule. |

Pipeline:

```bash
python intervention/directions.py --handoff <v2.1 handoff> --run <v2.1 run> --out <inputs>
python intervention/freeze.py --directions <inputs>/directions.pt --v21 <v2.1 run> --review-note "..." --out data/intervention_v1/registration.json
python intervention/run.py --v21 <v2.1 run> --directions <inputs>/directions.pt --registration data/intervention_v1/registration.json --model <Qwen2.5-7B-Instruct> --out runs/run-intervention-v1
python intervention/summarize.py --run runs/run-intervention-v1 --handoff <v2.1 handoff>
```
