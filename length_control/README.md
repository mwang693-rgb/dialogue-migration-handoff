# length_control/: length-direction control v1

Question: is the format direction d just a generic "be more detailed" instruction direction? See the [result](../reports/length-control-v1/README.md).

**Registered files.** Hashes are in `data/length_control_v1/registration.json`, so do not edit these files.

| File | Role |
|---|---|
| `protocol.json` | Instruction wording, the length direction l, the split of d into d_par and d_perp, the statistics M, L_perp and S_len, and the decision rule. |
| `build.py` | Selects the units and copies the format vectors. CPU. |
| `freeze.py` | Writes the registration. |
| `run.py` | GPU. Phase 1 estimates l from "detailed" vs "brief" prompts. Phase 2 steers the 81 open units with d, d_perp, −d_perp, d_par and a same-norm length vector. |
| `summarize.py` | CPU summary and decision. |

```bash
python length_control/build.py --handoff <v2.1 handoff> --directions <directions.pt> --transfer-inputs <transfer inputs> --out <inputs>
python length_control/run.py --transfer-inputs <transfer inputs> --format-dirs <inputs>/format_dirs.pt --registration data/length_control_v1/registration.json --model <Qwen2.5-7B-Instruct> --out runs/run-length-control-v1
python length_control/summarize.py --run runs/run-length-control-v1
```
