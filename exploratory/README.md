# exploratory/

These are analyses that were not registered. They guide later designs, but they never enter a confirmatory claim.

| File | What it does | Result |
|---|---|---|
| `format_subspace.py` | Splits the native-minus-pasted difference into shared, request-type, history-specific and residual parts at every layer, and compares them with the vectors from earlier tasks. CPU. | [reports/format-subspace](../reports/format-subspace/README.md) |

```bash
python exploratory/format_subspace.py --handoff <v2.1 handoff> --v21 <v2.1 run> --v1-units <intervention run>/units --recovery-results <dialogue-migration-recovery>/results --out runs/analysis-format-subspace
```
