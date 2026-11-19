# Analysis code for the confirmatory batch

Frozen at pre-registration. See `PREREGISTRATION.md`.

| File | Purpose |
|---|---|
| `splits.py` | source-level splitting; source-level and paired bootstrap |
| `scoring.py` | rule compliance, quality flags, manual-review sampling |
| `probe.py` | layer-wise linear probes, controls, layer selection |

## Install

    pip install numpy torch emoji nltk langdetect

## Score an existing batch

    python scoring.py --repo /path/to/dialogue-migration-handoff

Validated against the exploratory `02_strategy` batch: reproduces B 16/16,
C 15/16, A 15/16 as reported in the midway report.

## Probe

Self-test with synthetic activations, no GPU and no real data needed:

    python probe.py --selftest

Once the GPU side exports activations in the format described in
`PREREGISTRATION.md` §9 and in the `probe.py` docstring:

    python probe.py --activations path/to/activations.pt --cv 5 --out probe_results.json

Neither the hidden size `d` nor the layer set is hard-coded; both are read
from the file.
