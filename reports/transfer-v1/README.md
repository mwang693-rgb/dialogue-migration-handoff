# Transfer v1: does the length effect hold on unseen histories and tasks?

Model Qwen2.5-7B-Instruct, BF16, greedy decoding, one RTX 4090. Registered in commit `c8ef904` before any generation ([registration](../../docs/registrations/transfer-v1.md), [protocol](../../transfer/protocol.json)). Full numbers: [summary.json](summary.json).

## What was done

- **Units:** 375 native/pasted prompt pairs from 23 earlier exploratory runs in the sibling `dialogue-migration-recovery` workspace. No history overlaps v2.1. The units are grouped by task type, fixed before the run:

  | Group | Units | Histories |
  |---|---:|---:|
  | short | 230 | 138 |
  | open | 81 | 67 |
  | structured | 20 | 20 |
  | near-domain MT-Eval | 26 | 8 |
  | external-authored | 18 | 18 |

- **Vector:** the v2.1 layer-16 mean direction, unchanged and not refit. Each unit also gets one same-norm random vector.
- **Site and conditions:** as in intervention v1 (block 16 output, alpha 1). The conditions are B, C, C+mean, C−mean, C+random and B−mean.
- **Uncertainty:** history-cluster bootstrap with 4000 resamples. There are two primaries, so each uses a 97.5% interval.

## Registered results

**P1, length on open-ended tasks: the effect transfers.**

| Statistic | Point | 97.5% interval |
|---|---:|---:|
| L_sign = mean log n(C+mean) − log n(C−mean) | +0.355 | [+0.256, +0.458] |
| L_random = mean log n(C+mean) − log n(C+random) | +0.177 | [+0.083, +0.308] |

Mean answer tokens in the open group:

| B | C | C+mean | C−mean | C+random |
|---:|---:|---:|---:|---:|
| 431 | 360 | 438 | 295 | 367 |

- Adding the direction takes pasted answers to native length; subtracting it shortens them.
- A random vector of the same size does neither.
- 66/81 units move in the predicted direction, and no output degenerated.

**P2, KL specificity on short + open + structured: inconclusive.**

| Index | Point | 97.5% interval |
|---|---:|---:|
| S_random = (KL(C+random) − KL(C+mean)) / sum | +0.010 | [−0.048, +0.069] |
| S_sign = (KL(C−mean) − KL(C+mean)) / sum | +0.139 | [+0.065, +0.214] |

+mean is closer to native than −mean, but it is not reliably closer than random.

## By group (descriptive)

| Group | L_sign | L_random | S_random | S_sign |
|---|---:|---:|---:|---:|
| short | +0.088 | +0.037 | −0.028 | +0.117 |
| open | +0.355 | +0.177 | +0.141 | +0.237 |
| structured | −0.021 | −0.020 | −0.085 | −0.008 |
| near (MT-Eval) | +0.324 | +0.259 | +0.039 | +0.088 |
| external-authored | +0.044 | −0.017 | +0.237 | +0.352 |

- The length effect appears where native and pasted answers already differ in length.
- Structured outputs, whose length is fixed by the task, do not move.

## Note on an interim look

Partial results (298 units) were looked at while the run was still going. Nothing in the run was changed afterwards.
