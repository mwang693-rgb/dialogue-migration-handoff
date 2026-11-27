# Length control v1: is the format direction just a "be more detailed" knob?

Model Qwen2.5-7B-Instruct, BF16, greedy decoding, one RTX 4090. Registered in commit `973f33c` before any generation ([protocol](../../length_control/protocol.json), [registration](../../data/length_control_v1/registration.json)). Full numbers: [summary.json](summary.json).

## What was done

**Length direction l.** For each of the 294 non-open transfer units, take the same prompt twice:

- once with "Please give a detailed, thorough answer." appended to the last user turn;
- once with "Please keep your answer brief." appended instead.

This follows the instruction-direction method of Stolfo et al. (ICLR 2025).

Average the difference in the layer-16 state at the last prompt token over the 588 resulting pairs. This gives a generic instruction-length direction built without the 81 open units.

**Decomposition of the format direction d (norm 21.36):**

- d_par: the part of d along l (norm 1.92).
- d_perp: the rest of d (norm 21.28).
- len: a pure length vector, l rescaled to d's norm.

The cosine between d and l is **0.09** at layer 16, and |cos| ≤ 0.25 at every layer.

**Test units:** the 81 open units from transfer v1, using the same site and conditions as before.

## Registered result: `not_just_length`

| Statistic (95% intervals) | Point | Interval | Meaning |
|---|---:|---:|---|
| M = log n(C+len) − log n(C) | +0.81 | [+0.66, +0.96] | the length control works, so the test is valid |
| L_perp = log n(C+d_perp) − log n(C−d_perp) | +0.20 | [+0.14, +0.27] | d still lengthens answers with its length part removed |
| S_len = (KL(C+len) − KL(C+d)) / sum | +0.56 | [+0.52, +0.60] | C+d is far closer to native than C+len |

Mean answer tokens:

| B | C | C+d | C+d_perp | C−d_perp | C+d_par | C+len |
|---:|---:|---:|---:|---:|---:|---:|
| 431 | 360 | 438 | 407 | 319 | 387 | 744 |

- The pure length vector overshoots: 40/81 answers hit the token cap.
- The format direction lands at native length.
- KL to native is 0.074 nats/token for C+d against 0.235 for C+len.

## Careful reading

d's log-length shift is +0.198. Of that, d_perp alone gives +0.118 and d_par alone gives +0.089. So the 9%-norm component along l carries roughly 45% of the lengthening. The accurate statement is: **d is not the generic instruction-length direction, but it overlaps it a little, and that overlap does part of the work.**

## Reproducibility

The B, C and C+d token streams are identical to transfer v1 on all 81 units (same GPU).
