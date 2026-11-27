# Intervention v1: does the format direction act on the answer?

Model Qwen2.5-7B-Instruct, BF16, greedy decoding, one RTX 4090. Registered in commit `6675ac8` before any generation ([registration](../../docs/registrations/intervention-v1.md), [protocol](../../intervention/protocol.json)). Full numbers: [summary.json](summary.json).

## What was done

- **Data:** the 120 accepted v2.1 histories (30 per rule), each with an ordinary and a change request.
- **Direction d:** the mean of B − C (native minus pasted) at the last prompt token, at the output of block 16. It is cross-fitted over 5 source folds, so no history's own pair enters its own vector.
- **Edit:** add the vector at alpha 1 (its natural size, no tuning) at the last prompt token and at every generated token.
- **Conditions:** B, C, C+mean, C−mean, C+random (same norm), C+own pair delta, C+wrong-history delta and B−mean. That gives 1920 generations.
- **Primary outcome (ordinary requests):** teacher-forced KL from B along B's answer.
  - Gap closed: G = 1 − KL(B‖C+mean) / KL(B‖C).
  - Specificity: Δ = KL(B‖C+random) − KL(B‖C+mean).
  - Uncertainty comes from a source bootstrap.

## Registered result: inconclusive

| Quantity | Point | 95% interval |
|---|---:|---:|
| Gap closed G | −0.63 | [−0.90, −0.40] |
| Specificity Δ (random − mean) | +0.062 | [+0.042, +0.085] |

Adding the direction made the answer distribution **further** from native, not closer. The registered rule needs both lower bounds above 0 to count as an effect. Calling the direction inert would need G inside ±0.10. Neither holds, so the label is *inconclusive*.

Mean KL from B along B's answer, in nats per token:

| C | C+mean | C+random | C−mean | C+own | C+wrong |
|---:|---:|---:|---:|---:|---:|
| 0.099 | 0.162 | 0.224 | 0.270 | 0.213 | 0.241 |

Every edit hurts, but +mean hurts least and −mean most. The direction is not arbitrary: its sign matters.

## Post hoc pattern (not confirmatory): answer length

Mean words on ordinary requests:

| B | C | C+mean | C−mean | C+random | B−mean |
|---:|---:|---:|---:|---:|---:|
| 72.5 | 58.7 | 72.8 | 50.4 | 62.5 | 59.5 |

- Pasted answers are shorter than native ones.
- Adding the direction brings C up to native length, and subtracting it from B brings B down to pasted length.
- A random vector of the same norm barely moves length.
- Change requests show the same pattern (B 125.8, C 108.2, C+mean 118.3, C−mean 94.9), even though the vector was built from ordinary requests only.
- Rule compliance is unchanged (B 111, C 115, C+mean 114 of 120).

This pattern was seen after the run, so it was tested out of sample in [transfer v1](../transfer-v1/README.md).

## Other checks

- **Cross-format:** the same direction appears when the history is pasted with Q/A labels (cos 0.96 with the main direction at layer 16) or as JSON (cos 0.86). Paired sign is 100% in both cases.
- **First-token KL:** relative to C, the source's own delta closes 65% of the gap and the mean closes 23%.
- **GPU reproduction:** this GPU reproduces the v2.1 token streams exactly for 67/120 (B) and 70/120 (C) histories. Exact-text agreement is therefore noisy across GPUs, which is why baselines were regenerated in the same run.

## Not claimed

Authorship, persona, other models, or general harm from migrating a conversation.
