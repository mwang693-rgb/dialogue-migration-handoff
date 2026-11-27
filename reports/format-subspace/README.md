# Format subspace across layers (exploratory)

Not registered. Nothing here enters a confirmatory claim. Code: [exploratory/format_subspace.py](../../exploratory/format_subspace.py). Full numbers: [summary.json](summary.json).

## Question

The native-minus-pasted difference D (one per history and request type) mixes several things. How much of it is:

- **shared** across all histories (the "format direction" g),
- specific to the **request type** (q: ordinary vs change),
- specific to the **history** (a_s),
- **residual**?

The split follows the shared-plus-specific decomposition of Bürger et al. (NeurIPS 2024, *Truth is Universal*). Data: the 120 v2.1 histories × 2 request types, Qwen2.5-7B-Instruct, last prompt token.

## Share of D's squared norm

| Layer | Shared g | Request q | History a_s | Residual |
|---:|---:|---:|---:|---:|
| 4 | 0.87 | 0.03 | 0.08 | 0.02 |
| 8 | 0.67 | 0.08 | 0.19 | 0.05 |
| 12 | 0.59 | 0.10 | 0.22 | 0.10 |
| 16 | 0.48 | 0.10 | 0.30 | 0.12 |
| 20 | 0.30 | 0.07 | 0.42 | 0.21 |
| 24 | 0.17 | 0.04 | 0.51 | 0.28 |
| 28 | 0.12 | 0.03 | 0.54 | 0.31 |

The shared format direction dominates shallow layers and fades with depth. The history-specific part grows with depth.

## Other observations

- **The request-type part q is real but does not transfer.** It is significant and stable across half-splits (cos 0.98). Once it is orthogonalised against g, it has about zero cosine with earlier tasks.
- **The history-specific part is mostly the rule.** Its first principal component has η² = 0.82 with the rule.
- **Depth explains most of the earlier spread.** The earlier cross-task cosines (0.14–0.53) mostly reflect layer depth. Cosine with g is 0.67 at layer 8 and 0.35 at layer 16, then falls further. 132 of 133 older task vectors have a positive projection on g.
- **No per-history link to behaviour within a rule.** At layer 16, a history's projection on g does not predict its length gap, KL or steering gain (ρ ≈ 0 within rule).

## Why it matters

This analysis suggests the cleanest place to steer is a shallow layer, around layer 8 in Qwen. It motivated the registered shallow-depth (k = 2) follow-up in the [multi-model replication](../../replication/README.md).
