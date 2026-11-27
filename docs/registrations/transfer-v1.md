# Format-direction transfer v1 (registered before generation)

Intervention v1 pushed the v2.1 layer-16 format direction (mean nat − flat) into pasted prompts. Its registered gap-closing test failed: G = −0.63, labelled inconclusive. Two other patterns did show up:

- Answer length moved with the sign of the push. C + mean was longer, C − mean shorter, B − mean shortened B to C's length, and a random vector of the same norm did nothing.
- Teacher-forced KL toward B was smaller under +mean than under random or −mean.

The length pattern was post hoc, and both patterns come from one batch of 120 histories in a single instruction family. This registration asks whether they hold on histories and tasks that played no part in building the vector.

The design is `transfer/protocol.json`. The frozen inputs are `units.jsonl` and `vectors.pt`; their SHA256s are in `data/transfer_v1/registration.json`.

- **Units:** 375 native/pasted prompt pairs drawn from 23 earlier exploratory runs in `dialogue-migration-recovery`.
  - 267 use the old run's saved input IDs.
  - 108 are rebuilt with the old run's own serializer, and each matches the saved input token count.
  - No history overlaps v2.1.
  - Groups are fixed by task type:

    | Group | Units | Histories |
    |---|---|---|
    | short | 230 | 138 |
    | open | 81 | 67 |
    | structured | 20 | 20 |
    | near-domain MT-Eval | 26 | 8 |
    | external-authored | 18 | 18 |

- **Vector:** the v2.1 full-sample mean, unchanged, plus one same-norm random vector per unit. Nothing is refit.
- **Site and conditions:** as in v1, at the block-16 output (last prompt token and every generated token, alpha 1). Conditions are B, C, C+mean, C−mean, C+random and B−mean.
- **P1, length (open group):**
  - L_sign = mean log n(C+mean) − log n(C−mean).
  - L_random = mean log n(C+mean) − log n(C+random).
  - The effect transfers if both lower bounds are > 0. There is no effect if L_sign's interval lies inside ±0.05.
- **P2, KL specificity (short + open + structured):**
  - Uses the scale-free index (KL_B(X) − KL_B(C+mean)) / (KL_B(X) + KL_B(C+mean)), for X = C+random and X = C−mean.
  - Specificity transfers if both lower bounds are > 0. There is none if both intervals lie inside ±0.05.
- **Uncertainty:** history-cluster bootstrap, 4000 resamples, 97.5% intervals (two primaries, Bonferroni).
- **Secondary:** per-group and per-pool statistics, B-vs-C length, gap closed, first-token KL, exact agreement and termination. Task correctness with each pool's scorer is deferred to a later CPU pass and is descriptive.

The v1 reference values computed with the same formulas are listed in the protocol and are descriptive. Not claimed: other models or layers, authorship, task accuracy, or answer quality.
