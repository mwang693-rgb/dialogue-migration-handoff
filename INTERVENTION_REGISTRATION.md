# Format-direction intervention v1 (registered before generation)

The confirmation batch found no behavioral B/C gap (ordinary compliance B 112/120, C 115/120), but an exploratory CPU check found that B and C activations separate along a shared direction. At layer 16 the mean nat−flat direction transfers across rules and request types. It also aligns with the saved N−P vectors of every earlier exploratory task (cos 0.14–0.53; random ≈ 0.01) and is near-orthogonal to the earlier own-vs-external authorship direction. Those checks are exploratory. This registration asks one causal question:

**Is that direction used when the model writes its answer, or is it inert?**

The full design is `intervention/protocol.json`, and the vectors are frozen in `directions.pt` (SHA256 recorded in `data/intervention_v1/registration.json`). In short:

- **Data:** the 120 accepted v2.1 histories, with ordinary and change requests. The B and C prompt IDs must equal the saved v2.1 input IDs.
- **Site:** output of block 16, at the last prompt token and every generated token. The edit adds an absolute vector (alpha 1).
- **Conditions (1920 generations):** B, C, C+mean, C+random (same norm), C−mean, C+own pair delta, C+wrong-history delta, B−mean. The mean direction is cross-fitted over 5 source folds, so no source's own pair enters its mean vector.
- **Primary outcome (ordinary requests):** teacher-forced KL from B along B's greedy answer. Gap closed G = 1 − KL(B‖C+mean)/KL(B‖C); specificity Δ = KL(B‖C+random) − KL(B‖C+mean). Uncertainty comes from a source bootstrap (2000 resamples).
- **Decision:** a direction-specific effect if the lower bounds of both G and Δ are above 0. Effectively inert if G's 95% interval lies inside [−0.10, 0.10]. Otherwise inconclusive.
- **Secondary or descriptive:** change requests; own/wrong/reverse/B−mean; exact text agreement; rule compliance; integrity; reproduction of v2.1 token streams on this GPU; and a forward-only cross-format check (Q/A labels, JSON history).

Baselines are regenerated in the same run because this GPU is not the one used for v2.1, and BF16 kernels differ. The not-claimed list in the protocol applies: no authorship, persona, cross-model or general migration-harm claims.
