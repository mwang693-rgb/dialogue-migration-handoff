# Native vs pasted conversation history: a format direction inside the model

A conversation can be carried over in two ways:

- **B, native:** replay it as real chat turns.
- **C, pasted:** paste its transcript ("User: … / Assistant: …") into a single user message.

This repository asks whether the model treats the two differently, and whether that difference is visible inside it. The main model is **Qwen2.5-7B-Instruct** (BF16, greedy decoding). A replication on Llama-3.1-8B-Instruct and Gemma-2-9B-it is running.

Every confirmatory step was registered (protocol, inputs and code hashed and committed) before any of its outputs existed. Exploratory results are labelled as exploratory.

## Findings so far (Qwen2.5-7B-Instruct)

1. **Behaviour barely changes.** Pasting did not reduce rule compliance: B passed 112/120 and C 115/120 ([confirmation v2.1](reports/confirmation-v2_1/README.md), registered).
2. **The model clearly knows the difference.**
   - B and C states at the last prompt token are linearly separable (100% from layer 4 to layer 24).
   - The separation runs along one shared direction *d* = mean(B − C). It holds across rules and request types, and for other paste formats: Q/A labels (cos 0.96) and JSON (cos 0.86).
   - Separability comes from an exploratory probe. The paste-format check was a secondary analysis in [intervention v1](reports/intervention-v1/README.md).
3. **Adding d does not make pasted equal native.** At layer 16 with alpha 1 (its natural size, no tuning), the answer distribution moved further from native (gap closed G = −0.63; registered label *inconclusive*).
4. **d controls answer length, out of sample.** Pasted answers are shorter. Adding d restores native length, subtracting d shortens answers, and a same-size random vector does neither. On 81 open-ended units from unrelated earlier experiments, L_sign = +0.355 [+0.256, +0.458] ([transfer v1](reports/transfer-v1/README.md), registered).
5. **d is not simply a "be detailed" instruction direction** ([length control v1](reports/length-control-v1/README.md), registered).
   - Its cosine with that direction is 0.09.
   - With the overlap removed, d still lengthens answers.
   - It lands much closer to native than a same-size length push: KL 0.074 against 0.235.
   - The small overlap still carries about 45% of the lengthening.
6. **The shared part lives in shallow layers.** It is 87% of the B − C difference at layer 4, 48% at layer 16 and 12% at layer 28 ([format subspace](reports/format-subspace/README.md), exploratory).

## The study chain

| # | Study | Question | Kind | Code | Result |
|---|---|---|---|---|---|
| 0 | October exploration | Which tasks show a B/C difference at all? | exploratory | `reference_code/`, `experiments/` | [reports/october-exploration](reports/october-exploration/results.md) |
| 1 | Confirmation v2.1 | Does pasting reduce rule compliance? | registered | [confirmation/](confirmation/README.md), [analysis/](analysis/README.md) | [report](reports/confirmation-v2_1/README.md) |
| 2 | Intervention v1 | Is the format direction used when answering? | registered | [intervention/](intervention/README.md) | [report](reports/intervention-v1/README.md) |
| 3 | Transfer v1 | Does the length effect hold on unseen tasks? | registered | [transfer/](transfer/README.md) | [report](reports/transfer-v1/README.md) |
| 4 | Length control v1 | Is d just a verbosity knob? | registered | [length_control/](length_control/README.md) | [report](reports/length-control-v1/README.md) |
| – | Format subspace | Which part of B − C is shared, by layer? | exploratory | [exploratory/](exploratory/README.md) | [report](reports/format-subspace/README.md) |
| 5 | Multi-model replication | Do 1–4 hold in Llama-3.1-8B and Gemma-2-9B? | registered, running | [replication/](replication/README.md) | pending |

All registrations, with their commits, are listed in [docs/registrations](docs/registrations/README.md).

## Key terms

| Term | Meaning |
|---|---|
| A / B / C | online (cache from generation) / native role replay / pasted into one user message. B vs C is the main comparison. Older files call B and C `nat`/`flat` or `N`/`P`. |
| d, the format direction | Mean of (B state − C state) at the last prompt token, at one layer. It is cross-fitted, so a history's own pair never enters its own vector. |
| Steering | Adding a vector to one block's output at the last prompt token and at every generated token. Alpha 1 means the vector is used at its natural size. |
| Teacher-forced KL | Feed B's answer token by token and measure how far a condition's next-token distribution is from B's. Lower means more native-like. |
| L_sign, L_random | Mean log length ratio of C+d against C−d, and of C+d against C+random. Positive means d lengthens answers specifically. |
| S indices | (KL_X − KL_C+d) / (KL_X + KL_C+d). Positive means C+d is closer to native than condition X. |
| Intervals | History-cluster bootstrap. 97.5% when a study has two primary tests, otherwise 95%. |

## Repository map

```
README.md, AGENTS.md         start here (humans / AI agents)
PREREGISTRATION.md           confirmation v2/v2.1 protocol (registered at this path)
confirmation/  analysis/     study 1: build cases, run, score; probe and scoring code
intervention/                study 2 (steer.py: hook, generation and KL helpers reused later)
transfer/                    study 3 (summarize.py: bootstrap and statistics reused later)
length_control/              study 4
replication/                 study 5: model-agnostic engine, runner, shallow follow-up
exploratory/                 unregistered analyses
data/<study>/                frozen inputs, manifests and registration records
reports/<study>/             results: README.md (plain language) + summary.json
docs/                        registrations index and write-ups, activation interface, October notes
tests/                       CPU tests (python -m unittest discover -s tests)
scripts/                     October report builder, release check, October batch runner
experiments/ reference_code/ handoff.py   October data, scripts and scorer (read by later code)
provenance/                  source hashes, October selection, release manifest
third_party/mt_eval/         MT-Eval rule checker (original license)
```

Large outputs are not in git: generations, activations and run folders live under `runs/`, which is gitignored. The v2.1 histories and activations are in the [confirmation-v2.1 release](https://github.com/mwang693-rgb/dialogue-migration-handoff/releases/tag/confirmation-v2.1).

## Checks (CPU, no model needed)

```bash
python scripts/build_report.py                 # rebuilds reports/october-exploration from saved outputs
python -m unittest discover -s tests -v        # tests needing numpy/torch/transformers skip if absent
python scripts/verify_release.py               # file hashes + credential scan against provenance/release_manifest.json
```

The GPU runs need `pip install -r requirements-gpu.txt`, a local model and CUDA. Each study's README gives its exact commands. Every GPU script checks the registration hashes before it starts.

## October exploration (study 0)

Three small exploratory batches came first (October 10–14, 2026): local editing of books, code and tables; MT-Eval response rules; and contractions in casual replies. They compared A, B and C on Qwen-generated histories. They found no task where pasting caused a large, consistent loss, which led to the registered confirmation.

- Tables and examples: [reports/october-exploration](reports/october-exploration/results.md).
- Notes on everything else tried: [docs/experiment-notes.md](docs/experiment-notes.md).
- The October scripts under `reference_code/` still refer to the old workspace layout. Read them for the methods; do not run their `main()` functions.
- Third-party material keeps its original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
