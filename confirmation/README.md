# Confirmation generation

Use [version 2](../PREREGISTRATION.md). It fixes 120 accepted histories,
30 per MT-Eval rule, with a predeclared 240-candidate Dolly question reserve.
Each accepted history has three Qwen-generated practice replies, then ordinary
and changed-rule requests under A/B/C/C+reminder (960 final answers).

The candidates are in `data/confirmation_v2/candidates.jsonl`; reference Dolly
answers are not used. This is a constructed response-rule task, not an official
benchmark aggregate. All rejections occur before test generation and are saved.

Test CUDA first on the two old development cases. Then freeze, commit the
registration, and launch into a new directory. See [commands and tensor
interface](../docs/activation-interface.md). Model weights require a pinned,
SHA256-verified local snapshot. Do not put login credentials in the repository.

Deliver `activations.pt` (ordinary B/C), `activations_change.pt` (change B/C),
`outputs.jsonl`, histories, configs, registration, validation and file hashes.
The GPU step does not fit a probe. CPU probe corrections and human review are
explicitly listed in the protocol; do not use the inherited CV layer-selection
summary as a confirmatory result. Interventions need a separate registration.
