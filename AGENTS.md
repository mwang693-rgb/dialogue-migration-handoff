# Notes for AI agents

Read `README.md` first. It covers the question, the findings, the study chain, the terms and the map. This file adds the rules and mechanics you need before you change anything.

## Hard rules

1. **Never edit a registered file.** A file is registered if its path is a key under `code_sha256` in any `data/*/registration*.json`, or if it is a `protocol*.json` or a frozen input under `data/`. GPU scripts assert these hashes and refuse to run if they differ. To list the registered code files:
   ```bash
   python -c "import json,glob;print('\n'.join(sorted({k for f in glob.glob('data/*/registration*.json') for k in json.load(open(f,encoding='utf-8')).get('code_sha256',{})})))"
   ```
   A bug fix in registered code is an *amendment*. Make it only before generation, describe it in the new registration's `review_note`, and keep the superseded registration file.
2. **New design, new study.** Give it a new folder with its own `protocol.json` and `freeze.py`. Register it in `data/<study>/registration.json` and commit before any generation. A follow-up motivated by a seen result is registered separately, and its protocol says why (see `replication/protocol_shallow.json`).
3. **Label claims.** Registered primary tests decide. Everything else is secondary, descriptive or exploratory, and is called that in reports. Exploratory code goes in `exploratory/`.
4. **Do not commit** server addresses, credentials, model weights, `*.pt` files, or anything under `runs/`. `scripts/verify_release.py` scans for known credential and host patterns. It reads every file as UTF-8, so binary files (PDF, DOCX) also break it.

## Layout conventions

- Each study folder has the same files: `protocol.json`, a build step, `freeze.py`, `run.py` (GPU), `summarize.py` (CPU) and `README.md`. Results go to `reports/<study>/README.md` and `reports/<study>/summary.json`, copied from `runs/<run>/summary.json`.
- Shared helpers, reused across studies (these are registered, so import them and do not copy them):
  - `confirmation/common.py`: I/O, hashing, request text and the rule checker.
  - `intervention/steer.py`: `Steer` hook, `generate`, `teacher_forced_logprobs`, `kl_summary`.
  - `transfer/summarize.py`: `cluster_boot`, `stats`, `idx`.
  - `replication/engine.py`: `Engine` for Qwen2.5, Llama-3.1 and Gemma-2 (`FAMILIES`); `depth_layers(n) = round(n*k/7)`.
- Several study folders contain files with the same name (`run.py`, `summarize.py`). Load them by file path with `importlib.util.spec_from_file_location` under a unique module name, as the tests do.
- Layer numbering: layer 0 is the embedding output and layer k is the output of block k. Qwen's capture layers are 0, 4, …, 28, and the main steering site is block 16.
- Prompts: B uses the model's chat template with real turns. C puts `User:\n…\n\nAssistant:\n…` into one user message, followed by a separator and the request (`replication/engine.py:messages`).

## Commands

```bash
python -m unittest discover -s tests          # CPU; dependency-heavy tests skip when packages are absent
python scripts/build_report.py                # must leave reports/ unchanged (CI runs git diff on reports)
python scripts/verify_release.py              # must pass in a clean checkout
```

CI (`.github/workflows/cpu-checks.yml`) runs these on Python 3.11 without numpy, plus an activation-interface job with CPU torch.

After adding, removing or changing tracked files, regenerate the release manifest in a clean checkout of the commit:

1. Back up the old `provenance/release_manifest.json` to `runs/`.
2. Delete it.
3. Run `python scripts/verify_release.py --create`.
4. Commit the result.

## Where things are

- Server runs are pulled into `runs/<study>-4090/…` (gitignored).
- Each run folder holds `registration.json`, `protocol.json`, `status.json`, `units/*.json` (one per unit, with generations and teacher-forced KL) and `summary.json`.
- The v2.1 histories and activations are in the GitHub release `confirmation-v2.1`.
- Transfer inputs are built from the sibling workspace `../dialogue-migration-recovery`, which is not in this repository.
