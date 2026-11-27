# Registrations

Every confirmatory study follows the same steps:

1. Commit the protocol and a registration record before any model output exists.
2. The registration record (`data/<study>/registration*.json`) stores the SHA256 of the protocol, the frozen inputs and every code file the run uses.
3. The GPU script refuses to start if any of those hashes differ.
4. Amendments made before generation are written into the registration's `review_note`, and the superseded record is kept.
5. Anything motivated by a result already seen is registered as a separate follow-up and labelled as such.

To check that a result came from registered code, check out the registration commit and compare the hashes. Registered files are also kept unchanged at their original paths on the current branch.

| Study | Registered in | Protocol | Registration record | Write-up | Outcome |
|---|---|---|---|---|---|
| Response-rule confirmation v2 | `c54f212` (2026-11-19) | [PREREGISTRATION.md](../../PREREGISTRATION.md), `confirmation/protocol.json` | `data/confirmation_v2/registration.json` | [PREREGISTRATION.md](../../PREREGISTRATION.md) | The run stopped after 1 history (a tokenizer replay mismatch); replaced by v2.1 |
| Response-rule confirmation v2.1 (technical rerun) | `24d8c4a` (2026-11-19) | same | `data/confirmation_v2/registration_v2_1.json` | same, amendment at the top | Pasting did not reduce compliance (B 112/120, C 115/120); recovery gate not passed |
| Intervention v1 | `6675ac8` (2026-11-26) | `intervention/protocol.json` | `data/intervention_v1/registration.json` | [intervention-v1.md](intervention-v1.md) | Inconclusive (G = −0.63); post hoc length pattern |
| Transfer v1 | `c8ef904` (2026-11-27) | `transfer/protocol.json` | `data/transfer_v1/registration.json` | [transfer-v1.md](transfer-v1.md) | P1 length transfers; P2 inconclusive |
| Length control v1 | `973f33c` (2026-11-27) | `length_control/protocol.json` | `data/length_control_v1/registration.json` | in the protocol's `question` and `decision` fields | not_just_length |
| Multi-model replication v1 | `2155797`, amended in `fc5e1d1` (2026-11-27) | `replication/protocol.json` | `data/replication_v1/registration.json` (superseded: `registration_superseded_keyerror.json`) | [replication/README.md](../../replication/README.md) | Running |
| Replication shallow follow-up (k = 2) | `074e4bb` (2026-11-27) | `replication/protocol_shallow.json` | `data/replication_v1/registration_shallow.json` | protocol `why` field | Running |

`PREREGISTRATION.md` stays at the repository root because the confirmation registration records its hash at that path.
