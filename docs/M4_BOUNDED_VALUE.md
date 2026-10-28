# M4 — bounded-value CPU canary

The fixed three-seed engineering canary passed on 2026-10-12. Its sigmoid value
head, NumPy/JAX gradient checks, complete update replay and checkpoint recovery
checks passed. This does **not** establish stronger play or fix episode truncation.

The predecessor M3 screen remains unchanged: it completed but all three learning
gates failed. The follow-up diagnosis identified values above the terminal-win
reward and unanchored cutoff targets; it did not prove the cause of collapse.
See [diagnosis](../artifacts/diagnostics/m3-switch-screen-v1/HANDOFF.md).

## Measured result

Production: `artifacts/runs/m4-bounded-value-v1`; launcher exit code 0.
All three fixed seeds completed eight updates each, two Adam epochs per update.

| Seed | Updates | Training games | Decisions | Terminal | Cutoff/reset | Collected value range |
|---|---:|---:|---:|---:|---:|---|
| 431 | 8 | 256 | 6,258 | 92 | 164 | 0.494849–0.500239 |
| 641 | 8 | 256 | 6,261 | 69 | 187 | 0.488460–0.500000 |
| 853 | 8 | 256 | 6,228 | 87 | 169 | 0.492119–0.501175 |
| Total | 24 | 768 | 18,747 | 248 | 520 | 0.488460–0.501175 |

Every collected target was in [0,1]; every update and metric was finite. Maximum
recorded gradient norm was 0.031377. These short-run values are engineering
observations, not evidence of calibration, long-run stability or improved play.

Reported canary loop time was 1.648 seconds, including its checkpoint audits;
this excludes process startup and admission work and is not a throughput benchmark
or sustained qualification. CPU FP32, Threefry, exact versions and platform are
bound in `admission.json`.

Six checkpoint pairs (initial/final for each seed) reproduced the original next
update's learner, buffer, simulator state and metrics exactly. The production
process performed 12 discarded checkpoint-audit transactions in addition to the
24 counted training updates. Those 384 audit game executions are excluded from
the training total and include replayed addresses. The independent audit replayed
every update of all three trajectories and each saved checkpoint's next update.
No seed, checkpoint or opponent was selected for favorable performance.

## Implementation and verification

New code is isolated in `python/pokemon_bounded/`, with the fixed plan and
mathematical scope in `contracts/bounded-value-v1/`. Existing sealed code is
imported without mutation. The actor's logits, masks, sampling law and key
addresses are identical at identical parameters. The value head is now sigmoid;
zero raw value weights initialize to 0.5. The shared trunk, PPO settings, Adam,
same-player GAE and 12-turn cutoff remain inherited.

The source-bound regression passed **497 tests**: 240 Node, 236 existing Python,
21 new Python; all **361 source/configuration files** remained unchanged during
testing. The new tests cover full-network forward/gradient parity, directional
finite differences for all six parameter tables at one fixture point, saturated
heads, padding/target isolation, actor sampling at fixed parameters, linked GAE
targets, checkpoint corruption and full-trajectory evidence corruption. The
NumPy oracle is restricted to legal-logit spread <=8 and exact [0,1] targets;
it does not certify derivatives at the actor's rare-mass floor. Production
allows a 2e-6 FP32 target tolerance and checks every actual buffer.

Independent Fable 5.1 reviews covered the delegated NumPy implementation and the
integrated canary. The coordinator verified the review's manifest/configuration
questions and checked another 18,144 adversarial return entries, all in [0,1].
This finite check is not a universal FP32 error bound. See
[review dispositions](../artifacts/reviews/m4-bounded-value/review-dispositions.md).

All **1,024 live/archive source and evidence hashes** in the M3 receipt were
verified unchanged. Historical handoffs and current M3 progress documents remain
sealed; this new handoff records M4 without rewriting them. The new result is
recorded by [the M4 receipt](../artifacts/reports/m4-bounded-value.json), which binds
the source, report, checkpoints, tests, reviews and this handoff. Post-seal checks
are stored outside the sealed review directory.

## Limits and next step

A bounded constant value can still be an unsupported cutoff fixed point, and
sigmoid saturation can suppress value gradients. The shared trunk still couples
actor and critic updates. Exclusive checkpoint writes reject partial files but
make no atomic replacement or power-loss durability claim.

**Next: a separately versioned episode-continuation contract and simulator
fixtures**, followed by scalar/JAX/Showdown parity. Carry unfinished games across
collection boundaries, preserve request/forced-switch phases and event addresses,
and extend bounded turn/history representation deliberately. Learner integration
then needs same-player return linkage, explicit bootstrap versus terminal handling
and checkpoints that restore live game state and the next original update. Keep
the current M3/M4 artifacts as immutable baselines. A fixed, preregistered learning
screen comes only after continuation and checkpoint parity are established.

Full G2/G3/G4, sustained qualification of this variant and public ladder strength
remain open. No strength matches, public server, final reserved data or Prime
resources were used. Local CPU remains sufficient for the next correctness step.

GitHub preservation remains on hold pending the existing private/public visibility
choice; no files or run data from this step were pushed or uploaded.

## Reproduction

Preserve the current production directory. To audit its complete trajectory:

```sh
PYTHONPATH=python .venv/bin/python -m pokemon_bounded.canary artifacts/runs/m4-bounded-value-v1 --audit
```

The full test runner is `scripts/verify-bounded-value.py` (requires a fresh review
output directory); it invokes the unchanged predecessor suite plus the new tests.
Do not silently rerun production into an existing directory or modify its frozen
source. Sealing is `scripts/seal-bounded-value.py artifacts/runs/m4-bounded-value-v1`;
with unchanged inputs it must produce identical receipt bytes. Source admission
intentionally rejects a later expanded or changed tree: use the archived source
and recorded runtime for historical reproduction.
