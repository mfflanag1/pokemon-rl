# M1 binary policy: sampling precision and matching PPO logs

The additive `pokemon_binary` package resolves the inherited FP32 inverse-CDF
sampling mismatch for the two-action fixed Rattata duel. It integrates a new
policy law into the fixed-shape collect/update loop, with a distinct checkpoint
format and source-bound contract. **Full G2/G3 remain open. Local CPU is sufficient;
Prime is unnecessary.** The previous sealed learner and its two-hour soak are
unchanged. This policy has a separate short qualification, not a new two-hour soak.

## Policy decision

For two legal actions, compute `d = logit0 - logit1`. The less likely action is
0 when `d < 0`, otherwise 1; at a tie both have mass 1/2. Its probability is the
FP32 value `r = max(2^-24, exp(-abs(d)) / (1 + exp(-abs(d))))`.
The two exact sampling masses are this dyadic number `r` and its real complement
`1-r`. The floor is about 5.96e-8. This is a deliberate change to the policy family,
not a claim to preserve arbitrarily small old softmax probabilities.

The sampler draws two uint32 words in a new `DYB1` random domain. It selects the
rare action precisely when their high/low concatenation is below `r * 2^64`.
For an admitted FP32 number, `r = m * 2^(e-150)`, with a 24-bit significand `m`
and biased exponent `e` between 103 and 126. Thus the threshold is the integer
`m * 2^(e-86)`, with shift 17 through 40. Two uint32 operations represent it
exactly, with no FP64 support or floating uniform grid. Counting integers below
that threshold gives mass `r`, **conditional on uniformly distributed word pairs**.
The tests do not prove PRNG independence or measure rare-event frequency.

The behavior log and PPO log use `log(r)` for the rare action and `log1p(-r)` for
the common action. Rounded displayed probabilities do not determine complements.
The gradient follows ordinary FP32 autodiff through these expressions; at and
below the floor the actor derivative is zero. There is no straight-through integer
quantizer. Floating-point transcendental, entropy and gradient arithmetic retain
normal rounding error; the exact claim concerns the finite-word selection map.

With one legal action, its mass is one and policy gradient/entropy are zero.
Terminal padding has action -1, zero probabilities/logs/value. The `rare_action`
and `rare` helper fields are internal and immaterial when fewer than two actions
are legal; the scalar oracle uses different sentinel values for those fields.
A test pins the selected-log gradients at equal logits on the admitted JAX version.

## Qualification

Accepted campaign: `artifacts/runs/m1-binary-policy-v2`, content hash
`74b10a37c3cac44f80ac59e78b06be23d12fd63e25726725124a1ce246682c63`.
See [the completion receipt](../artifacts/reports/m1-binary-policy.json).

| Check | Result |
|---|---:|
| Source-bound regression | 218 tests: 63 Node + 155 Python |
| Independent integer-ratio threshold probes | 100,002 exact matches |
| Injected boundary selections, both action orderings | 600,012 exact matches |
| Collector and fused/split parity | 24 batches, 1,536 games, 9,614 decisions |
| Maximum fraction of numerical tolerance used | 1.96% (atol 3e-6 + rtol 3e-5) |
| Three seed qualification | 100 updates each; 19,200 games, 120,274 decisions |
| Saved boundaries with exact compared next cycle | 5 |
| Short timed CPU run | 60.000 seconds, 56,678 cycles |
| Timed games / submitted decisions | 3,627,392 / 22,730,738 |
| Timed executed moves | 19,948,487 |
| Operational throughput | 378,844 submitted decisions/s |
| Unexpected compilations after warm-up | 0; baseline 216 events, real positive control passed |
| Peak process RSS | 562,987,008 bytes (536.906 MiB) |

The short run's smallest `exp(behavior_logp)` was 1.25e-6; it did not empirically
exercise the floor. Deterministic mirrored-tail and threshold-boundary tests
exercise the floor and the historical 1.81e-12 nominal example. No rare-event
frequency inference is made from the soak. Process RSS is not a device-memory
headroom certificate. This throughput is for repeated tiny duels, not the intended
full format, and is not a controlled performance comparison with the old soak.

The reference collector runs its existing Python control loop with the new actor
passed explicitly. It checks semantic actions, features, logs, values, game/seat/
turn identities, same-player links, terminal rewards and GAE. Separate collection
and update are compared with the fused executable. The scalar integer-ratio oracle
and analytic selected-log derivatives are independent of the production bit shifts
and autodiff; its probability formula is shared, evaluated through NumPy.

The new checkpoint format rejects old sampler checkpoints. Exact save/load checks
compare the following cycle; failure-injection tests cover rollback after collection
and after update, plus interrupted publication. These remain whole-episode,
completed-update boundaries. Old rollout logs cannot be reused under the new law.

## Review and reproduction

The claude-review skill supplied
independent review. Auto routing selected a Codex oracle delegate for exploration;
that delivery failed ownership checks and used an incompatible test framework.
The failure is retained. Only the inspected scalar source was salvaged; new local
unittest checks and Fable snapshot QA cover it. The review found no blocking policy
issue; forced-row helper semantics and the pinned tie-gradient convention are
explicit above. See [verified dispositions](../artifacts/reviews/m1-binary-policy/review-dispositions.md).

```sh
# Existing pinned environment; no installation or paid compute required.
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m1-binary-policy
npm run jax:binary-policy -- artifacts/runs/NEW-binary-policy-run
.venv/bin/python scripts/seal-binary-policy.py artifacts/runs/NEW-binary-policy-run
```

The preliminary v1 run also passed; v2 requalifies an explicit checkpoint schedule
after unused inherited interval fields were removed from the plan. Both are retained.

Use a fresh campaign directory. Completion receipts are write-once. The generic
source-bound regression wrapper is reused unchanged; its log is stored separately
from earlier evidence. Existing contract, simulator, learner and fixed-loop source
hashes are checked on admission and exit. Final reserved teams remain unused.

## Next work

The numerical action-sampling decision is now implemented and qualified in this
binary profile. Next version the exact Showdown finite-word chance mapping and
expand fresh player-channel reference comparison from 1,000 to 10,000 games.
The current environment sampler still targets nominal 1/24 critical probability;
this policy change does not resolve that separate finite-word difference.

The fixed public duel cannot provide the full hidden-species/set/item/ability/Tera
mutation strata. Broader mechanics need new profiles and fixtures. G3 also needs
frozen-opponent evaluation over multiple training seeds, critic calibration and
a new two-hour stability run for whichever final policy/chance contract is selected.
Preregister random, damage-greedy and checkpoint opponents after the relevant G2
work. Keep final evaluation sealed. No GPU or Prime resources are needed next.
