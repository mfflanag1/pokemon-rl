# M3 switching/hidden-set engine and categorical learner

This stage ports the sealed two-member `gen9-starter-pairs-v1` reference profile
to independent scalar and batched CPU JAX implementations, then tests a four-action
categorical learner. It does not port unrestricted Random Battles. Public ladder
strength and full G2/G3/G4 remain open; Prime is unnecessary.

The versioned [contract](../contracts/switch-engine-v1/README.md) and
[fixed plan](../contracts/switch-engine-v1/plan.json) define 24 constructed teams,
hidden reserve/species and speed variants, voluntary and faint replacement switches,
the twelve-turn cutoff, bounded public history, and exact own versus percentage
opponent HP. A compact 323-feature player projection supplies both policy and value.
The actor never receives the opponent's private roster, speed, exact HP/PP, chance
state, commitments or winner label.

## Fidelity and information boundary

The scalar and JAX engines independently implement integer damage, ability activation,
priority, outgoing speed ties, switch targets, faint cancellation, PP expenditure,
replacement waits and turn advancement. Conditional replay compares full privileged
state and executed events, then separately compares each compact player view against
an independent encoder of the real receiving player's protocol history.

The frozen campaign replays all 72 sealed cases and uses 10,000 fresh development
seeds, cycling the declared team grid and five fixed action styles. The initial 24
fresh cases are also replayed through completely unpatched Showdown Sessions;
private views and privileged audit (including PRNG state) must match at every step.
The passivity test neither overrides chance nor forces outcomes.

Chance traces bind each executed attack to its recorded critical and roll words.
Gen 9 may shuffle tied moves repeatedly before first execution; the audit binds
the effective tie to the final such shuffle. JAX uses independent event-addressed
Threefry words and the sealed exact word-to-outcome maps. This establishes the
declared conditional semantics and finite-word laws, not sodium stream equivalence.

Information checks reuse the sealed 1,872 hidden-state invariance comparisons,
576 private-speed positive controls and 24 reserve-reveal pairs. Additional tests
mutate JAX private reserves/speeds, hidden exact opponent HP within the same public
bucket, and private PP. Exhaustive real `getHealth()` tests cover all 357 HP values
across the three species, and the damage grid covers 3,456 conditional values.

## Categorical collect/update loop

The actor/value model is a shared 323→32 tanh network with four action logits and
a scalar value. Categorical samples use a two-level tree of the sealed binary
finite-word policy: independent node words, a 2^-24 rare-mass floor, and matching
conditional log likelihoods. A leaf may have probability as small as 2^-48.
All sixteen masks and extreme tails are tested; moderate-logit PPO losses and
autodiff gradients agree with the sealed NumPy categorical reference.

Each update collects 32 games for at most fourteen request groups, giving a fixed
896-row two-player buffer. Waiting and completed-game rows are inert padding.
Transitions link only the same player in the same game. Terminal rewards are win1/
loss0; twelve-turn cuts bootstrap from that player's value under the frozen collecting
actor, then explicitly reset. No cutoff becomes a draw or evaluation score.
Two complete masked PPO epochs follow collection; the plan fixes all hyperparameters.

Three seeds (101/211/307), each 64 updates, are an engineering smoke. Its 6,144 primary
training games are separate from checkpoint replay and injected rollback work.
There is no strength threshold, opponent screen, ladder test or selection of a best seed.

Checkpoints bind the complete model, Adam state, seed, counters, plan, source tree
and admitted CPU/runtime. Supported resume is only at a completed-update boundary;
live games and partial buffers are unsupported. Original in-memory updates following
boundaries 0 and 32 are fingerprinted before loading and repeating from disk. Boundary
64 is also loaded exactly. Injected collection/update failures must leave the prior
state intact. Interrupted file writes preserve the published checkpoint and `.partial`
evidence; they do not silently resume or delete the partial file.

## Verification and reproduction

Production execution and sealing use:

```sh
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m3-switch-engine
npm run jax:switch-engine -- artifacts/runs/m3-switch-engine-v1
.venv/bin/python scripts/seal-switch-engine.py artifacts/runs/m3-switch-engine-v1
```

The campaign requires a new output directory and preserves a failure record on
errors. Source-bound regression must finish before production capture, because old
provenance tests temporarily alter sentinels. The receipt checks predecessor live/
archived hashes, source/runtime admission, full conditional/player-view replay,
counter and telemetry linkage, and fresh original-next-update disk replay. Sealing
must repeat to the same receipt and every source/evidence hash must match live and
archived copies. Historical handoffs and sealed evidence remain unchanged.

## Measured production results (2026-10-11)

The fixed run `artifacts/runs/m3-switch-engine-v1` completes successfully. Its
[receipt](../artifacts/reports/m3-switch-engine.json) binds the source, runtime,
source-bound regression and archived evidence after fresh sealing audits.

| Check | Measured result |
|---|---:|
| Fresh Showdown games | 10,000 |
| Fresh state boundaries / player views | 136,635 / 273,270 |
| Fresh decisions / executed events | 241,384 / 234,307 |
| Replacement boundaries / recorded chance draws | 11,886 / 332,815 |
| Fresh terminal / explicitly truncated games | 2,186 / 7,814 |
| Legacy games / boundaries / player views | 72 / 937 / 1,874 |
| Hidden equalities / own-speed controls / reserve-reveal pairs | 1,872 / 576 / 24 |
| Fresh unpatched passivity controls | 24 |
| Primary learner updates / games / decisions | 192 / 6,144 / 140,983 |
| Learner terminal / explicitly truncated games | 3,469 / 2,675 |
| Exact original-next-update replays / checkpoint loads | 6 / 9 |
| Injected in-memory collection/update discard checks | 6 |
| Source-bound regression | 443 tests: 237 Node + 206 Python |
| Unchanged tested source/configuration files | 320 |

There are zero scalar/JAX/reference transition, event, mask or player-view
mismatches. Each primary update uses valid masked actions, finite buffers/gradients/
parameters and independently checked NumPy forward-link GAE. The three seeds
contribute 47,287 / 46,892 / 46,804 decisions respectively, each exactly 64 updates
and 2,048 games. Checkpoint audit replay games and injected discarded work are
separate from the 6,144 primary training games.

Reference capture took 126.066 seconds, including legacy/passivity work. The
subsequent complete learner smoke took 3.700 seconds, including its checkpoint and
rollback audits. This short measurement is not a sustained throughput qualification.
The process high-water RSS was 835.219 MiB and includes the preceding Python parity
work; it is not an isolated learner/device-memory measurement. Exhaustive protocol
history reconstruction was the slow verification phase, not the JAX collect/update
loop. No GPU or Prime resource was used.

Independent Fable snapshots reviewed mechanics, learner and campaign admission.
Coordinator verification rejected an incorrect HP-rounding suggestion using the
actual pinned source and exhaustive getHealth tests, and fixed chance-trace,
unpatched-passivity and evidence-binding gaps. The independently delegated damage
helper received opposite-model Codex QA and local execution before integration.
See [review dispositions](../artifacts/reviews/m3-switch-engine/review-dispositions.md).
Eleven copied-evidence sealer negatives reject source/runtime/plan drift, incomplete
work, broken bindings/counters, reserved-data use and forged replay evidence without
changing production hashes or writing a receipt.

Preflight failures, the initially incorrect coordinator stat test, all review
snapshots and 17 original preflight evidence files remain preserved.

The next gate is this exact switching/categorical configuration's own sustained CPU
qualification, followed by a preregistered restricted learning-strength evaluation.
The earlier public-duel soak and full-battle tactical heuristic do not qualify this
new learned policy. Broader simulator mechanics, public deployment and competitive
strength remain separate work.

Before a strength screen, preregister treatment of unfinished games: the current
cutoff cannot be scored as a draw. Full battles require an expanded admitted
horizon/history and PP/mechanics scope. No ladder rank or improvement in full-format
playing strength is inferred from this engineering smoke.
