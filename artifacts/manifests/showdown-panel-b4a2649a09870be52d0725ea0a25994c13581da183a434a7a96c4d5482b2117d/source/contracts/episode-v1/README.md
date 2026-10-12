# Episode contract v1

R4 adds `phase_episode` bundles without changing the frozen R2 battle-v1 schema.
The entry point is `schema.json` (Draft 2020-12); battle-v1 references resolve
locally in both AJV and Python's `referencing.Registry`. Read [R4_EPISODES.md](../../docs/R4_EPISODES.md)
for lifecycle rules and runnable commands.

The bundle contains `episode`, `raw_output`, and `raw_output_hash`. The episode
uses R2 canonical JSON and the usual schema version, artifact kind, manifest ID,
content hash and artifact ID. Its payload includes explicit initial teams/seeds,
controller and submission order, the exact input tape, N decisions, N transitions,
N+1 captured frames, per-player normalized protocol, spectator protocol, outcome,
and optional nonterminal bootstrap views. All recorded rewards are integer
numerators over denominator 2 (win 2, draw 1, loss 0). Nonterminal rewards are 0.

An episode key binds specification, fixture recipe, controller, submission order
and manifest. Each decision ID binds that key and the complete decision body;
episode content hashes also cover frames, linkage and outcome. Retries reuse a
transport submission ID and do not create a new decision. Indices are zero based.
Request IDs count every private request, including wait and error re-requests.
Group IDs count joint simulator request advances; those clocks differ.

The R4 manifest nests the R2 reference manifest and fingerprints R4 lifecycle,
phase, fixture, validator and schema sources. Official and constructed profiles
have separate identities. Campaign metadata additionally fingerprints the writer,
worker, storage and CLI sources. Replay requires the matching current runtime;
historical artifacts need their pinned source snapshot and dependencies restored.

## Information boundary

**The entire bundle is privileged diagnostic data.** Frames include both private
request hashes, selected internal mechanics and both pending choices. Raw `end`
messages contain both teams. Cross-player phase/group/commitment labels are not
actor features. R2's player-view loader rejects the `phase_episode` artifact kind.

A decision's `view` contains only that player's current request, its visible
protocol delta and prefix hash, own request count, and visible admissibility mask.
Delta bounds are `[protocol_start, protocol_end)` in that player's history. Append
deltas only along that player's trajectory; another player's intervening decision
is never its next observation. Action masks can contain choices later rejected by
a hidden restriction. Such feedback is preserved, not converted to a different
choice. R5 must build and adversarially test the actual actor feature loader.

The selected mechanics audit includes RNG state, queue action identifiers, HP,
PP, boosts, statuses, item/ability, Tera state and effect names. It omits effect
state, counters, callback state and other engine fields. It is neither a complete
restart state nor proof of all mechanics. Replay starts from initial teams and the
full input tape. Normalize only timestamp protocol lines; retain raw evidence.

JSON schema checks structure. Node/Python semantic validators additionally check
hashes, channel projection, history/request boundaries, action encoding, rejection
invariants, commitments, safe cuts and exact same-player transition linkage. Python
does not independently simulate mechanics; exact engine replay is a Node check.
