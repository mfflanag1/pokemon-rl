# Player features v1

Scope: pinned Gen 9 Random Battle singles with HP Percentage Mod. Wire schema
`1.0.0`, feature version `r5-player-features-1`. R2/R4 contracts are unchanged.
`layout.json` defines every column and shape; `catalog.json` preserves R2 category
indices and adds public `asone` and request-only `recharge`. Zero means unknown,
one means explicitly absent. Missing numeric values have separate knowledge flags.

| Tensor | Shape | Meaning |
|---|---|---|
| `own` | 6 × 26 | Exact current own request, in current request order |
| `own_public_active` | 27 | Own active appearance in the visible protocol |
| `opponent_appearances` | 6 × 27 | Last six encounter segments, newest first |
| `advertised_moves` | 4 × 8 | Currently selectable move identities and advertised PP/disabled flags |
| `actions` | 14 × 5 | Request-relative action kind, slot, subject ID, Tera and admissibility |
| `action_mask` | 14 | Boolean visible admissibility |
| `global` | 17 | Turn, phase, request count, observable flags and memory counters |
| `event_bytes` | 64 × 192 | Recent visible mechanical events, UTF-8 byte + 1 |
| `fact_bytes` | 32 × 128 | Latest explicit weather and unmatched field/side starts |

Both byte tables have row lengths and overflow flags. Zero padding carries no
content. Byte truncation may split a UTF-8 sequence; consume byte tokens directly.
Encounter/event truncation counts are in `global`. The full verified own history
and exact request remain available from `PlayerHistory.snapshot()`.
`unstructured_events` counts unsupported message kinds and recognized subject
messages without a safely matched encounter; it is not a mechanics-coverage total.

Own `base_*` stats are the request's `baseStoredStats`, not effective boosted or
transformed combat stats. Current private `ability` and `base_ability` are separate.
Own fainted max HP is retained only if previously observed in an own request.
Opponent HP is the displayed fraction, with a percentage flag; no exact integer
HP is inferred. Unknown item/ability/Tera remain unknown until visible evidence.

Appearances are not inferred team slots. Each switch/drag starts a new segment.
Illusion `replace` updates that segment without rewriting previous actor inputs;
`entry_species` keeps its initial apparent species. Transform changes display
species only from the target's visible appearance. Last-seen move/item/ability
fields are historical evidence, not assertions about a full or current set.
Appearance `active` marks the latest occupant segment until replacement; consult
`fainted` separately. It does not assert that the occupant can currently act.
`item_removed` and `ability_suppression_seen` describe observed events, not a
complete current effect state. `true_roster_slot_known` is always zero.

Moves 0–3 are ordinary; 4–7 are their Tera choices; 8–13 switch to current request
slots 0–5. Forbidden actions cannot be decoded. Wait and pass are control flow,
never a categorical-head placeholder. Missing PP is unknown, including forced
Recharge/Struggle advertisements. Never remap actions by species or original team
order after a switch.

`PlayerHistory.append(view)` accepts the R4 own-player view only. It verifies
schema, owner, delta bounds, prefix checksum, latest request, counters, view ID and
mask. Updates commit transactionally; exact current retries return the same input.
Foreign requests anywhere in history, stale deltas and malformed views fail.

`ActorInput.policy_inputs()` and `.value_inputs()` return the same tensor-only
content. IDs and layout/catalog hashes are transport metadata. Use
`load_actor_input(wire)` for external feature JSON: it checks strict shape/schema,
semantic bounds and content identity. `ActorInput` instances are trusted internal
values; neither Python type annotations nor content hashes authenticate a sender.
Only projected reference views supply the information boundary.

`assemble_episode(bundle, manifest)` validates privileged R4 data once, then emits
immutable player trajectories with separate actor inputs, transition targets and
truncation bootstraps. Rewards, action labels, future observations, episode IDs,
opponent requests, initial teams and simulator audits are excluded from inputs.
The low-level `project_player_views` does not certify an episode; use it only after
validation or in explicitly diagnostic noninterference tests.

This first encoder is intentionally bounded and lossy. Old boosts, volatile
conditions, exact durations and older reveals may be absent from its fixed tensors.
Full protocol history remains available for a later recurrent or richer encoder.
No learned policy, privileged critic, complete belief state or engine-fidelity
claim is attached to this interface.
