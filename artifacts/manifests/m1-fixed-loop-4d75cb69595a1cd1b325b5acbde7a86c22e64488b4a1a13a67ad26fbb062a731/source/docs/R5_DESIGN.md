# R5 design contract

R5 is an additive pure-Python observation layer. Frozen R1–R4 producer/consumer
sources and their manifests stay unchanged. The policy and value encoder receive
exactly the same typed actor input; targets, simulator truth and campaign metadata
have separate types and no route into that input.

A stateful PlayerHistory accepts only the R4 view object (not an episode, frame,
request alone or arbitrary protocol). It validates schema, owner, monotone delta
bounds, incremental canonical prefix hash, request count/turn, latest request,
view ID, visible admissibility and mask. A failed append does not change history.
Identical current-view delivery is idempotent; stale/conflicting delivery fails.
The offline adapter selects views in same-player order; it validates a full bundle
once at its boundary and emits actor inputs separately from transition targets.

Features contain: six own Pokémon in current request order; four currently
advertised moves; fourteen action masks/bindings; current phase/turn; a bounded
recent public-event byte window; and conservative opponent appearance summaries.
Category IDs preserve the pinned R2 vocabulary with explicit unknown/none values;
R5 adds the request-only Recharge and ambiguous public As One labels.
Values and knowledge flags are separate. Own request data remains available in a
typed private snapshot, including exact HP, stats, moves, item, ability and Tera.
Request `stats` are explicitly baseStoredStats, not boosted combat stats. Requests
are authoritative for own current data. No opponent data is filled from the Dex,
initial teams, generator priors, raw end messages, audits, commitments or RNG.

Opponent appearances are encounters, not asserted roster slots. Start a new one on
every opponent switch/drag. Preserve history in the loader, expose a fixed recent
window with truncation flags. `replace` updates only the current appearance when
Illusion breaks, without repairing earlier snapshots or merging by nickname/species.
Distinguish apparent species from unknown true identity. Transform/form changes
update only publicly justified display state, never copy hidden target stats,
abilities, moves or items. Opponent HP retains its shown denominator and precision;
no integer HP is inferred. Revealed item/ability/move evidence is explicitly
last-observed, not a guarantee of a current set or current ability after changes.
All public events are retained in history; unsupported structured interpretations
are visible and remain in bounded event tokens, never silently treated as known.

The fixed integer tensor view is a feature interface for a future learner, not a
trained model. Public events use UTF-8 bytes (0 is padding, byte+1 is data), with
row/byte lengths and overflow flags. This avoids inventing hidden semantics for
unmodeled effects. It is a deliberately conservative first encoder; bounded memory
and appearance duplication are explicit limitations, not complete belief state.

Acceptance: shared Node/Python semantic-action round trips in every request phase,
private delta/retry/corruption tests, explicit known/unknown and HP precision cases,
Illusion/Transform adversarial fixtures, at least 1,000 paired adapter-carrier
perturbations with byte-identical actor inputs and masks, separate live simulator
interventions with unchanged visible history, positive visible-reveal
controls, deterministic policy/value probe invariance, and scanning all 1,000 R4
episodes plus saved fixtures for reconstructibility. Do not treat edits to invalid
privileged artifacts as valid simulator counterfactuals: distinguish adapter
noninterference tests from actual live hidden-state tests. No learned-logit,
complete mechanics or final-evaluation claim.
