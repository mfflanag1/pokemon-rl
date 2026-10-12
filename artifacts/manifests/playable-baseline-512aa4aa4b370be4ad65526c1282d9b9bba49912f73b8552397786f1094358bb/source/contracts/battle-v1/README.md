# Battle contract 1.0.0

R2 supports the pinned `gen9randombattle` singles reference. JSON Schema Draft
2020-12 describes structure; the Node and Python validators additionally enforce
hashes, identity, request ownership, and action/mask consistency. Call the complete
validator, not a bare schema validator. Only `1.0.0` is implemented: future versions,
including minor versions, require an explicit reader upgrade. Additive optional
fields may receive a minor version; changed meanings/required fields need a major.

## Canonical bytes and identity

The `pokemon-json-integers-v1` profile is deliberately narrower than general JSON
and is **not RFC 8785**. Wire numbers must be integer literals (no decimal point or
exponent), in [-9007199254740991, 9007199254740991]. `-0` normalizes to `0`.
Floating point features are outside this initial contract. Seeds are
`{"kind":"showdown-sodium","hex":"<64 lowercase hex digits>"}`.

Reject duplicate object keys, invalid UTF-8, lone Unicode surrogates, non-JSON
values, and nesting deeper than 100 containers. Strings are not Unicode-normalized.
Sort object keys by Unicode scalar value, preserve array order, use no whitespace,
and encode UTF-8 without BOM. Escape quote/backslash and controls with JSON's
short escapes (`\b\t\n\f\r`), otherwise lowercase `\u00xx`; leave slash and all
other Unicode characters literal. The strict JSON loaders are part of the API.

Every artifact has exactly `schema_version`, `artifact_kind`, `manifest_id`,
`payload`, `content_hash`, `artifact_id`. The content hash is lowercase SHA-256 of
canonical bytes of the first four fields. `artifact_id = kind + ':' + content_hash`.
A manifest's `manifest_id` is SHA-256 of its payload, avoiding a self-reference;
other artifacts carry that manifest ID. IDs detect alteration, not authenticity.
Collection loaders must resolve references against the expected manifest.

## Actions: `gen9-singles-request-v1`

| Semantic action | Policy index | Node wire command |
|---|---:|---|
| `{"kind":"move","slot":0..3,"terastallize":false}` | slot | `move slot+1` |
| Same with `terastallize:true` | 4 + slot | `move slot+1 terastallize` |
| `{"kind":"switch","slot":0..5}` | 8 + slot | `switch slot+1` |
| `{"kind":"pass"}` | none; reject tensor conversion | `pass` |

The policy has **14 indices**. Move slots index `request.active[0].moves` and
switch slots index `request.side.pokemon` **at that request**, not initial team
order or persistent Pokémon identity. A locked move or Struggle occupies its
actual request slot (normally zero). The request's move ID disambiguates it.
Struggle does not get a second tensor index. Only the Node codec handles protocol
strings. Python maps semantics to indices; both implementations derive masks from
the same visible request and shared action vocabulary.

An action records the attempted request slot, not a promise of the executed move.
For example, when every live move is secretly disabled, an advertised move attempt
can resolve to Struggle before the simulator emits any disabled-choice rejection.
Retain the resulting protocol evidence; never relabel the earlier visible request
using future knowledge. A request with no advertised enabled move fails explicitly.

Wait has no actions and an all-false mask. Pass is a control-only action with an
all-false policy mask; the caller must handle it without sampling a fake action.
Normal choices omit truthy-disabled moves; Tera is offered only when advertised;
switches exclude active/fainted Pokémon and are suppressed only by `trapped`.
`maybeTrapped`, `maybeDisabled`, and `maybeLocked` do not reveal hidden truth.
Masks describe **admissible attempts**, not guaranteed engine acceptance. A zero
PP count is not an independent veto when the request still advertises a move.

For forced switches, use the request order and Revival Blessing's active
`reviving` flag: eligible fainted targets for revival, live bench targets otherwise.
`forceSwitch:[false]` or no eligible target produces pass, following the pinned
reference AI. This encodes control behavior; it does not assert every synthetic
pass request is naturally reachable. Reject doubles, team preview, non-Gen9
transformation fields and unknown request properties instead of guessing.

## Artifact boundaries

* `manifest`: R1 source/build/config/license evidence, schema/vocabulary/source
  hashes, Node producer version and contract versions. Consumer Python version is
  recorded in the verification report, not injected into every producer manifest.
* `team`: seed, original official six-member output, and separately resolved Dex
  identifiers. Preserve original names/forms and generator metadata verbatim.
* `player_view`: viewer's own request and visible protocol prefix, phase,
  request counter, admissible actions, 14-bit mask, and explicit unknown opponent
  private state. No seeds, team corpus IDs, decisions, outcomes, or future protocol.
* `decision`: submitted semantic action, pre-submission view reference, observed
  acceptance and feedback. This is training/audit data, **not an actor input**.
* `episode`: ordered decision/team references, seeds, input tape, result and R1
  evidence hashes. Privileged; only the player-view loader is an actor API.

`phase` is `move`, `switch`, or `wait`, as determined by the request. R2 does not
infer whether a switch was a pivot or faint replacement, nor whether decisions
belong to a joint resolution group. R4 will add that richer transition model.
No `FullState`, belief-state or target artifact is accepted as a player view;
their complete schemas are deferred until their semantics are implemented.

Individual validators do not resolve foreign artifact IDs or prove dataset closure.
Collection consumers must resolve the episode's ordered, unique decision IDs and
check their view/episode references. Both team IDs may be equal for a mirror match.
Remaining-Pokémon counts are not rewards and are not used to infer the winner.
An episode includes the raw evidence hash, so its artifact ID can change with raw
timestamps; `episode_key` and normalized reference hash identify the deterministic
setup and normalized result separately.

`known_integer` and `known_id` schema fragments use `{"state":"unknown"}` or
`{"state":"known","value":...}`. Known numeric zero is a value; unknown never
contains a value. Vocabulary token 0 is `<unknown>`, token 1 is `<none>`, and
remaining tokens are lexicographically ordered pinned Dex IDs. Known empty IDs
map to `<none>`; absent knowledge maps to `<unknown>`; unlisted known IDs fail.
Dex vocabulary inclusion is not a claim of format reachability or engine support.

The active table is `showdown-dex-ids-v2`. R3 added 68 cosmetic species IDs that
`Dex.species.all()` omits but `Dex.species.get()` resolves. The original v1 table is
preserved as `vocabulary.v1.json`. Token indices can change between revisions:
persist the exact table version and hash with tensors/checkpoints, and migrate by
string ID. The artifact schema remains 1.0.0; manifests bind the revised table hash.

R2 preserves protocol evidence, not a finished model feature parser. Consumers
must use a trusted Node projection: schema validation cannot prove that arbitrary
text supplied by an untrusted producer contains no secrets. R5 owns structured
history features and broader information-invariance tests.
