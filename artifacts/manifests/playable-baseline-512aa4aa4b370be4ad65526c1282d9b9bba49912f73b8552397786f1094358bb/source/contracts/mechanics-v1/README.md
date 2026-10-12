# R6 mechanics contract, major version 1

Producer: `packages/mechanics/`. Independent consumer and fail-closed boundary:
`python/pokemon_reference/mechanics.py`. JSON uses the existing R1/R2 canonical
UTF-8 sorted-key convention and SHA-256. Each document's `content_hash` hashes
all fields except itself. File byte hashes include the trailing newline.

- `r6-mechanics-1`: profile and pin, source/file and fragment hashes with one-based
  lines and zero-based half-open UTF-8 byte spans; generator alternatives; entries;
  call/construction/identity-write sites; conservative domain bounds.
- `r6-fixtures-1`: fixed teams, explicit RNG seed, full initialization/choice tape,
  normalized protocol and selected state checkpoints; assertion and replay result.
- `r6-catalog-1`: manifest/fixture hashes, owner tickets, family requirements and
  individual callback obligations. Reference case links do not certify a family.
- `r6-report-1`: file/producer hashes, inventory and reference-case counts.

An entry key is `domain:id`. Domains are moves, abilities, items, forms,
conditions and core. Empty item ID means no held item; it is not an unknown ID.
Cosmetic aliases reference their base source. Embedded conditions are indexed
inside their owning move/ability/item, including nested callback paths.

Reachability is `witnessed-initial` or `conservative-envelope`. Initial evidence
is independently `observed-initial`, `declared-candidate`, `not-established-initial`
or `profile-runtime`. The full base Dex intentionally includes candidates from
other generations, nonstandard entries and impossible combinations. It is a safe
planning superset, not a proof that every entry can occur in this format. No
unreachability exclusions or minimal closure are certified by version 1.

`unsupported` is the only valid engine support status in R6. Promoting an entry
requires a new engine implementation/attestation contract; editing this JSON to
`implemented-tested` cannot grant simulation access. `SupportGate.require()` and
`require_profile()` always reject until that engine contract exists. The pinned
Showdown reference remains usable. Unknown identities and empty profiles reject.

Every AST call and constructor in the five base tables, selected core modules
and generator is recorded. Identity field assignments are additionally recorded. A pathway ID hashes
its owner, source file, start byte, end byte and kind; end bytes distinguish
nested or chained calls with the same starting position.
Known sinks use full-domain bounds. Other calls/constructors retain a
`pinned-runtime` source obligation, including callback aliases; this classification
is deliberately coarse, not a resolved call graph. All callback-valued properties
in the five loaded tables must match the AST callback index, including arrays.
The pinned runtime includes upstream/lib code and dependencies bound by R1's
verified source/build receipt; those are not individually reimplemented by R6.
Runtime active-move modifications and freshly constructed effect objects are
covered by their source bodies, not a fictitious finite set of all object values.

A validator must reject unknown major versions, bad content/source hashes,
missing initial corpus roots, duplicate identities, broken owner/pathway/catalog
links, unclassified pathways, unknown families, invalid support claims and
source paths escaping the project. Source validation is local and read-only;
Python does not execute or reinterpret TypeScript. Validation is integrity and
consistency checking, not an authenticity signature or an engine parity test.

Future fixtures must add failure paths, immunity/suppression, switching cleanup,
random branch/draw-schedule coverage and cross-mechanic interactions. Neither
scripted teams nor observer interventions are official generator witnesses.
