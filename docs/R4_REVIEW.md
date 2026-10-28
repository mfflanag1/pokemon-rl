# R4 review and verification

R4 was reviewed through the authorized claude-review skill
using selected source snapshots, and bounded Python implementation was delegated.
The review tools did not execute the main project's tests; checks described below
were run by the coordinator unless explicitly attributed otherwise.

## Source/lifecycle research

Fable research run `20261010-133750-ab4cf5b8` received opposite-model QA. Its complete
answer was rejected after source verification found incorrect fixture assumptions,
request-count precision and termination details. Individually verified mechanics
facts were retained. [Immutable adjudication evidence](R4_SEMANTICS_AUDIT.md)
records the concrete corrections. Pinned, hash-labeled Showdown source excerpts are
in `artifacts/reference/r4-channel-source.txt` and `r4-lifecycle-source.txt`.

## Python implementation

Fable implementation run `20261010-134705-46b154c6` produced only the independent
Python episode validator. Codex QA accepted its snapshot; the coordinator reran all
eight fixed acceptance checks and verified its action encoding against R2 source.
[Immutable acceptance evidence](R4_PYTHON_DELEGATION.md) distinguishes that accepted
snapshot from subsequent coordinator changes for finalized IDs and linear history
parsing. Final project tests and campaign validation exercise the integrated version.

## Lifecycle snapshot review

Review `20261010-135646-bd3f57b6` found two material issues; both were reproduced and
fixed before the reference campaign:

1. A fresh-looking private re-request could permit truncation before the rejected
   player's next decision. Require an `initial` or `resolved` joint boundary and
   preserve the zero-reward transition through the eventual re-request decision.
   The natural `r1-smoke-8` regression cuts exactly at the rejection's budget and
   asserts that collection continues to a safe boundary. The review's example
   submission order was inaccurate: the uncommitted-opponent case is p1-first.
2. Decision IDs could collide between different submission schedules or scripted
   actions under the same initial specification. Bind submission order/controller
   into the episode key and the entire decision body into its ID. Tests cover both
   reversed submission order and different actions with the same scripted key.

The reviewer also suggested that an opponent might terminate a natural game while
a rejected player's choice remains unresolved. This was not substantiated by the
pinned execution path: normal resolution requires both choices. The constructed
first-choice tie now explicitly requires that first choice to have been accepted.
No natural-last-rejection terminal was manufactured to accommodate this claim.

Constructed initial-state probes do not establish natural mid-turn timing. Their
profile, marked intervention message, documentation and fixture index explicitly
limit that evidence; natural episodes separately supply actual continuations and
replacements. Showdown emits pending requests only with a log update, so a marked
message is required when constructing a live-switch request without a faint log.

## Acceptance scope

Regression checks cover serialization/replay, R1-visible protocol compatibility,
private commitment preservation, actor-view stability, exact retries, stale/invalid
submissions, identities, joint/pivot/faint/revival phases, terminal ties, safe cuts,
rehashed semantic corruption, writer interruption/resume, orphan preservation,
manifest mismatch, corrupted bytes and independent campaign accounting. Corruption
tests rebind decision IDs and successor references where needed so failures reach
the semantic invariant instead of stopping only at a stale outer hash.

The full campaign report is the authority for executed counts, hashes and timings.
Mechanics parity with a new simulator, exhaustive reachability, adversarial R5
feature tests, final data splitting and learning behavior remain outside R4.

## Focused follow-up

Review `20261010-140623-18e300f0` completed with no blocking findings in scope. It
checked the revised lifecycle/identities, private-view boundary, Node/Python
semantic parity and campaign accounting. Three proposed edge cases were verified:

- Python canonical strings explicitly use `ensure_ascii=False`, matching Node.
  A direct check with `Sirfetch’d`, accented text and Unicode separators preserved
  raw non-ASCII bytes. Existing R2 cross-language Unicode scalar tests also pass.
- The claimed Unicode failure-line discrepancy confused `str.splitlines()` with
  the implementation's **bytes** `splitlines()`. A canonical failure record with
  U+0085/U+2028/U+2029 counts as one line in both implementations. JSON escapes
  ASCII CR/LF inside messages, so ordinary generated failure records stay single
  physical lines. No accounting change was necessary.
- Python dict equality can treat integers as booleans, but the complete validator
  first enforces the action schema. Direct schema validation of `terastallize: 1`
  raised `ValidationError`. The codec helper is used after that check; no corrupted
  episode is accepted through this proposed route.

The full regression completed with **51 Node + 28 Python tests passing** (7 R2,
6 R3, 15 R4). An initial newly added actor-loader test expected `ValueError` while
the existing R2 loader correctly raised its schema `ValidationError`; the test
expectation was corrected and the full suite rerun. No production behavior was
changed for that test failure. Thirteen saved fixtures additionally passed exact
Node replay and independent Python validation.

Only the Node test file changed after the focused review snapshot (stronger
rehashed-corruption assertions and natural pivot/replacement cutoff checks); all reviewed production source hashes still
matched before the campaign. The review's specific Sirfetch’d species example is
absent from the pinned Gen9 random set catalog. Its Unicode serializer concern was
checked directly rather than injecting a non-generator team into official data.
