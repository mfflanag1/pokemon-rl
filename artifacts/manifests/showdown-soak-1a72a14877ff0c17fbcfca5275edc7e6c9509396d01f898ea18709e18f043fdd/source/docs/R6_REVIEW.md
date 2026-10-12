# R6 review and verification

## Independent review

The authorized `claude-review` workflow reviewed a frozen implementation packet
in run `20261010-160002-cbbbf96f` (Fable 5.1). It returned findings after 121.85
seconds. This was a static review, not executed tests or approval. The packet,
request hashes, response, status and the coordinator's dispositions are retained
under `artifacts/reviews/r6/` and in the final source archive.

Verified dispositions:

1. **Pathway ID collision — confirmed and fixed.** The post-expansion draft had
   216 duplicate ID groups. For example, nested `replace` calls at
   `config/formats.ts:3164` shared their starting byte but had different endings.
   Python rejected the draft as designed. IDs now bind both start and end bytes,
   plus owner/file/kind. Node rejects duplicates during generation; Node and
   Python independently check unique identities in the final manifest.
2. **Form enumeration RNG channels — hardened.** Pinned `getForme` uses only
   `sample`; direct source inspection confirmed that fact. Enumeration now
   additionally forbids direct PRNG access and alternate random methods, and
   restores the generator after errors. A fake generator consuming `prng.next`
   proves that the guard fails explicitly. All 509 source entries enumerate to
   the same 561 distinct initial forms present in the R3 sample.
3. **Initial evidence labels — fixed.** Python rejects unknown classes and a
   resealed zero-count entry labelled `observed-initial`. Reachability and initial
   evidence each have independent count consistency checks.
4. **Cosmetic inherited pathways — fixed defensively.** Current pokedex entries
   have no call sites, as the AST scan confirms. Derived forms now explicitly
   have an empty owned-pathway list and retain the base source link.
5. **Stale artifacts — resolved.** Draft runs v1–v3 are not accepted deliverables.
   Final generation uses a fresh v4 destination. Producer drift and changed
   artifacts reject; no immutable draft was overwritten to disguise a mismatch.
6. **Redundant callback traversal — deferred.** It is harmless at this scale and
   does not affect emitted semantics. No unsupported performance claim is made.

Additional coordinator hardening checks exact pathway-bound/family mappings,
family evidence links, report domain counts and source refs for generator methods.
Those checks were added after the review snapshot and tested locally. The review
is not presented as approval of files that subsequently changed.

## Delegation outcome

A separate bounded Fable research run (`20261010-154653-bcbbad31`) timed out after
600 seconds with no result. No research findings were integrated and no QA
completion is claimed. The runner only permits adjudication of completed runs;
the attempted `unverified` adjudication was therefore refused. Its failed status
and an explicit local verification record remain the evidence. This timeout was
not relabelled as research success.

## Verification boundaries

Node executes the 16 scripted reference cases twice and checks semantic outcomes
in both player seats. Python independently checks artifact/source integrity,
corpus roots and linkage. It does not rerun TypeScript or authenticate arbitrary
JSON signed by no trusted party. Resealing cannot grant custom-engine support.

The manifest is a conservative source/data envelope, not a minimal reachability
proof, exhaustive interaction test suite or an accelerated-engine parity result.
All 3,509 entries are unsupported for custom simulation. There are 131 embedded
condition owners and 1,938 callback paths in addition to the 35 global conditions.
The 83 generator item candidates include old/nonstandard branches and literal
aliases; candidate membership does not establish feasible initial generation.

Final full regression: 56 Node tests and 51 Python tests passed. Fresh generation
after regression reproduced all four R6 files byte for byte, and independent
Python validation passed. The R5 report's bound source hashes are unchanged.

Final automated results and sealed artifact identities are recorded in
`artifacts/reports/r6-mechanics.json` and [the handoff](R6_MECHANICS.md).
