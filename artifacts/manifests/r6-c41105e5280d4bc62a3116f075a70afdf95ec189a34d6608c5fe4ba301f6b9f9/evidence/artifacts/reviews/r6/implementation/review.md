**Provisional verdict: findings (no blocking integrity flaw within declared scope; one plausible ID-collision bug introduced by the latest expansion, plus a stale-artifact hazard).**

## Findings

**1. Medium — pathway ID collision after retaining all calls** (`source-index.mjs:63-91`, `manifest.mjs:84`)
ID = `hashJSON([owner, file, start_byte, kind])`. Nested call expressions sharing a start position and both classified `source-owned-call` (curried/IIFE forms such as `fn(a)(b)`, `getHandler(x)(args)`, `(cond ? f : g)(y)()`) produce identical IDs. Node emits duplicates silently; `mechanics.py:98` then rejects the whole bundle ("Duplicate pathway"). Previously only sink-named calls were emitted, so this was unlikely; now every CallExpression in `sim/*.ts` is included. Verification: after regeneration, assert `new Set(manifest.pathways.map(p=>p.id)).size === manifest.pathways.length` in the Node test, or include `end_byte` (or an ordinal) in the ID preimage and in the Python recomputation at `mechanics.py:103`.

**2. Medium-low — initial-form enumeration only intercepts `generator.sample`** (`manifest.mjs:32-53`)
Branches of `getForme` that draw via `this.prng.*`, `this.random`, or `this.randomChance` are invisible; `calls <= 1` passes with `calls === 0` and only one form is recorded. The line-77 check (every R3-observed form must be declared) partially mitigates, and the pikachu test gives one positive witness, but an unobserved alternate form reached via another RNG method would be labeled `not-established-initial` without any failure. Suggest asserting `calls === 1` for species whose `getForme` body contains any `random`/`sample`/`prng` identifier (via `stringLiterals`/AST), or documenting this limit in `item_candidate_scope`-style prose.

**3. Low — Python validator does not cross-check `initial_class`** (`mechanics.py:85-87`)
`reachability` is tied to `observed_count`, but `initial_class` is free text relative to counts; a resealed manifest could label a 0-count entry `observed-initial`. Not a support-gate risk (support is always rejected) but it weakens the "independent verification" claim. Add `(observed_count>0) == (initial_class=='observed-initial')` and membership in the four README values.

**4. Low — derived cosmetic entries copy `pathways` from parent** (`manifest.mjs:118`)
Works only because pokedex entries never contain calls; if they ever did, `mechanics.py:108` fails with "Pathway linkage mismatch". Set `pathways: []` explicitly on derived entries.

**5. Operational — stale artifact vs. producer drift** (`scripts/mechanics-inventory.mjs:13`, `mechanics.py:149-150`)
The pending Python tests read `artifacts/runs/r6-mechanics-v1`, which predates the expansion. `report.producer` hashes will now mismatch (`Producer source drift`), and `save()` refuses to overwrite changed `manifest.json`. The run directory must be removed/renamed before regeneration; otherwise "full tests pending" cannot pass.

**6. Nit** — `callbacks()` computes per-callback `pathways` (line 102) that are never emitted; `manifest.mjs:96` re-walks each entry. Harmless.

Fixture assertions checked: Ditto maxhp 237 (base 48, 31 IV, 0 EV, L100) is correct; sleep-talk/zero-to-hero tapes and null-choice reversal are consistent; `|t:|` normalization and `prng.getSeed()` checkpoints make replay deterministic.

## Scope limitations
- Cannot confirm `hashJSON`/`hash_json` canonical equivalence, `sha256` UTF-8 encoding of string fragments, or `UPSTREAM == root/vendor/pokemon-showdown`.
- Did not inspect pinned Showdown source; finding 1 is a structural risk, not a confirmed instance.
- Line counting: TS treats `\r`/U+2028/2029 as line breaks, Python counts only `\n`; harmless unless such characters occur in pinned source.

## Verdict
Findings. Claims in README/scope are appropriately hedged; the support gate is fail-closed as described. Address finding 1 (and the regeneration workflow in 5) before trusting the post-expansion artifact.
