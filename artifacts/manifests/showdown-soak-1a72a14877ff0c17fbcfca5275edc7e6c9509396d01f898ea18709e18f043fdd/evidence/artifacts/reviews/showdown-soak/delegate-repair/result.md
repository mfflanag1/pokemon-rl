Both owned output files are written. Nothing further is needed from the tools — I have no shell available, so I cannot run `node --test` or a byte-level `diff`; the controller will do that.

## Result

**Created (only the two owned paths):**
- `packages/showdown-soak/schedule.mjs`
- `tests/showdown-soak/schedule.test.mjs`

**The fix** (`packages/showdown-soak/schedule.mjs`, inserted after the range check at the candidate's line 67, before `session` is computed):

```js
  // -0 is a safe integer that passes the range check; canonicalize it to +0
  // before deriving any fields so every emitted number is strictly equal to 0.
  if (Object.is(index, -0)) index = 0;
```

Why there: `Number.isSafeInteger(-0)` is `true` and `-0 < 0` is `false`, so `-0` legitimately passes validation, but it would then propagate — `Math.floor(-0 / n)` and `-0 % n` both yield `-0`, and `String(-0)` is `"0"` so only the numeric fields were affected. Placing the normalization after validation (so error messages and rejection behavior are unchanged) and before the first derived field means `index`, `session`, `within_session`, `pair`, `mirror`, `tactical_team`, `baseline_team` all come out as `+0`, satisfying `assert.equal(gameSpec(-0, false).index, 0)` under `node:assert/strict` (which uses `Object.is` semantics for primitives). No other line in the file changed.

**Test file:** copied verbatim from the candidate, including the existing `-0` assertion at line 390; no test was added, removed, or modified.

## Limitations
- I did not execute the tests or a diff — I have no shell in this environment, and the task assigns acceptance checks to the controller. Suggested verification: `node --test tests/showdown-soak/schedule.test.mjs` and `diff artifacts/reviews/showdown-soak/delegate-initial/schedule.mjs packages/showdown-soak/schedule.mjs` (expect exactly a 3-line addition) plus `cmp` on the test file (expect no output).
- Byte identity of the test copy relies on my transcription from the Read output; the one plausible divergence is trailing-newline handling, which `cmp` will reveal immediately.
- `plan.json` and the input snapshots under `artifacts/` were read but not touched.
