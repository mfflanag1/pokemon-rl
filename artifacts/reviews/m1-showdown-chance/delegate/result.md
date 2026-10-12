Nothing further is needed — the file is written and I've checked it against the frozen tests by inspection. Result:

## Result

Created `python/pokemon_showdown/mapping.py` (only owned path touched). Nothing else exists or was modified.

**`kernel(words)`** — `@jax.jit`, pure, 1-D uint32 input, returns `{'bucket24','crit','roll','tie0','tie1'}`:
- `bucket24` (int32) via 16-bit limb split: `hi=w>>16`, `lo=w&0xFFFF`, `m=lo*24`, `s=hi*24+(m>>16)`, `bucket24=s>>16`. All intermediates `< 2**21`, so no uint32 wrap; no float, no 64-bit type, no modulo. Proof written in the module docstring, including derivation that `crit ⇔ w < 178956971`, i.e. conditional mass `178956971/2**32` as in `contract.json["critical"]`.
- `crit` (bool) = `bucket24 == 0`; `roll` (int32) = `100 - (w>>28)`; `tie0` (int32) = `w>>31`; `tie1` = `1 - tie0`.
- Trace-time dtype/shape check raises `TypeError`/`ValueError` without coercion.

**`map_words(words)`** — strict wrapper: raises `RuntimeError` unless `jax_enable_x64` is False and `jax.default_backend()=='cpu'`; accepts only nonempty 1-D `np.ndarray`/`jax.Array` of dtype uint32; rejects every other type/dtype/shape (`TypeError` for type/dtype, `ValueError` for shape/empty) with no lossy coercion; then calls `kernel`.

Docstring explicitly disclaims sodium PRNG / raw-seed / word-schedule / trajectory equivalence per `contract.json["streams"]`.

**Hand-check against fixture boundaries** (inference, not execution): `w=178956970` → `hi=2730, lo=43690, m=1048560, m>>16=15, s=65535, >>16=0` ✓; `w=178956971` → `lo=43691, m=1048584, m>>16=16, s=65536, >>16=1` ✓.

## Limitations
- I did not run the unittest controller or any Python; acceptance checks are the controller's. No test files were changed.
- `jax.config.read("jax_enable_x64")` and `isinstance(x, jax.Array)` assume JAX ≥ 0.4.1; if the pinned JAX is older, swap to `jax.config.jax_enable_x64` / `jax.core.Tracer`-free check.
- Environment errors in `map_words` use `RuntimeError` (not `TypeError`/`ValueError`), so they are distinguishable from input rejection; adjust if the controller expects otherwise.
