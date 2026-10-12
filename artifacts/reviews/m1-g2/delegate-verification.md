# Delegated reducer verification and disposition

Run: `20261010-173129-67c1f57e`, Fable 5.1, bounded implementation of
`python/pokemon_jax/views.py`. Opposite-model QA: Codex, verdict revise.

The runner's declared check failed during import because the coordinator omitted
`PYTHONPATH=python` from its check argv. This is a coordinator invocation error,
not evidence of a reducer logic failure. The run does not qualify as an accepted
implementation delivery under the recorded checks; record it as rejected, with
this cause preserved rather than quietly replacing check evidence.

Independent executed checks: the same six supplied tests pass against the untouched
isolated candidate when run with `PYTHONPATH=python`. The candidate also passed
1,000 fresh Showdown games / 18,000 padded player-view comparisons after integration
in the first successful campaign. The patch added only the owned reducer file.

Verified QA findings: false-valued unsupported request keys and embedded requests
without a side were accepted. Added rejection and regression coverage. Reported
own profile facts were not cross-checked; added checks for request-visible details,
ability, item, moves, Tera type and stats. Nature/IVs/EVs are not in Showdown requests;
those are enforced by existing exact-team admission, not inferred by this reducer.
QA's suggestion that every such property is available in the request is overbroad.

The coordinator integrated the checked reducer, then made these local corrections.
The original delegated artifact and its QA remain unchanged. Final regression and
campaign receipts cover the corrected integrated source separately. No claim that
the original failed runner invocation passed or that model review executed tests.
