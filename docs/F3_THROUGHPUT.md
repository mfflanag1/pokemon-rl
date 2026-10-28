# F3 — CPU batching and Apple GPU comparison

Completed 2026-10-22. **Use persistent CPU workers for the present model.** Eight CPU workers plus batched inference improved warm collection throughput by **1.87×** versus the existing serial collector. Metal executed the policy and learner correctly on the tested inputs, but took **57% longer** than CPU on the matched full-game comparison. This is performance evidence, not a new learning or strength result.

The machine available to this session identifies as **Apple M3 Max**, 14 CPU cores, 30 GPU cores, 96 GB memory, macOS 26.6.2. This differs from the M3 Pro description. Hardware metadata omits machine identifiers.

## What changed

The additive `python/pokemon_throughput/` package runs separate pinned Node/Showdown processes through a reusable `Pool`. Each inference call batches both player projections from active games, with a fixed capacity of twice the worker count. Unused rows contain synthetic zeros and all-false masks. Encoding still happens on CPU; the simulator remains Node/CPU even in the Metal experiment.

The policy is frozen throughout a collection call. Actor seeds depend on episode, side and decision count, never worker identity or completion order. Results return in input order, and merged GAE links retain unique episode/side identities. Wait requests, forced switches, unavailable-choice rejections, natural terminals and administrative cutoff bootstraps retain F2 semantics. A worker failure aborts the entire cohort and closes the pool; no partial result enters an update.

F2 code, checkpoint, plan, receipt and historical handoff remain unchanged. Its original campaign still uses its original serial collector. F3 provides the tested pool for a future campaign; it does not retroactively change the F2 experiment. Adding new executable files changes the live source inventory, so replaying a historically frozen campaign requires its frozen source tree. Preservation checks below concern the original file bytes and archived evidence, not admission of an expanded tree as the old experiment.

## Matched CPU measurements

The predeclared benchmark uses the trained F2 seed-2207 final checkpoint, 32 official train self-play games and 16 development games against random/base-power opponents. Every configuration replays the same 48 specifications three times, with rotated configuration order. Each timing cell contains **4,108 decisions and 48 natural terminals**, with no cutoff. These repeated executions are not independent training samples.

| Configuration | Median warm collection | Decisions/second | Speed versus serial |
|---|---:|---:|---:|
| F2 serial actor/collector | 5.105 s | 804.6 | 1.00× |
| Batched, one CPU worker | 4.134 s | 993.7 | 1.23× |
| Batched, two CPU workers | 3.507 s | 1,171.2 | 1.46× |
| Batched, four CPU workers | 3.244 s | 1,266.4 | 1.57× |
| Batched, eight CPU workers | 2.727 s | 1,506.2 | 1.87× |

The eight-worker range was 2.692–2.834 seconds, versus 5.087–5.165 for serial. CPU runtime: Python 3.14.3, JAX/jaxlib 0.11.2, NumPy 2.5.4, FP32. `jax_threefry_partitionable=True` was verified explicitly.

These times exclude worker startup, warm-up/compilation, report writes and optimizer parity checks. First eight-worker startup took 3.15 seconds plus 0.51 seconds warm-up; serial startup/warm-up took 0.42/0.48 seconds. **Reusing the pool matters:** starting eight processes for every small cohort can erase the warm-throughput gain. This is not a measured 1.87× reduction of the entire earlier 207-second learning campaign.

The separate eight-game serial cProfile replay spent about 0.56 seconds in worker requests and 0.46 seconds in the actor, including 0.22 seconds in feature encoding. These are overlapping cumulative Python-profile measurements with instrumentation overhead. In the parallel trace, `worker_barrier_seconds` includes serialization, simulator work, IPC and decoding; `encode_infer_sync_seconds` combines encoding, device dispatch and synchronization. Coordinator timing is partial; these fields are not pure hardware counters or an exhaustive additive decomposition.

## CPU versus Metal

The isolated `.venv-metal-f3` uses Python 3.12.2, JAX/jaxlib 0.4.34, jax-metal 0.1.1 and NumPy 1.26.4. These pins follow the [plugin's documented compatibility baseline](https://pypi.org/project/jax-metal/); [Apple describes Metal support as experimental](https://developer.apple.com/metal/jax/). The original `.venv` was not modified. Success here does not establish compatibility with our newer JAX 0.11.2 environment or with arbitrary JAX operations.

Both device controls explicitly use partitionable Threefry, the same checkpoint and eight workers. GPU availability was not inferred from installation: the backend was forced to Metal, and all 28 inspected actual actor/update output arrays resided on Metal. The corresponding CPU arrays resided on CPU.

Before full-game timing, the actual network, hierarchical sampler and PPO update ran on both devices. All 273 synthetic actor rows had identical action outputs; float outputs and update states passed declared tolerances. Maximum measured float discrepancy was 4.77e-7. An independent host-generated set of 72 threshold-boundary cases (below/at/above thresholds from 2^-24 through 1/2) passed exactly on each device. These are bounded correctness checks, not exhaustive GPU qualification.

| Same isolated runtime, eight workers | Median warm time for 48 games | Range |
|---|---:|---:|
| CPU | 3.088 s | 3.076–3.100 s |
| Metal GPU | 4.841 s | 4.807–4.941 s |

Metal took 1.57× as long. Each device ran three repetitions; actions, requests, feature hashes, phases and outcomes also matched the current-runtime serial reference exactly. Real-game value outputs passed tolerance checks. This comparison isolates device choice within the older compatible runtime. The current-runtime 2.727-second CPU result is separately labeled because changing JAX/Python versions is an additional intervention. CPU device repetitions preceded GPU repetitions, rather than randomized interleaving; this short benchmark does not establish thermal or long-run stability.

The current 928-input, 64-hidden-unit network is small. Metal's warmed actor call took roughly 4–5 milliseconds in the first probe, including dispatch/synchronization. CPU calls were much shorter. Larger models and larger inference batches may change that balance; no such speedup was demonstrated here.

## Verification and evidence

- **47 focused tests pass:** 22 Node, 10 F1 Python, 10 F2 Python and 5 new collector tests. The new cases exercise delayed workers, variable game lengths/partial batches, player-only input rejection, worker failure cleanup, rejection linkage, cutoff bootstrap, ordered rows and update parity. Existing real Illusion and trapping tests also pass.
- All 15 current-runtime CPU timing cells preserve exact actions, request IDs, features, outcomes and row linkage. Maximum collected value/log-probability errors are 5.96e-8 / 4.77e-7. The first eight original game trajectories in each batched configuration produce the same six-step PPO update within tolerance: maximum parameter difference 1.57e-7, Adam first-moment difference 9.32e-10. Parameter bytes are **not** identical.
- Across the main CPU and device comparisons there are **1,008 measured game executions**, all natural terminals, representing 48 repeated specifications; **86,268 measured decisions**. There are also 42 short warm-up executions and 8 serial profiling replays, plus separately accounted tests and synthetic probes. No policy was selected or improved by this benchmark.
- The original benchmark invocation stopped before collecting any games because its new loader used canonical JSON hashing rather than the existing byte-hash checkpoint binding. The loader was corrected to use the original binding. Its log and partial source snapshot are preserved as `f3-throughput-v1`; the complete CPU result is v2. No seed/game was skipped or replaced.
- The pinned simulator build was verified, and all new Python modules compiled. Independent review findings and their verified dispositions are preserved in the review directory. The research delegate's overbroad numerical claims were rejected rather than treated as evidence.

Primary results: [CPU report](../artifacts/runs/f3-throughput-v2/report.json), [matched CPU device report](../artifacts/runs/f3-cpu-device-games-v1/report.json), [Metal report](../artifacts/runs/f3-metal-device-games-v1/report.json), [completion inventory](../artifacts/reports/f3-throughput.json). Source copies, fixed policy, per-game traces and profile are retained with the CPU run; validation and installation records are under `artifacts/reviews/f3-throughput/`.

## Next use

Use a persistent eight-worker CPU pool as the candidate collector for the next explicitly budgeted learning campaign, preserving its frozen-policy cohort size and checkpoint boundaries. Measure that complete collect/update/checkpoint loop before extrapolating this collection gain. Continue improving observations and the opponent/distribution panel through matched-budget learning comparisons. Revisit GPU use when model size or batching materially increases.

Full G2/G3/G4 and public ladder strength remain open. No public server, final reserved evaluation data or Prime resources were used. GitHub remains on the existing unresolved visibility hold.
