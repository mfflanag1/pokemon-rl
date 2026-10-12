# Prime Intellect: where it helps this project

Checked 2026-10-10. **Use Prime as a candidate compute provider; keep our JAX simulator and PPO learner independent of it.** A later LLM-agent benchmark is a separate, optional integration. These recommendations are our fit assessment, not a claim that undocumented integrations are impossible.

## Product decisions

| Offering | Decision | Reason and trigger |
|---|---|---|
| GPU pods / marketplace | Evaluate at first GPU milestone | SSH-accessible instances can host our own runtime. Choose by measured complete-cycle throughput per total dollar, available memory, CPU/RAM, storage and reliability. |
| Hosted Training | Outside the core plan | Documented loop targets language-model completions and rewards, not our small noncausal JAX encoder and PPO value/GAE loop. Revisit only for a language-model policy experiment. |
| `prime-rl` | Do not adopt as our trainer | Its documented inference/orchestrator/trainer design is built around vLLM and LLM training. Replacing it to support our arrays, JAX model and rollout semantics offers no demonstrated saving. |
| Verifiers / Environments Hub | Optional later benchmark | Useful if we want language models to play through a Showdown observation/action interface, compare agent strategies, or share that evaluation. It does not supply a verified Gen9 GPU simulator. |
| Sandboxes | Optional CPU test infrastructure | Could package reference Showdown workers or isolate independent bot evaluations. Start locally; remote lifecycle overhead is unjustified until CPU evaluation is a bottleneck. |
| Prompt optimization / GEPA | Only for a future text-policy branch | A prompt optimizer does not replace gradient training of our categorical tensor policy. |
| Multi-GPU / clusters | Defer | First establish one-GPU performance. Parallel independent seeds may be more useful than distributed training of one small network. |

The GPU CLI documents availability queries, instance creation and SSH access. JAX compatibility still needs a real test of the selected image, driver and accelerator; pod access alone is not a verified benchmark. [Prime GPU provisioning](https://docs.primeintellect.ai/cli-reference/provision-gpu)

The hosted environment documentation describes completion generation, multi-turn orchestration and reward-driven language-model updates. `prime-rl` separates inference, orchestration and training. Those interfaces differ from our direct array-based simulator and per-player PPO buffers. This is the basis for declining integration now. [Hosted environment model](https://docs.primeintellect.ai/hosted-training/environment-model), [prime-rl architecture](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md)

`prime-rl` does support extension: its algorithms documentation gives a custom loss import hook, a PPO-clip loss example and custom algorithm classes. That is extensibility within a token/LLM training architecture, not a documented plug-in path for our JAX encoder. The architectural mismatch, rather than an alleged absence of PPO or extension hooks, is the reason to keep our learner separate. [Algorithms and custom loss](https://docs.primeintellect.ai/prime-rl/algorithms)

## Existing Pokémon reference worth inspecting

[HyperPotatoNeo/pokemon-RL](https://github.com/HyperPotatoNeo/pokemon-RL) documents Showdown orchestration, text-state translation, verifiers integration and LLM training through `prime-rl`. Its README lists Gen9 OU production configurations and Gen1 Random Battle examples. It is a useful candidate reference for reference-server orchestration and a later LLM opponent/evaluation branch; its documented text interface is not our JAX tensor simulator. We have inspected the README, not run its tests or validated Gen9 Random Battle compatibility, current Prime API compatibility, or Hub publication. Pin and audit dependencies and licensing before source reuse. Do not copy its training assumptions into our PPO loop without review.

## Current service and documentation caveats

Prime's pricing page says shared LoRA training would stop accepting new runs on **October 8, 2026**, and describes a transition to dedicated runs. We have verified the published notice, not account-level enforcement. The full-finetuning documentation describes closed-beta access and native `prime-rl` configurations. It would be inaccurate to describe all hosted training as LoRA-only. [Service notice](https://docs.primeintellect.ai/hosted-training/models-and-pricing), [full fine-tuning](https://docs.primeintellect.ai/hosted-training/full-finetuning)

Documentation is in transition: one environment overview still shows an older `prime-rl` source configuration, while the full-finetuning page says that shape has been removed. Pin the actual CLI/framework versions and use their validated schema if we later implement that branch. Do not copy a hosted example into the custom JAX learner.

Sandboxes are general isolated compute, not inherently tied to text rewards. The current overview says GPU sandboxes are coming soon and require an explicit account grant; filesystem checkpoints do not capture running process memory. They may help CPU reference/evaluation work, but are not our assumed GPU training service or a replacement for application checkpoints. [Sandbox overview](https://docs.primeintellect.ai/sandboxes/overview)

## What was actually checked

- Local `prime` CLI reports version **0.5.40**. A command emitted an update notice naming **0.10.0**; no upgrade was performed.
- Read-only Hub search for Pokémon returned an **authorization error**.
- Read-only H100 availability queries returned **authorization errors**; an empty result printed alongside those errors is not evidence of zero stock.
- Public documentation and repository material were inspected. No suitable Gen9 Showdown Hub environment was verified, and no candidate was installed or smoke-tested. This is an unresolved discovery task, not proof that no such environment exists.
- No GPU, sandbox, hosted-training run, or hosted evaluation was provisioned. Live account-specific offers, availability and total prices remain unknown.

Local authentication needs attention only when we choose to use the service. It does not block the reference/data sprint.

## Concrete GPU decision procedure

1. After the restricted engine and learner pass their gates, obtain a current offer with exact accelerator, GPU memory/count, CPU/RAM, disk, region, provider, interruption terms and total hourly/storage costs. Prime's availability API exposes offer metadata; example prices in its documentation are not live quotes. [Availability API](https://docs.primeintellect.ai/api-reference/check-gpu-availability)
2. Select a pinned CUDA/JAX-compatible runtime. Within a bounded benchmark, verify device detection, numerical canaries, compile time, memory, collect/update decisions per second and checkpoint/restore.
3. Compare `total billed cost / useful training decisions`, including setup, idle CPU bottlenecks and evaluation. Choose Prime only if the offer is competitive and operationally suitable.
4. Keep manifests, checkpoints and logs in durable portable storage. Prove resume before using interruptible capacity; record lost work and recovery cost.
5. Enforce the selected run/time cap and teardown procedure. Confirm pod and storage billing separately after the run.

For a later language-model branch, expose only `PlayerView` and semantic legal choices to an agent, use the same pinned Showdown rules and frozen evaluation opponents, and score battle outcomes. Validate that wrapper with a small evaluation before considering Hub publication or hosted RL. This preserves a fair comparison with our structured policy without putting text generation into its training path. [Verifiers repository](https://github.com/PrimeIntellect-ai/verifiers)
