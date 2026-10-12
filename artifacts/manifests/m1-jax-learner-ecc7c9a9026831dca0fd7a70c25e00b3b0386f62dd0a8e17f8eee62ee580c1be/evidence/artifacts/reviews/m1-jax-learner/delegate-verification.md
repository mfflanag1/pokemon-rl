# Bounded JAX numerical core: verified acceptance

Run `20261010-180004-f1e0eebe`, Fable 5.1; opposite-model Codex QA: accept.
The declared controller command explicitly sets PYTHONPATH=python and passed all
five supplied tests. Before integration, every selected input hash matched the
request, and git apply --check passed. Only the owned core.py file was added.
The integrated core remains byte-identical to the candidate.

Independent verification on the integrated source:
- Eight numerical tests now pass, including exact upper/lower clip-boundary
  derivatives, stopped target gradients, finite differences and linear-model
  parameter gradients checked with a NumPy chain rule.
- Frozen campaign v1 passed 96 PPO/mask cases, 96 GAE cases, 384 Adam steps and
  96 accumulated table updates. Fixed tolerance: atol 3e-6, rtol 3e-5.
- Max logit-gradient absolute error: 8.940696716308594e-08.
- Max GAE-advantage error: 9.5367431640625e-07.
- Max Adam-parameter error: 1.1920928955078125e-07.
- All six original-budget synthetic canaries pass >=99%, with JAX autodiff updates
  and the unchanged original CPU synthetic collector.

These are executed checks, not inferences from QA. CPU/x64 admission is enforced
by the trainer/campaign; raw numerical kernels assume admitted FP32 inputs.
FP32 global-norm reduction differs from NumPy's FP64 reduction within the frozen
numerical tolerance. No bit-identical cross-backend optimizer claim is made.
