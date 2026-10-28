# Bounded-value CPU canary v1

This additive algorithm changes the M3 value head to `sigmoid(h @ vw + vb)`.
The six parameter tables, shared tanh trunk, actor logits, exact four-leaf sampler,
key addresses, observation boundary, Adam and PPO settings remain the same.
Equal parameters give equal actions/log probabilities; learning changes the
shared trunk, so equality after updates is neither expected nor claimed. Zero
raw value weights/bias initialize predictions to 0.5, unlike the M3 linear head.

Terminal rewards remain win=1/loss=0. With gamma=1, lambda=.95 and values in
[0,1], a linked nonterminal GAE return is a convex combination of a future value
and future return. The last cutoff return is its bounded bootstrap; the terminal
return is its terminal reward. Returns therefore stay in [0,1] in exact
arithmetic, with 2e-6 tolerance for the existing FP32 implementation. No clipping
of returns is introduced. Tests include gaps/waits, terminals and cutoff tails.

The independent NumPy reference differentiates the full network analytically and
uses the sealed NumPy clipped-PPO oracle in a smooth actor domain: maximum legal
logit spread <=8. This conservative bound keeps conditional masses above the
2^-24 actor floor. It does not certify derivatives at that floor. Separate tests
check inherited sampling/log-probability behavior at extreme logits. Directional
finite differences validate all six NumPy gradients; JAX gradients are compared
with this reference. Padded rows are removed or sanitized before matrix products.

The fixed canary uses all three plan seeds and eight updates per seed. Initial
and final checkpoints have separate format/admission identities and exact
original-next-update learner/buffer/final-state/metric replay. Audit outputs are
discarded and are not counted as training progression. A later audit replays the
entire eight-update trajectory for every seed. Source/runtime drift fails closed.

This is still the sealed 12-turn switching slice. Cutoff games reset; the game
does not continue to turn 13. A bounded constant value can remain unanchored, and
the sigmoid can saturate. The canary establishes bounded engineering behavior,
not calibration, removal of the diagnosed failure's cause, sustained stability,
strong play or full simulator coverage. Episode continuation is the next design
gate. No public server, final reserved data, Prime or strength evaluation is used.
