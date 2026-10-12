# switch-engine-1

The sealed `switch-v1` roster and twelve-turn cutoff remain the admitted scope.
This port adds explicit conditional chance, independent scalar and JAX transitions,
a bounded player-only observation and a four-action learner smoke after parity.
No old implementation or receipt is edited.

Original member slots A/B never reorder internally. State holds species IDs 0/1/2,
speed variants, HP, PP, active original slots, phase (0 joint, 1 replacement,
2 terminal), turn, group and winner (-1 live, otherwise absolute winning seat).
Only the fainted side acts during replacement; -1 is the waiting action. A joint
step resolves both commitments atomically. Switches precede moves; within the same
priority class faster outgoing active Pokémon acts first, with explicit tie bit.
A fainted actor does not execute or spend PP. Finish replacement before advancing
the turn. Terminal/cutoff states are absorbing in the batched kernel; checked host
submission rejects another active decision. Imported arbitrary states are not a
supported resume format: checkpoints restore the complete learner at a completed
update, with all terminal and cutoff games already reset; live-game resume is unsupported.
An interrupted checkpoint leaves the prior published file intact and preserves
the `.partial` as evidence. A later save to that path fails explicitly until the
partial is preserved under another name; campaign failure does not silently resume.

History has 32 padded rows of nine integers:
`turn, actor, kind, member, species, move, affected_member, HP_after, crit`.
Kind 0 is a switch, kind 1 an executed damaging move. Species is 0/1/2;
move is 0/1 or -1 for a switch. HP is the switched-in member's HP for switches
and the opponent target's HP for moves. Initial send-outs are represented in the
initial observation, not in history. No damage roll is actor-visible. The player
projection converts actor to relative 0=self/1=opponent and HP to exact own or
ceil(100*HP/maxHP) for opponents (clamp 100 to 99 when HP is below max); unused history rows are -1. Faint is implicit in
HP zero. Every reachable event fits: at most 24 joint actions plus two replacement
switches. Terminal winner labels are absent from the actor input.

Compact PlayerView fields: own[2,9] = species+1, HP, maxHP, attack, defense,
special attack, special defense, speed, active flag; pp[2,2]; opponent[2,4] =
species+1 (0 unknown), public HP percent (-1 unknown), active flag, fainted flag;
context[5] = turn, own phase (0 move,1 replacement,2 wait,3 terminal), own active
original slot, opponent active original slot, history count; history[32,9]; mask[4].
Both policy and value receive only this same projection. PP is reconstructed from
the observer's executed moves, including while waiting; own HP incorporates private
damage events after a stale last full roster request. Opponent reserve and speed
variants must not enter initial input before they cause public evidence.

The learner collects whole capped games under one frozen actor, then updates.
Transitions link only the same player and episode; waiting contributes no decision.
Turn-13 cutoffs bootstrap from that player's fresh view under the collecting actor,
then explicitly reset. They are never terminal draws or evaluation scores.
The 64-update three-seed run is an engineering smoke, not a learning promotion.

Chance capture records all raw critical/roll words and repeated queue tie shuffles.
Gen 9 can sort tied moves twice before execution; the final shuffle before the first
executed action supplies the effective tie outcome. The JAX sampler uses independent
semantic-event Threefry words with the same finite-word outcome law, not the sodium
draw stream. Twenty-four fresh traces are compared against completely unpatched
Sessions, including private views and final PRNG state at every step.

Categorical sampling uses a root split (moves vs switches) and two child splits.
Each split uses the sealed binary FP32 rare mass (floor 2^-24) and exact uint64
word threshold represented by two uint32 words. Each leaf's mathematical law is
the product of those conditional masses; stored likelihoods sum the corresponding
log/log1p values. Tests cover all masks, extreme tails and positive reachability.
Moderate logits agree with the sealed NumPy four-action softmax/PPO gradient;
floored extremes intentionally follow the declared conditional tree law.
