# Restricted CPU transition slice

`gen9-rattata-duel-v1` is a constructed 1v1 duel, not official Gen9 Random Battles.
Both players have the exact same level-50 male Rattata, Serious nature, 31 IVs,
zero EVs, Run Away, no item, Normal Tera type and ordered Tackle/Quick Attack moves.
All sets are public by definition of this fixed profile. Initial HP is 105,
Attack 76, Defense 55 and Speed 92; initial PP is 56/48. The actual pinned reference
format is `gen9customgame@@@!Team Preview,Terastal Clause`. Initialization begins
at the joint move request. Tera, preview choices, switching, cancellation and
all choices except the two move slots are outside this API.

The Node entry point accepts only the exact declared teams. Python accepts only
the exact profile document. Unknown actions and malformed/missing/extra chance
events reject before returning a transition. Immutable state makes failed calls
non-mutating. Manually constructing internal State objects is a diagnostic facility,
not admission of arbitrary teams or statuses. R6 remains unchanged and continues
to reject the complete official format for custom engines.

## Mechanics and scope argument

Pinned `data/abilities.ts` defines Run Away without battle callbacks. These two
moves are Normal physical attacks with power 40 and accuracy 100, priorities
0 and 1. No admitted move/item/ability can change status, stages, form, typing,
weather, HP through healing, or move/ability/item identity. Contact flags have no
reactive partner in this profile. Normal hits Normal neutrally, with STAB.

With fixed stats, base damage after +2 is 26. A critical hit first floors 1.5x,
then the uniform integer damage percent 85..100 is floored; STAB uses Showdown's
4096 fixed-point rounding, with halves rounded down. The full 32 crit/roll cases
have damage 33..58. Crit chance is 1/24; damage draws use 16 equal outcomes.
Priority selects order; equal priorities have equal speeds and use a tie outcome.
The first KO terminates immediately; the canceled second move consumes no PP.

Every executed move hits and deals at least 33 HP. If neither side has fainted,
each has attacked once in the completed turn. A game starting at 105 HP must end
by the fourth turn. Each Pokemon uses at most four PP, below either slot's starting
PP. Thus Struggle, PP depletion, simultaneous fainting, replacements, healing and
status cannot arise from admitted starts. Tests still reject externally supplied
exhausted PP and exercise KO/PP boundary behavior. These exclusions rely on the
exact profile and cannot be copied to a larger profile without new analysis.

## Chance and comparison contract

Showdown runs normally with independently derived mechanics seeds. An instrumented
wrapper records executed hit identities `(turn, actor, move)`, crit flags, raw
damage percentages and resulting damage; equal-priority first actor is recorded
as the semantic tie outcome. The independent Python integer transition consumes
those outcomes, recomputes damage/order/HP/PP/terminal state, and compares every
normalized own-request snapshot. It rejects extra events after KO. Enumeration
also forces each crit/roll pair through the reference damage function.

This is **conditional transition parity**, not a claim that matching raw seeds
produce the same draws in another simulator. Tied order is supplied by the
reference; the campaign does not independently test its distribution. The declared
chance weights are source facts, not fitted from the fixtures. A later JAX engine
must expose the same semantic event contract and separately validate its samplers.

Snapshots cover turn, phase, winner, HP, PP and own move requests. This slice does
not yet compare public reveal messages, opponent HP display, full R5 histories or
all hidden-information perturbations. It is a CPU transition specification and
fixture set for a JAX port. `production_ready` stays false; full G2 remains open.

## R6 obligations retained

- `moves:tackle`, `moves:quickattack`: accuracy, ordinary physical damage, priority,
  PP and KO interruption (`ENG-MOVE`, `ENG-CORE`).
- `abilities:runaway`, `items:` (no held item), `forms:rattata`: exact source/profile admission
  and absence of in-scope callbacks (`ENG-ABILITY`, `ENG-ITEM`, `ENG-FORM`).
- Core ordering/damage/requests: every semantic chance branch, both seats, canceled
  actions, rejected stale/unsupported events, terminal requests and duration bound.

No R6 entry is globally upgraded to implemented-tested by this narrow evidence.
