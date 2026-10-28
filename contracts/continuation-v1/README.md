# Continuation simulator v1

This is a bounded extension of the starter-pair mechanics. The team roster,
semantic actions, stats, integer damage and event-addressed chance law are retained.
The separately versioned simulator admits turns1–60 and complete history up to128
rows. It carries live games through fresh turn13/25/37/49 requests. Turn61 is an
administrative truncation, never a draw, terminal reward or automatic reset.

Five 12-turn windows exercise repeated continuation while retaining a small exact
state space. Each side has two members, so at most two forced replacement groups
are possible across the whole game. Sixty joint turns plus those replacements
need at most62 groups and122 public move/switch events. The history is neither
truncated nor wrapped. Each member has at least80 total initial PP and can execute
at most60 attacks; total PP exhaustion and Struggle are outside this admitted
horizon. Other excluded mechanics remain excluded. This is not full-game coverage.

At a window boundary, finish forced replacements before pausing. The mechanical
player view retains its fresh legal request; the collection window's separate mask
is empty while paused. Paused lanes are exact no-ops and retain group counters.
Serializing a live simulator carry includes all private simulation fields, complete
public history, group addresses, game IDs, master seed, plan and source/runtime
binding. Only a player's projected view may enter a later actor. Carry bytes are
audit-only and are not policy input. A carry is a simulator checkpoint, not a
learner checkpoint; no optimizer, rollout buffer or update state is implemented.

The old M4 network's history shape/normalization and empty-game checkpoint format
are incompatible. They are intentionally not passed the expanded state. Later
learner integration needs a new encoder, live-game rollout state, detached
bootstrap values, same-player return links and original-next-update replay.

## Fixed fixture catalog

144 fixtures = all24 own teams × six styles, with a fixed opponent permutation
and source-bound Showdown seeds. No seed selection or policy optimization occurs.

| Style | Primary coverage |
|---|---|
| Signature attack | Type damage, low-HP abilities, crits and faint replacements |
| Normal attack | Physical damage, ordering and terminal outcomes |
| Always switch | Five boundaries, 120 retained events, explicit turn61 truncation |
| Switch through24, then attack | Actual terminal outcomes after two window boundaries |
| Periodic switch/attack | Mixed request timing and post-boundary replacements |
| Hash-selected legal action | Deterministic mixed legal sequences across both seats |

Coverage gates additionally require forced replacement at a window's last turn,
outcomes and attacks after12, and history beyond32. Every fixture is replayed on
unpatched Showdown; passive random instrumentation may observe but not choose
outcomes. The scalar and JAX ports then match every state, executed event, legal
mask and both compact player views. Full trajectory replay checks each serialized
fresh boundary and its original next simulator step, including frozen no-ops.

Host validation rejects malformed shapes, bounds, padding, phases and counter
linkage; it does not prove arbitrary supplied state reachable. The scalar oracle
assumes admitted states and checks action/chance legality and history capacity.
Scientific evidence comes from fresh reachable reference traces and their replay,
not from accepting a self-consistent checksum as proof of reachability.

There are no training updates, strength matches, public server calls, final
reserved-data accesses, GPU or Prime use in this milestone. Full G2/G3/G4 remain
open. The 60-turn ceiling and self-play learning/strength are separate future gates.
