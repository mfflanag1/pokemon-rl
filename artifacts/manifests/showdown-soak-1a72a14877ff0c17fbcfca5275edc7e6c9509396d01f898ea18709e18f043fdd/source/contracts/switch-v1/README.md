# switch-reference-1

This is a new two-member fixed-team reference scope, not a change to the sealed
Rattata duel. Use the pinned stock Gen 9 Anything Goes format with Team Preview
removed and Terastallization forbidden. Custom Game's debug mode exposes exact
opponent HP, so it is not used. Require reportExactHP=false and debugMode=false.
Only sets in profile.json are admitted, with distinct species and original names
A/B. Six species orders times four speed-variant assignments give 24 teams.
These constructed teams do not consume official train/dev/final partitions.

Moves have no accuracy misses at ordinary unmodified accuracy. Vine Whip/Tackle/
Scratch are physical; Water Gun/Incinerate are special. Type effectiveness, STAB,
critical hits, damage rolls, speed/ties, and Overgrow/Torrent/Blaze at <=1/3 HP
are obligations. Incinerate's berry/gem removal is unreachable because all items
are empty and no move/ability can create or transfer them. No status, stage,
weather or terrain creator exists in this closed roster. Twelve resolved turns
are below the minimum initial PP (24 for Incinerate), so exhaustion is excluded.
A switch can still stall: cutoff is truncation at a fresh joint boundary, never
a terminal draw. These exclusions hold only for the exact admitted profile.

Actor boundary: an R1 player view has exactly side, request_id, turn, request,
protocol. It contains that side's current private request and its own channel
history. Cross-player phase/seed/teams/audits never enter it. Full snapshots use a new structured player-input adapter with R5's public event
reducer. The old R5 schema/encoder requires six-member teams and cannot accept
this profile. The new adapter retains the latest own roster during wait requests
and verifies each request against its own history. It is diagnostic structured
input, not yet a fixed-shape tensor layout or trained model. Values and policies
receive the same input.

The new action codec has four booleans and uses zero-based indices:
0/1 = the active request's first/second move; 2/3 = switch to original A/B.
Request pokemon order changes on switching; find A/B by exact own ident suffix,
not current list position. Encode a switch as that mon's current 1-based request
position. Moves require non-disabled, positive PP. Wait has no legal action.
Force-switch has only non-active, non-fainted switch choices. No pass/Struggle/
Tera/revive/trap/team-preview action is admitted. Reject unknown indices, bools,
stale request IDs and extra submission fields before writing to Showdown.

The JSON golden file produced by fixtures records current player views and
codec masks/commands. Python must independently reproduce those and reconstruct
structured player inputs from full player-only snapshots. Treat fixture audit fields as
privileged diagnostics, never as input to the actor adapter. Reject extra fields
on the view. Direct protocol/private-request mutation is not a valid information
counterfactual: compare independently initialized real games with the same public
history and different hidden sets instead. Legitimate reveals must change input.

No new JAX transition, sampler, learner or strategic-strength claim is part of
this reference milestone. All older source and evidence remain immutable.

Conditional damage grid: 18 attacker/defender/seat rows, three HP values around
the ability threshold, two moves, two critical outcomes and sixteen damage rolls.
Only these diagnostic calls inject HP, willCrit and bounded random(16); the real
Showdown damage/rounding code executes unchanged. Natural episodes use unmodified
Showdown randomness. Accuracy checks can consume PRNG draws even at 100 accuracy;
future port work must specify its chance law without assuming raw-stream alignment.
