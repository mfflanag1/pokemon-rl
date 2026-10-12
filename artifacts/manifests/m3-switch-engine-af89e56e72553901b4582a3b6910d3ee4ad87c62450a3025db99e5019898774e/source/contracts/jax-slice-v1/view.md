# Fixed-duel player view v1

`pokemon_jax.views.reduce_view(view)` accepts exactly the R1 PlayerProjection wire
keys `side,request_id,turn,request,protocol`, with side p1/p2 and protocol a list of
strings. It has no simulator state, seed, chance or opponent-request argument.
Return a dict with only `header` and `history` of Python integers.

Header: [turn, terminal, relative_winner, own_hp, opponent_hp, own_tackle_pp,
own_quickattack_pp, opponent_tackle_seen, opponent_quickattack_seen, legal0, legal1].
Relative winner: 0 live, 1 own win, 2 opponent win. Terminal masks are [0,0].
History: eight rows, padded with six -1s, chronological executed hits, each
[turn, relative_actor(0 own/1 opponent), move(0 tackle/1 quickattack), crit(0/1),
raw_damage, target_hp_after]. On a KO, raw damage cannot be inferred from HP;
therefore history raw_damage MUST instead be **HP lost** (capped at pre-hit HP).
The column is named hp_lost in all implementations. No sampling outcome/roll is
visible. Record at most eight hits; profile's actual maximum is seven.

Start HP 105, PP [56,48]. Both exact HPs are public in this pinned Custom Game.
Public switch lines initialize exactly one Rattata per side; both include exact
HP 105/105. Public move lines reveal and consume the executing actor's PP only.
A queued move canceled by KO has no move event and must not be revealed. Critical
lines precede damage. Damage strings are N/105 or '0 fnt'. Winner lines use P1/P2.
A move must have exactly one damage line with optional single critical line.
Faint lines refer to the just KO'd target. Normal metadata lines may be ignored:
empty, t:, gametype, player, gen, tier, rule, raw, teamsize, start, upkeep, request.
Unknown protocol commands/mechanics, impossible HP increases, malformed move/
critical/damage sequences, duplicate winner, wrong side, missing initial switches,
wrong turn, extra top-level keys must raise ValueError. Switches after initialization
are unsupported. Turn lines progress 1,2,..., max 4. Terminal stays on current turn.

For live views validate current request has one own active Pokemon and one active
request, two enabled moves in order tackle/quickattack with PP equal to reconstructed
own PP, maxpp 56/48, own condition equal to public own HP, no canTerastallize,
teamPreview, forceSwitch or wait. Only own request.side.id may appear. At terminal,
PlayerProjection may retain a stale previous own request; ignore its PP but still
reject a request for the other side. Embedded protocol |request| payloads are
transport metadata; only request side identity must be checked, never treat their
contents as public events. Outer turn must equal last protocol turn, request_id
must equal number of request lines and be a nonnegative int. Input mutation forbidden.

This is a restricted semantic projection, not byte-for-byte protocol parity or a
replacement for the R5 full-format history features. Fixed species/ability/item/
move set are public profile constants; perturbing them requires rejecting admission.
