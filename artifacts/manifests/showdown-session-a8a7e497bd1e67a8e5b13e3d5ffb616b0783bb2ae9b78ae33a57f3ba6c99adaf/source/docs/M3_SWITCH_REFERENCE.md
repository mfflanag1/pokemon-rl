# M3: switching and hidden-set reference scope

The new `gen9-starter-pairs-v1` profile expands the public Rattata duel to two
Pokémon per side, meaningful move/switch choices, and hidden reserve/speed variants.
This milestone establishes the reference contract and fixtures. Its JAX transition,
fixed-shape actor, categorical policy, learner and qualification are still pending.
Full G2/G3 remain open. Local CPU is sufficient; Prime is unnecessary.

## Admitted battle profile

The roster contains Bulbasaur (Vine Whip/Tackle, Overgrow), Squirtle (Water Gun/
Tackle, Torrent), and Fennekin (Incinerate/Scratch, Blaze). Each team contains two
distinct species in a fixed order. Level 50, male, Serious, all IVs 31, no item;
each member has either zero EVs or 252 Speed EVs. Six species orders times four
variant assignments give 24 teams. These are constructed development teams,
not samples from the official reserved dataset.

Grass, water and fire attacks create a matchup cycle. Normal attacks offer an
alternative into resisted matchups. Switching changes the recipient of an incoming
attack and costs the switching Pokémon its attack. Low-HP abilities add a damage
threshold. This creates useful choices to test; no claim of balance or strategic
strength follows from the design alone.

The oracle uses the pinned stock Gen 9 Anything Goes format, removes Team Preview
and forbids Terastallization. Custom Game was rejected because its debug mode
reveals exact opponent HP. Both `debugMode` and `reportExactHP` must be false.
Opponents receive percentage HP and no initial reserve species/stat disclosure.
The roster and possible sets are public; the selected reserve and speed variants
are private until their effects or public actions provide evidence. Move sets are
fixed by species in this scope: this is not arbitrary hidden-moveset coverage.

The [profile and contract](../contracts/switch-v1/README.md) specify four semantic
actions: the two active move slots and switches to original A/B. Showdown changes
private request order after a switch, so commands resolve original names to the
current request position. Both Node and Python implement this mapping independently.
Waiting has an empty mask; faint replacement has only the living bench switch.
Stale IDs, mismatched view hashes, duplicate seats, extra submission fields,
concurrent steps and invalid actions reject before a second commitment.

Twelve resolved turns are below the minimum initial PP of 24. Games still live
at the next fresh joint boundary are **truncated**, not scored as draws. Replacement
requests finish before that boundary. No new reward/GAE treatment of persistent
truncated games has been implemented; that is a future learner requirement.

## Measured reference evidence

The accepted run is `artifacts/runs/m3-switch-reference-v1`. Its complete fixture
and summary files reproduce byte for byte in a second fresh Node process. Python
independently verifies the full action/view/actor evidence and recounts coverage.
The source-bound full regression passes **264 tests (75 Node + 189 Python)**.
The [completion receipt](../artifacts/reports/m3-switch-reference.json) binds source,
run, review, regression and all preserved drafts. The run report content hash is
`a12d08c0296b8b3f2eac878e6193c85afcdee9637f9143c9cfbd5ca38741dc19`;
the fixture content hash is
`74e64ca4abb8391b1ad6c8172b874b507f624f558b4486362b918d032fd08ab0`.
All sealed predecessor source/evidence is checked against its archive. The 128
final teams remain reserved and no final episodes are read or generated.
The fixture roster contains 72 scripts: every team leads one attack-only, one
mixed-switch and one switch-only case against a fixed opponent pairing. Each case
is repeated with opposite submission order, requiring identical mechanics audits,
player protocols and outcomes. This is not an exhaustive 24×24 full-game campaign.

- 24 terminal games and 48 explicit truncations; 1,656 submitted decisions.
- 74 replacement frames with a waiting opponent; 720 voluntary switches.
- 31 public critical-hit events, 206 super-effective and 596 resisted events.
- 85 attacks begun at or below the low-HP ability threshold and 138 joint frames with tied
  active speeds. These are exercised cases, not chance-frequency acceptance tests.
- 1,826 live/wait player views have exact independent Node/Python mask and command
  agreement. All 5,426 fixture/probe/counterfactual views reconstruct the structured actor input.
- 1,152 independently initialized information probes cover both observer seats,
  all 24 own teams and all 24 opponent teams. Grouping by equal visible information
  gives 1,008 identical initial views across hidden reserve/lead-speed choices and
  864 identical post-turn views across hidden reserve choices.
- 576 own-view positive controls distinguish private speed variants. Forty-eight
  retained reveal examples yield 24 paired actor-input controls: the same history
  before revealing two different reserves, different input after the reveals.
- 3,456 conditional damage values cover all nine species matchups in both seats,
  three HP values around the ability threshold, both moves, both critical outcomes
  and all sixteen damage rolls. Boosted signature damage changes across the
  threshold; the normal move does not. These diagnostic calls inject HP and chance
  outcomes; natural fixture games use untouched Showdown randomness.

Every nonterminal player view is checked for foreign private requests and exact
opponent HP leakage. Status, healing, stage changes, Tera and unprojected split
messages fail this scope. The latest own private roster remains available while
waiting. Policy and value receive identical structured input.

The old R5 schema requires six-member teams and is preserved. The new adapter
uses only its public-history reducer, with a separate request/history validator.
It retains all observed encounters and public events; this is **not yet a fixed-shape
JAX feature layout**. Fixture audits and seed/team metadata are privileged evidence
and cannot be passed as actor views. Full outcome-bearing terminal histories reject.

## Mechanics obligations for the next port

| Obligation | Reference evidence now | Still required for JAX |
|---|---|---|
| Type chart, STAB, physical/special integer damage, crits and damage rolls | Conditional damage grid and natural events | Exact grid parity and reachable transition parity |
| Overgrow/Torrent/Blaze | Below/at/above threshold, normal-move controls, natural attacks | Damage modifier order and threshold parity |
| Move/switch ordering and speed ties | Natural scripts, opposite submission order | Explicit chance law, phase/state transition parity |
| Voluntary and faint-replacement switching; wait | Natural phases and independent codecs | Fixed-shape pending/request state and cancellation behavior |
| Own exact / opponent percentage HP and private sets | Negative HP checks, counterfactuals and reveal controls | New observation tensors and player-view differential tests |
| PP and terminal/truncated outcomes | Request/tape linkage and explicit cutoff | Collector accounting, bootstraps or whole-episode training choice |

Incinerate cannot remove an item in this roster because every item is empty and
no admitted mechanism creates or transfers one. There are no status/stage/weather/
terrain/hazard/heal/recoil/pivot/revival creators. These are closed-profile exclusions,
not conclusions about full-format reachability. The R6 full-format inventory remains
unsupported. Accuracy checks may consume Showdown PRNG words at 100 accuracy;
a future port must specify chance laws without assuming raw-stream alignment.

## Reproduction and preservation

```sh
# Use a new output directory for each run; do not run alongside npm test.
npm run reference:switch -- artifacts/runs/NEW-switch-reference
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/m3-switch-reference
.venv/bin/python scripts/seal-switch-reference.py artifacts/runs/NEW-switch-reference
```

The campaign verifies the M2 live/archive evidence, inherited runtime and final-data
reservation; records current source admission; generates the complete fixture bundle
twice in separate Node processes; requires identical compressed fixture/summary bytes;
and performs independent Python validation. The sealer binds this to the full
regression and review evidence. Current progress documents are updated before sealing;
historical handoffs and all predecessor implementation/evidence remain unchanged.

Drafts v1/v2 failed coverage-reader assertions (fainted `status='fnt'`, then absent
active Pokémon during replacement). Their failure records remain preserved.
Draft v3 passed natural fixtures; v4 adds the damage grid and strict codec; v5
retains all counterfactual views for independent reconstruction. Only
the final source-bound run is the completion evidence.

The [claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md) was used.
The delegated codec delivery failed because the coordinator repeated `--owned` and
only its final argument was registered. No candidate was integrated; controller QA
and adjudication are unavailable for a failed run. The local replacement and actor
boundary received independent snapshot review. Details and verified dispositions
are retained in `artifacts/reviews/m3-switch-reference/`.

Next implement the versioned scalar/JAX state, damage and switching transitions,
with an explicit chance interface and player-view reducer. Require this reference
catalog and fresh hidden-state differential tests to pass before training a policy
in the expanded scope. Keep the sealed binary-policy/public-duel implementation intact.
