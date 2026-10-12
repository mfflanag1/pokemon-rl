# Restricted-slice research adjudication

Run 20261010-170729-a716560c (Fable), opposite-model Codex QA 3428109ca4.
Verdict: rejected as a complete source-backed executable specification. Independently
verified arithmetic and profile ideas were retained; no code was delegated or integrated.

QA correctly identified that final damage does not identify a unique raw damage roll:
several 85..100 percentages map to each damage bucket. Our reference wrapper records
the actual randomizer draw directly, not by inverting HP deltas. The implementation
uses one exact custom format with Team Preview removed and Terastal Clause applied;
runtime assertions verified first request at turn 1 and absence of canTerastallize.
The exact set fixes male gender, Serious nature and ordered move slots.

The candidate also falsely stated that supplied teams are ignored in gen9randombattle.
Local pinned sim/battle.ts:3163-3166 starts with options.team and immediately returns
it when supplied, before calling the generator. Prior R6 fixtures likewise use
constructed teams through that format. This unsupported claim was not retained.
Using a distinct profile is still required to label the restricted distribution.

Independently read local pinned moves.ts (Tackle/Quick Attack), abilities.ts (Run Away),
pokedex.ts (Rattata), battle-actions.ts damage path, and battle.ts statModify,
calculatePP, modify and randomizer. Fixed stats 105/76/55/92 and PP 56/48 were checked
against the initialized live reference. All 32 crit/roll damage cases were executed
against Showdown and matched the independent Python formula (33..58).
The four-turn/at-most-four-PP bound follows from minimum damage 33, no healing,
guaranteed hits and one move per surviving side per completed turn. It is valid
only for this exact profile; no general mechanics or official-team coverage is claimed.

The candidate's secondary fetched summaries, omitted-source assumptions, optional
phase specification and proposed distribution-test/whole-outcome-tree analysis
were not accepted as evidence. The implemented campaign explicitly claims only
conditional transition/request parity, not chance-distribution, public-history,
JAX or full G2 certification. Its separate static review and executed parity report
are the applicable evidence, not the research candidate's opinion.
