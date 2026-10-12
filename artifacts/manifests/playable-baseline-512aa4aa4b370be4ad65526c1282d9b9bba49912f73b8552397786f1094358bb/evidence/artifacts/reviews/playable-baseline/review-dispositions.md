# Review dispositions

The Fable snapshot review completed with findings. It did not execute tests.
The coordinator verified each issue against pinned source and executable checks.
No second Fable approval is claimed for subsequent corrections.

1. **Request-before-public-update concern: rejected for the pinned runtime.**
   sim/battle.ts:1381-1412 stores activeRequest and clears sentRequests without
   emitting. sendUpdates at :3265-3271 sends the public update, then requests.
   server/room-battle.ts:782-816 forwards public updates in order. The added oracle
   test checks startup and next-turn ordering; the final network smoke separately
   records public-turn/type availability when each wire request arrives and checks
   the turn used by every emitted decision. Delaying until a *newer* turn would
   incorrectly stall forced replacements and refreshed requests within a turn.
   Public-server version changes remain unverified and fail visibly when incompatible.

2. **Repeat challenge clearing: verified and fixed.**
   server/ladders-challenges.ts:131-151 removes a challenge and calls update with
   reversed endpoints. :220-237 sends the empty /challenge PM to both parties.
   The prior client ignored the clear message when its own name was the sender.
   Clearing now accepts either endpoint direction; accepting a new challenge still
   requires the bot to be the recipient, an allowlisted sender and exact format.
   A unit test uses the reversed message. The final network smoke plays a rematch
   on the same authenticated connections, then another game on fresh connections.
   Challenges received while a game is active are intentionally not queued.

3. **Local developer format differs in team delivery/validation: scope clarified.**
   Stock Random Battle rules are copied verbatim, but explicit R7-development teams
   replace server team generation. The developer format has its own name and is
   unrated. The two selected network teams pass the real server validator; no claim
   is made that all official generated teams were exhaustively validated by this
   new developer format. Direct benchmark games use gen9randombattle itself.

4. **All popups are fatal: retained conservative behavior.**
   This client reports a server rejection and stops even when a popup might be
   benign. It is documented as a limitation for unattended use, not silently
   handled as a completed match. There is no production reliability/ladder claim.

Additional coordinator checks/fixes:
- Ranked identity prefixes and away suffixes are normalized without dropping an
  @-ranked player; unit challenge tests include this case.
- Revealed Tera typing persists across switches. Illusion replace messages may
  omit HP; preserve previously visible HP and ensure actor state round-trips as
  JSON. The first 64-game production replay exposed this issue; that run is not
  qualified. A new source-bound production run and full replay are required.
- Own-name and seat checks reject foreign requests; actual hidden-reserve tests
  reconstruct equal visible inputs from different hidden teams.
- Local cloud-env bind override was avoided with an explicit port. Stock static
  landing files are copied and checked; the browser UI itself is not automated.
- Wins are derived from winner names, not the simulator remaining-Pokemon score.
  Teams stay fixed while policies swap seats. Replay independently reconstructs
  actor inputs/actions, checks partitions, hashes, counters and win scores.

The snapshot-to-final source deltas are in review-source-deltas.json. New provenance
and receipt wiring were locally verified; they were not in the Fable packet.
Full G2/G3 remain open. No public account, ladder, final reserved data or Prime was used.
