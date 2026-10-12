Verdict: **revise**

The candidate is useful research, but it does not meet the requirement that every asserted binding rule have directly inspected official-primary support.

Concrete findings:

- The central Smogon announcement was not directly inspected. The candidate admits its forum content came through a “summarizing fetch tool.” My direct request to the [announcement](https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/) returned HTTP 403. Therefore its wording, date, scope, and registration requirement must be marked **unverified from the primary page**, not placed under “Binding rules with directly inspected official support.”

- It likely overstates the registration requirement’s operational status. The reported phrases “we’re creating a registry,” “details … to be confirmed,” and unsettled enforcement suggest an announced future mechanism, not proof that a usable registry presently exists. “Before any public-server game, obtain registration” is unsupported and apparently impossible until the mechanism is published. The defensible checklist item is: **do not deploy until staff publishes or confirms the registration procedure; current applicability to challenge-only bots remains unknown.**

- The [Pokémon Showdown rules page](https://pokemonshowdown.com/rules) was directly accessible and supports only the ordinary rules quoted: no cheating/system-gaming and no staff impersonation. It contains no bot registration, naming, direct-challenge, or ladder-bot requirements.

- The candidate improperly labels source-code behavior as “binding rules.” Code can document server behavior, but throttles and matchmaking gates are not necessarily policy obligations. More importantly, the cited raw/blob pages were not demonstrated to be the live production version, and the candidate itself says production deployment is unverified.

- “Ordinary ladder matching and direct challenges are not blocked by this code” is an overbroad absence claim. The candidate inspected only selected files and explicitly did not inspect `/search` and `/challenge` command handlers. It should say **no additional block was found in the inspected portions; end-to-end behavior is unknown**.

- The table’s “registered-account requirement: none” has the same defect. Absence from `prepBattle` does not prove absence elsewhere in authentication or command handling.

- The throttle checklist is too absolute. “≤5 concurrent games, ≤12 battle preps/3 min/IP, ≥10 s between outgoing challenges” should be described as **observed defaults in the cited pinned source**, subject to configuration, exemptions, production version, and exact counter semantics—not official bot policy.

- The direct-challenge versus ladder distinction is consequently not established at the policy level. The sound conclusion is: **the inaccessible announcement reportedly contains no challenge-only exception; ladder-specific code appears to exclude designated bots from certain suspect/ladder-tour matchmaking; ordinary ladder and direct-challenge authorization remain unverified.**

- The candidate appropriately separates community material, records several unknowns, makes no unsupported claims about the supplied local implementation, and reports no external mutations.

Required revision: move the announcement-derived requirements and all production-behavior claims into an explicitly unverified section; distinguish policy from implementation defaults; remove the claim that registration can presently be obtained; and make both direct-challenge eligibility and ordinary-ladder eligibility unknown pending direct primary-page inspection and confirmation of the live registration mechanism.