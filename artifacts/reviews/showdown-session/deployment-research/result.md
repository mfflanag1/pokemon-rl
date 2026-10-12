I have what I need; no further fetches required. Final result below.

# Deployment checklist: named automated bot on Pokémon Showdown (main server)

Scope: read-only research, 2026-10-11. No logins, contacts, registrations, or challenges were made. "Official" = Smogon staff announcement, pokemonshowdown.com pages, or the `smogon/pokemon-showdown` / `smogon/pokemon-showdown-loginserver` repositories. Everything else is marked third-party.

## A. Binding rules with directly inspected official support

**1. Registering the bot is mandatory (announced policy; mechanism not yet published).**
- Source: https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/ — Smogon Forums › Site Info › Announcements, title "★ Initial policy on bots", post #1 by Django (NU Leader; Tournament Director/Senior Staff badges). Thread is closed to replies.
- Date: page shows only "Friday at 6:13 PM"; no absolute timestamp was exposed. Inferred 2026-10-09 (today is Sunday 2026-10-11); a search engine snippet independently described it as posted "about a day" earlier. Treat the absolute date as inferred.
- Quotes: "we're creating a registry for people to register their bots, and this is mandatory." / "Running a bot without this is a bannable offense" / "the first concrete step we will be taking is creating a register of self-declared bots." / "Given this, running an unregistered bot will be against the rules." / "The details here are to be confirmed." / "If you are working on these projects and want to help, then DM chaos and he will point you in the right direction"
- Purpose stated: registry is to preserve the human-only parts of the ecosystem — "(suspect reqs, usage stats, general laddering)".
- Enforcement caveat, post #2 by chaos (Owner), "Friday at 8:14 PM", edited 8:15 PM: "There is some support among staff for just registering bots that don't comply and not necessarily banning them." / "I'm not sure what the enforcement mechanism is going to be atm, could be case by case".
- Checklist action: before any public-server game, obtain registration via the channel staff publish (currently only "DM chaos" is named). Not doing so is stated to be bannable, with enforcement explicitly unsettled.

**2. No AI assistance in tournaments.**
- Same post #1: "using AI tools to assist during tournaments is ghosting" … "would result in an immediate ban." / "This is already covered by ghosting rules".
- Checklist action: never use the bot (or its policy) to advise a human in a Smogon tournament game.

**3. Global rules apply; the rules page itself says nothing about bots.**
- Source: https://pokemonshowdown.com/rules (no last-updated date shown). Sections: Main rules, Chat rules, Username rules, YouTube. The page "contains no mention of bots, automated play, scripts, clients". Relevant quoted fragments: Main rule #4 includes "by intentionally losing against yourself or a friend in a ladder match, by tricking your opponent into forfeiting, etc"; Username rules: "Names may not impersonate staff".
- Checklist action: avoid any self-play/alt win-trading on ladder; bot name must not impersonate staff. No official naming convention for bots (e.g., "Bot" suffix) exists on this page.

**4. Server-enforced restrictions on accounts designated as bots (official code, very recent).**
- Login server: `smogon/pokemon-showdown-loginserver` commit bac1c6c "Support bot user type", 2026-10-03T22:19:27Z, `src/user.ts`: `} else if (group === 10) {` → `userType = '10';`. The flag comes from the `users.group` database field; the code shows no self-service path to set it (who sets it is **not shown** — inference: staff/admin).
- Server: `smogon/pokemon-showdown` commit 2619487 "Support bot user type", 2026-10-03T22:17:21Z, `server/users.ts`: `//   10: bot` … `this.isUserBot = true;`. Follow-ups: 51ad80f "Support virtual suspect formats" 2026-10-07; ef3d535 "Fix isBot tracking" 2026-10-09 (`rating.isBot` recorded on ladder updates); pinned commit 942cda7 "Fix virtual formats" 2026-10-10T03:32:18Z (matches `showdown.pin.json`).
- In the pinned `server/ladders.ts` (https://raw.githubusercontent.com/smogon/pokemon-showdown/942cda71d14bf0618bd57991708a2e2734b6daba/server/ladders.ts): `if (user.isUserBot && this.virtualCustom === 'suspect')` → popup "Bots are not allowed to queue for suspect tests.\n\nIf you feel this designation is in error, please contact staff by typing /helpticket in chat." Matchmaking comment (51ad80f): "if one user is a bot and one is a suspect user or a ladder tour player, do not match".
- Checklist action: expect a designated bot account to be excluded from suspect-test queues and from matching against suspect/ladder-tour (forced-prefix) searchers; ordinary ladder matching and direct challenges are not blocked by this code. Whether the production server actually runs this commit is **unverified** (inferred from the pin selection date).

**5. Server-enforced throttles (apply to everyone, pinned code).**
- `server/monitor.ts`: "Due to high load, you are limited to 12 battles and team validations every 3 minutes." (per IP, `count <= 12` in 3 min) and "Due to high load, you are limited to 5 games at the same time." Both skipped if `Config.nothrottle`/`noipchecks`.
- `server/ladders.ts` `makeChallenge`: "You challenged less than 10 seconds after your last challenge!"; "This user already has 3 pending challenges." + "You must be autoconfirmed to challenge them."; "The user '<name>' is not accepting challenges right now."
- Checklist action: ≤5 concurrent games, ≤12 battle preps/3 min/IP, ≥10 s between outgoing challenges.

## B. Direct challenge vs. ladder — explicit distinctions

| Aspect | Direct challenge | Ladder (`/search`) | Basis |
|---|---|---|---|
| Registry requirement | Policy says "Running a bot" — no challenge/ladder carve-out stated | Same | Smogon post #1 (no distinction made) |
| Code gating | 10 s throttle, pending-challenge cap, blockchallenges | `format.searchShow` required ("Error: Your format … is not ladderable."); bot excluded from suspect queues / suspect & ladder-tour matches | pinned `ladders.ts` |
| Registered-account requirement | none in `ladders.ts` (`prepBattle` checks no `registered`/`named`) | none in `ladders.ts` | pinned `ladders.ts` (command-layer checks not inspected) |
| Policy priority | — | Registry exists to protect "general laddering", suspect reqs, usage stats | Smogon post #1 |

Unknown: whether staff intend any formal difference (e.g., challenge-only bots exempt). The post gives none.

## C. Third-party / non-binding material (not rules)
- Policy Review discussion https://www.smogon.com/forums/threads/bots-are-now-capable-of-suspect-reqs-do-we-care.3787411/ (freezai, Aug 25, 2026; 4 pages, only page 1 read). UT (all titles "Alumnus"): "Using a bot to get your suspect reqs already counts as cheating in a suspect test and voids your reqs." — discussion, not an announcement.
- mikumiku37 thread https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/ (Shao, Oct 2, 2026): user project; no staff rules stated.
- Kaiepi "Pokémon Showdown Bot F.A.Q." https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e (last active Dec 21, 2022; linked from the official README): community doc; no allow/deny rules; mentions `/j botdev` room (in a comment) and asking an RO to trust a locked bot.
- Ladder Seasons thread https://www.smogon.com/forums/threads/pokemon-showdown-ladder-seasons-season-one-announced.3740067/ (shiloh, Tiering Lead, Mar 30, 2024; post #3 by Hecate): "any form of ladder boosting or cheating will be punished harshly in this system" — predates bots policy, doesn't mention bots.
- PROTOCOL.md (https://raw.githubusercontent.com/smogon/pokemon-showdown/master/PROTOCOL.md): technical only; "gives no rules of conduct for third-party clients or bots".

## D. Blocked / inaccessible / unverified
- Smogon thread absolute timestamps: not exposed in fetched HTML; RSS (`…/index.rss`) returned "The requested page cannot be represented in this format." → date **inferred** 2026-10-09.
- My guessed Announcements forum URL (`/forums/forums/announcements.11/`) returned 404 — no bearing on findings.
- https://pokemonshowdown.com/news/ : only the first ~100k chars read (posts Oct 9, 2026 back to Mar 30, 2025); no bot/registry item found in that range.
- `server/chat-commands/moderation.ts` truncated at 100k chars; no bot-designation command found in the read portion. How `users.group = 10` gets set is **unverified**.
- Whether `/search` or `/challenge` command handlers add registration checks outside `ladders.ts`: not inspected.
- https://pokemonshowdown.com/pages/privacy: exists; no bot/account text.

## E. Remaining unknowns (do not treat as rules)
1. How/where to register (form, thread, or DM only); approval criteria; turnaround; whether existing accounts get `group=10` applied by staff.
2. Any naming/disclosure requirement for the bot account (none published).
3. Enforcement start date and whether unregistered challenge-only bots are tolerated ("case by case" per chaos).
4. Rate/volume expectations beyond server throttles; whether bot ladder ratings count toward usage stats (code records `isBot` on ladder updates, implying separate handling, but no published statement).
5. Whether production `sim3.psim.us` runs the pinned bot-type code.

## Limitations
All forum/page content came through a summarizing fetch tool; quotes are as returned by it and should be spot-checked against the live thread before being cited externally. No claims are made about the local project's behavior beyond the supplied handoff (`docs/PLAYABLE_BASELINE.md` notes "current server bot requirements … remain deployment checks", consistent with the above). No files were written and no external state was changed.

Sources: [Smogon: ★ Initial policy on bots](https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/) · [PS Rules](https://pokemonshowdown.com/rules) · [pinned ladders.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/942cda71d14bf0618bd57991708a2e2734b6daba/server/ladders.ts) · [pinned monitor.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/942cda71d14bf0618bd57991708a2e2734b6daba/server/monitor.ts) · [pinned users.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/942cda71d14bf0618bd57991708a2e2734b6daba/server/users.ts) · [commit 2619487](https://api.github.com/repos/smogon/pokemon-showdown/commits/2619487) · [commit 51ad80f](https://api.github.com/repos/smogon/pokemon-showdown/commits/51ad80f) · [commit ef3d535](https://api.github.com/repos/smogon/pokemon-showdown/commits/ef3d535) · [loginserver commit bac1c6c](https://api.github.com/repos/smogon/pokemon-showdown-loginserver/commits/bac1c6c) · [loginserver user.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown-loginserver/master/src/user.ts) · [PROTOCOL.md](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/PROTOCOL.md) · [PS News](https://pokemonshowdown.com/news/) · [Policy Review: Bots are now capable of suspect reqs](https://www.smogon.com/forums/threads/bots-are-now-capable-of-suspect-reqs-do-we-care.3787411/) · [mikumiku37 thread](https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/) · [Ladder Seasons](https://www.smogon.com/forums/threads/pokemon-showdown-ladder-seasons-season-one-announced.3740067/) · [Kaiepi Bot FAQ (community)](https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e) · [PS README](https://cdn.jsdelivr.net/gh/smogon/pokemon-showdown@master/README.md)
