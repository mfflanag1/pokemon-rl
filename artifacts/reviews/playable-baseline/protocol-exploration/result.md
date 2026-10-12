No further requests needed — I have the evidence for every acceptance item. Final report follows.

# Showdown protocol research for a minimal Gen9 Random Battle singles bot

**Scope/limits up front.** The workspace snapshot contains only the six supplied files; its git history is a single "Input snapshot" commit, so the upstream pin is **unrecorded**. Anything below marked *[upstream master]* was read from `raw.githubusercontent.com/smogon/pokemon-showdown/master/...` on 2026-10-11 via a summarizing fetcher (quotes ≤125 chars, no line numbers), so treat those as strong-but-unpinned evidence. Supplied-file citations are `file:line`. I ran no servers and no code.

## 1. Transport / native WebSocket framing

**Supported facts**
- Native WS endpoint is `ws://HOST:8000/showdown/websocket` (`PROTOCOL.md:6-12`). The server's own upgrade handler accepts only `GET` + `Upgrade: websocket` on paths matching `^/showdown(?:/[^/.]+/[^/.]+)?/websocket/?$` and otherwise returns `400` (`sockets.ts:432-446`). The 2-segment form is the SockJS-framed session endpoint; the bare `/showdown/websocket` is sockjs-node's "raw websocket" (no `o`/`a[...]`/`h` frames, **no heartbeats** — sockjs-node README *[upstream]*). SockJS prefix is `/showdown` (`sockets.ts:404`).
- Client→server frame: `ROOMID|TEXT`, ROOMID may be empty, TEXT may contain newlines (`PROTOCOL.md:37-46`). The socket worker **drops** empty frames, frames >100KB (sends `|popup|…100KB`), frames starting with `{`, and frames with no `|` or ending in `|` (`sockets.ts:503-519`). So `|/trn …` (empty room) is valid; `foo` (no pipe) is silently dropped.
- Server→client frame: optional `>ROOMID\n` then newline-separated `|TYPE|DATA` lines; empty lines ignored; `>` prefix omitted for lobby/global (`PROTOCOL.md:56-71`). Room broadcasts are split per spectator/player channel using `|split|` (`sockets.ts:249-268`), so players receive their own private view.
- Chat throttle *[upstream users.ts]*: `THROTTLE_DELAY = 600` ms, `THROTTLE_BUFFER_LIMIT = 6`, `THROTTLE_MULTILINE_WARN = 3`; excess queued messages are dropped with a "typing too quickly" notice. `--no-security` sets `nothrottle` (see §7) which also lifts the multiline cap.

**Recommendation**: send at most ~1 command per 600 ms; never batch more than 5 lines in one frame; implement your own keepalive/reconnect since the raw endpoint sends no heartbeats.

## 2. Client authentication

**Supported facts**
- On connect the server sends `|challstr|KEYID|CHALLENGE` (`users.ts` *[upstream]*: `connection.sendTo(null, '|challstr|${keyid}|${connection.challenge}')`, `keyid = Config.loginserverpublickeyid`, default `4` per `config-example.js:193`). The *whole* string after `|challstr|` (contains `|`) is the `challstr` parameter (`PROTOCOL.md:235-248`).
- Registered login: POST `https://play.pokemonshowdown.com/api/login` with `name, pass, challstr`; response starts with `]` then JSON; finish with `/trn USERNAME,0,ASSERTION` (`PROTOCOL.md:242-254`). `/trn` → `user.rename(name, token, registered, connection)` *[upstream core.ts]*.
- Token verification *[upstream users.ts `validateToken`]*: token = `data;sig`, `data` splits into `[signedChallenge, signedUserid, userType, signedDate, signedHostname]`; rejected if `signedChallenge !== connection.challenge` or `|signedDate − now| > Config.tokenexpiry || 25h`; signature checked by `Verifier.verify` against `Config.loginserverpublickey` (`config-example.js:191-208`). Failure → `|nametaken|NAME|Your authentication token was invalid.`
- Local-dev bypass: `if (!token && Config.noguestsecurity)` accepts `/trn NAME` with no token, as an **unregistered** user (`config-example.js:456-463`; *[upstream users.ts]*). Success is signalled by `|updateuser|USER|1|AVATAR|SETTINGS` (`PROTOCOL.md:256-262`).

**Inference**: a stock local server (default `noguestsecurity=false`) still depends on the public login server to mint assertions even for unregistered names; the only fully offline path is `noguestsecurity=true` / `--no-security`.

## 3. Challenge acceptance

**Supported facts**
- Commands: `/utm TEAM` (optional; `null` for random formats), `/challenge USER, FORMAT`, `/accept USER`, `/reject USER`, `/cancelchallenge USER` (`PROTOCOL.md:444-466`). `/accept` resolves the target via `splitUser(target)`/`pmTarget`, looks up `Ladders.challenges.search(user.id, target.id)`, requires `chall.to === user.id`, then `Ladders.acceptChallenge(connection, chall)` which runs `prepBattle(connection,'challenge')` (team validation, popups on failure) and `Ladders.match(...)` → `Rooms.createBattle` with `challengeType:'challenge'` *[upstream core.ts, ladders.ts]*. Challenge battles are unrated (`room-battle.ts:556-557`, `rooms.ts:2250-2253`) and have a 300 s challenge-timer bank (`room-battle.ts:48-50`).
- Challenger throttle: 10 s between challenges unless `Config.nothrottle` *[upstream ladders.ts]*; "already challenging someone" rejection exists. `Config.pmmodchat` (default `false`, `config-example.js:367`) gates `/challenge`.
- Room created as `battle-<formatid>-<N>` (`rooms.ts:1647-1659`), both players auto-joined (`rooms.ts:2293-2298`); on connect you receive `|init|battle\n|title|…` plus log (`rooms.ts:2011-2014`) and, if a request is pending, `|request|…` (+`|sentchoice|…` if you already chose) (`room-battle.ts:957-962`).

**⚠ Divergence (important):** the supplied `PROTOCOL.md:428-442` (and upstream master's PROTOCOL.md) say challenges arrive as `|updatechallenges|JSON{challengesFrom,challengeTo}`. But upstream **server** code (`server/ladders-challenges.ts` `getUpdate`/`send`/`updateFor`) emits `|pm|FROM|TO|/challenge FORMAT|TEAMBUILDERFORMAT|MESSAGE|ACCEPTBUTTON|REJECTBUTTON`, and `|pm|FROM|TO|/challenge` (no args) when it is cancelled/accepted/rejected; `updatechallenges` appears nowhere in upstream `users.ts`, `ladders.ts`, or `ladders-challenges.ts`. The official client (`panel-chat.tsx` `handleLine`/`parseChallenge`) parses exactly that `|pm|` form by `split('|')` and has no `updatechallenges` handler. **Recommendation**: handle both; in practice key off `|pm|` + `/challenge ` prefix, and verify the pinned server by grepping it for `updatechallenges`.

## 4. rqid / noCancel / undo

**Supported facts**
- Server assigns `rqid` per request: increments `this.rqid`, injects it into the JSON, and stores `isWait = request.wait ? 'cantUndo' : false` (`room-battle.ts:803-816`). `rqid` is absent when using the sim directly (`SIM-PROTOCOL.md:753-756`).
- `/choose CHOICE|RQID` (`SIM-PROTOCOL.md:758-761`): server splits on `|` (`room-battle.ts:624`). Rejections: `isWait` not boolean → `|error|[Invalid choice] There's nothing to choose` (`:627-630`); all players waiting **or** `rqid` mismatch → `…too late to make a different move…` (`:631-636`). A missing `rqid` is accepted. On acceptance `isWait=true` and `>pN CHOICE` is written to the sim (`:637-640`). `/undo` has symmetric rules (`:642-661`).
- `noCancel`: a sim-level request flag, set when `!supportCancel || !multipleRequestsExist` *[upstream battle.ts `getRequests`]*; `room-battle.ts` never reads it (grep: no match) — it is advisory to clients. Server-side undo is blocked by `choice.cantUndo` (set by `maybeTrapped`/`maybeDisabled`/`maybeLocked` paths or when a side's choice is finalized without `supportCancel`) producing `|error|[Invalid choice] Can't undo: A trapping/disabling effect would cause undo to leak information` *[upstream side.ts/battle.ts]*, which `room-battle.ts:799-801` maps to `isWait='cantUndo'`.

**Recommendation**: always append the latest `rqid`; never undo; treat `wait:true` requests as no-ops (sending anything yields "nothing to choose").

## 5. Forced switch and Revival Blessing

**Supported facts**
- Forced-switch request: `{forceSwitch:[true], side:{…}, noCancel?, rqid}`; the non-switching side gets `{wait:true, side:{…}}` *[upstream battle.ts]*; `isWait` for the waiting side is `'cantUndo'` (`room-battle.ts:811`). Answer with `switch N` (1-based slot or name) (`SIM-PROTOCOL.md:700-723`). `pass` is only legal when `forcedPassesLeft > 0`; bare `switch` auto-picks the next unfainted slot *[upstream side.ts]*.
- Revival Blessing: request `side.pokemon[i].reviving === true` for the active mon (`pokemon.ts getSwitchRequestData` *[upstream]*); server checks `slotConditions[pos]['revivalblessing']`; you must send `switch N` where N is a **fainted** slot, else `Can't switch: You have to pass to a fainted Pokémon`; bare `switch` picks the first fainted slot *[upstream side.ts]*.

## 6. Unknown trapping rejection and retry

**Supported facts**
- Request flags: `trapped:true` only when the sim is certain and a switch is otherwise possible; `maybeTrapped:true` when uncertain (`pokemon.ts getMoveRequestData` *[upstream]*). `SIM-PROTOCOL.md:728-739`: an invalid choice yields `|error|[Invalid choice] …`; if the attempt **reveals** info you get `|error|[Unavailable choice] …` followed by a fresh `|request|`.
- Mechanics *[upstream side.ts `emitChoiceError`]*: on `switch` with `pokemon.trapped`, the request entry is updated (`trapped=true`, `maybeTrapped` deleted) and re-emitted with `update=true`; the error type is `[Unavailable choice]` iff the request changed. Server side (`room-battle.ts:796-818`): `[Unavailable choice]` does **not** match the `[Invalid choice]` branch, so `isWait` stays `true` until the follow-up `|request|` arrives, which bumps `rqid`, resets `isWait=false`, and — because `request.update` is truthy — **does not** restart the timer (`:816`). A plain `[Invalid choice]` resets `isWait=false` with the **same** `rqid`.

**Recommendation (retry algorithm)**: on `[Unavailable choice]`, wait for the next `|request|`, re-decide from it (switch now excluded), send with the **new** rqid — a retry with the old rqid hits the "too late" error at `:633`. On `[Invalid choice]`, retry immediately with the same rqid. Choosing `switch` when `maybeTrapped` is set also sets `cantUndo`, so don't plan on undo.

## 7. Local server startup without modifying upstream

**Supported facts** *[upstream unless noted]*
- `./pokemon-showdown start [PORT] [--no-security] [--skip-build]`; `start` runs `build()` then loads `dist/server/index.js`; Node must have `Set.prototype.intersection` (`package.json` engines `>=22.18.0`). The numeric positional becomes the port (`sockets.ts:586-594`) and **forces `Config.ssl = null`** (`sockets.ts:69-71`).
- `build` copies `config/config-example.js` → `config/config.js` if missing; `config/*` (except `formats.ts`), `logs/`, `dist`, `node_modules` are gitignored — so editing `config/config.js` or using CLI flags leaves upstream untouched. `config-loader.ts` applies `FLAG_PRESETS`: `--no-security` → `nothrottle`, `noguestsecurity`, `noipchecks`.
- Default `port 8000`, `bindaddress 0.0.0.0` (`config-example.js:8,16`); `lazysockets` (`:35`) disables listening; `subprocesses.network/simulator` can be `0` to run in-process (`:88-89`).

**Recommendation**: `./pokemon-showdown start 8000 --no-security` with two bot connections doing `|/trn Alice` / `|/trn Bob`, then `|/challenge bob, gen9randombattle` and `|/accept alice`. Set `backdoor=false` (`config-example.js:391-400`) if the box is reachable.

## 8. Deployment constraints

- Official server: `wss://sim3.psim.us/showdown/websocket` (`PROTOCOL.md:10`); login server is hard-coded (`config-example.js:187-191`); registered accounts need password auth; 600 ms throttle; 10 s challenge throttle.
- Timer: off by default for challenges unless `Config.forcetimer` or `/timer on` (`room-battle.ts:607`; `config-example.js:375-381`); disconnect timer 60 s, 150 s/turn cap (`room-battle.ts:46-53`). Leaving the room sends `|request|null` and marks you inactive (`room-battle.ts:1018-1026`).
- Sim crash → `|bigerror|The simulator process crashed…` and battle ends (`room-battle.ts:754-758`); a sim error within 30 min of boot locks the server down (`:823-831`).

## 9. Minimal end-to-end tests
1. Connect raw WS; assert `|challstr|` arrives; bad path returns HTTP 400.
2. `|/trn Name` under `--no-security` → `|updateuser|Name|1|…`; without the flag → `|nametaken|`.
3. Alice `/challenge`, Bob observes `|pm|…|/challenge gen9randombattle|…`; Bob `/accept alice` → both get `>battle-gen9randombattle-N` `|init|battle` and a `|request|` with `rqid`.
4. Turn loop to `|win|`/`|tie|` using `move 1|rqid`; assert no `[Invalid choice]`.
5. Force a faint: assert one side gets `forceSwitch`, the other `wait:true`; sending a choice on the wait side yields "nothing to choose".
6. Stale-rqid test: send a choice with `rqid-1` → "too late" error.
7. Trap test (Arena Trap/Shadow Tag/Magnet Pull seed): send `switch` → `[Unavailable choice]` + new `|request|` with `trapped:true` and `rqid+1`; retry succeeds.
8. Revival Blessing seed: `reviving:true` → `switch <fainted slot>` accepted; `switch <healthy slot>` rejected.
9. Disconnect mid-battle; reconnect; assert `|request|` + `|sentchoice|` replay (`room-battle.ts:958-961`).

## 10. Risky edge cases
- `|updatechallenges|` never arriving on current servers (docs stale).
- Non-heartbeat raw endpoint → silent half-open sockets.
- Throttle drops silently (only a `|raw|` notice).
- `rqid` reset to 0 after battle end (`room-battle.ts:763`).
- Moves with `disabled:true` or `maybeDisabled` trigger the same `[Unavailable choice]`+re-request path.
- Team-preview is **not** used in gen9randombattle but `team` requests exist for other formats (`SIM-PROTOCOL.md:657-668`).
- Port positional disables SSL; `0.0.0.0` bind exposes a backdoor-enabled server.

**Unresolved**: exact pinned upstream commit; `TeamValidatorAsync` behaviour for a missing team in random formats (PROTOCOL.md says `null` is fine; not verified in code); `THROTTLE_DELAY_PUBLIC_BOT` value.

Sources: [PROTOCOL.md (upstream)](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/PROTOCOL.md), [sim/side.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/sim/side.ts), [sim/battle.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/sim/battle.ts), [sim/pokemon.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/sim/pokemon.ts), [server/users.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/users.ts), [server/chat.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/chat.ts), [server/chat-commands/core.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/chat-commands/core.ts), [server/ladders.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/ladders.ts), [server/ladders-challenges.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/ladders-challenges.ts), [server/config-loader.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/config-loader.ts), [server/index.ts](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/server/index.ts), [pokemon-showdown launcher](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/pokemon-showdown), [build](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/build), [package.json](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/package.json), [.gitignore](https://raw.githubusercontent.com/smogon/pokemon-showdown/master/.gitignore), [sockjs-node README](https://github.com/sockjs/sockjs-node), [client panel-chat.tsx](https://raw.githubusercontent.com/smogon/pokemon-showdown-client/master/play.pokemonshowdown.com/src/panel-chat.tsx). The web search for the `updatechallenges` removal commit returned only unrelated results (e.g. [thegamer.com](https://www.thegamer.com/pokemon-unite-rotom-challenge-reduced/), [pokecommunity.com](https://www.pokecommunity.com/members/kraydl.414320/recent-content)) and was not used.
