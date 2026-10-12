Verdict: **revise**

The report is technically strong and covers every requested topic, but it does not meet the evidence standard for a review of the *pinned* source.

### Concrete findings

1. **Major — central conclusions rely on unpinned `master`.**  
   The candidate admits that authentication internals, challenge acceptance, `noCancel`, forced switching, Revival Blessing, trapping mechanics, startup, Node requirements, and throttling were taken from current upstream rather than the supplied snapshot ([prompt.txt](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261011-120242-989c20dc/qa-bb6c691f04/prompt.txt:10)). These are not merely peripheral details; they comprise most acceptance items. URLs to mutable `master` files do not establish what the pinned server does.

2. **Major — the challenge-protocol recommendation is not established for the pin.**  
   The supplied `PROTOCOL.md` explicitly documents `|updatechallenges|`. The candidate instead recommends primarily recognizing PM-embedded `/challenge` messages based on current upstream server/client files ([prompt.txt](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261011-120242-989c20dc/qa-bb6c691f04/prompt.txt:39)). Handling both is a sensible compatibility recommendation, but the claim that the documented message “never” arrives cannot be presented as a pinned-source fact without the pinned `ladders-challenges.ts`.

3. **Major — evidence is not precise enough for many primary claims.**  
   References such as “upstream users.ts,” “battle.ts `getRequests`,” and “side.ts `emitChoiceError`” provide neither commit IDs nor line links. The source list contains mutable raw URLs but does not map individual claims to exact locations. This falls short of the criterion requiring precise primary file/line or official-URL evidence.

4. **Moderate — snapshot provenance is asserted without usable evidence.**  
   The candidate says the workspace had six supplied files and a one-commit git history, but the review artifact contains the files only as embedded prompt text and no repository metadata. The “pin is unrecorded” limitation is appropriately disclosed, but it should lead to an `insufficient evidence` qualification for pin-specific behavior, not confidence that every acceptance item has evidence.

5. **Moderate — several recommendations are insufficiently supported.**  
   “Never batch more than 5 lines in one frame” is not derived from the cited socket framing code and appears to conflate message framing with user-command throttling ([prompt.txt](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261011-120242-989c20dc/qa-bb6c691f04/prompt.txt:20)). The report should either cite the exact multiline-throttle implementation or label the number as a conservative operational policy.

6. **Moderate — local deployment guidance is incomplete from a security perspective.**  
   Disabling the default backdoor is correctly recommended, but the proposed local command retains the default `0.0.0.0` bind while enabling `--no-security`. A minimal safe recipe should bind explicitly to loopback or require network isolation. `noguestsecurity`, `nothrottle`, and `noipchecks` materially weaken any externally reachable instance.

7. **Moderate — the tests are scenarios, not fully reproducible end-to-end tests.**  
   “Force a faint,” an ability “seed,” and a “Revival Blessing seed” do not specify fixtures, deterministic teams/input logs, expected message ordering, or how the target state is reached. The trap test also assumes the refreshed request has exactly `rqid + 1`; that is plausible for a single side update but should be asserted as “different/newer rqid,” since the server counter is battle-global.

8. **Minor — reconnect coverage omits required setup.**  
   A reconnect test must establish whether the client re-authenticates and explicitly rejoins the room or relies on server rejoin behavior. It should also assert handling of `|request|null`, duplicate replayed requests, and `|sentchoice|` without resending the choice.

### What is well supported

The supplied files do substantiate several important points:

- Native WebSocket path filtering and rejection of malformed/oversized client frames.
- Client/server room framing.
- `rqid` injection, stale-ID rejection, wait handling, and reconnect replay in `room-battle.ts`.
- Timer constants, challenge timer differences, simulator-crash handling, and early-boot lockdown.
- Default bind address, guest-security setting, backdoor setting, and login-server configuration.
- Documented forced-switch choice syntax and `[Invalid choice]` versus `[Unavailable choice]` behavior.

The report also does a good job distinguishing many recommendations from facts and explicitly acknowledging unresolved questions.

### Required revision

Recast the report into three evidence classes:

- **Pinned-source facts:** only claims demonstrated by the six supplied files, with exact line citations.
- **Current-upstream observations:** commit-pinned URLs and exact lines, explicitly not guaranteed to match the snapshot.
- **Recommendations/inferences:** operational advice clearly separated from protocol guarantees.

For complete acceptance, obtain the pinned `users.ts`, `ladders*.ts`, `sim/{battle,side,pokemon}.ts`, launcher, config loader, package manifest, and commit identifier. Otherwise the honest overall conclusion should be: **insufficient evidence for pin-specific authentication, challenge signaling, `noCancel`, Revival Blessing, and hidden-trapping internals**.