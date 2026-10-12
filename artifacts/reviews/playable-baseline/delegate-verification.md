# Protocol exploration disposition

The Fable research packet completed; opposite-model Codex QA requested revision.
The coordinator classifies the overall packet as **rejected against its precise
pinned-evidence acceptance criteria**, while retaining independently verified findings.
The input packet omitted showdown.pin.json and several implementation files. This
was an underspecified coordinator packet; mutable master fetches do not establish
pin-specific facts. No delegated implementation was integrated.

Verified directly against pinned commit 942cda71d14bf0618bd57991708a2e2734b6daba:
- server/ladders-challenges.ts:220-237 emits PM-embedded /challenge messages.
  Both actual local-server smoke games completed through this path.
- server/room-battle.ts:624-640 checks rqid; :796-818 refreshes unavailable requests.
- sim/side.ts:527-534 sends Unavailable choice followed by an updated request.
  The new oracle test exercises real Arena Trap rejection and successful retry.
- server/users.ts:641 allows local no-token login only with noguestsecurity.
  The smoke authenticates two clients this way, twice on separate transports.
- server/users.ts:33-36 defines 600 ms default throttle, 25 ms public-bot throttle.
  The baseline uses 750 ms default decision delay; no public throttle claim is tested.
- server/sockets.ts:66-89 can override bindaddress through cloud-env unless an
  explicit port is provided. The initial sandboxed smoke failed to listen; the
  corrected launcher passes an explicit port and the successful server log reports
  127.0.0.1. It also disables console/backdoor access and public login-server traffic.
- The packet suggestion that gitignored config/build changes leave upstream
  untouched is not used: our immutable build receipt also hashes those files.
  The launcher instead uses a separate runtime copy and verifies the original build.

The QA's generic multiline-throttle and rqid+1 recommendations were not adopted.
Request IDs are treated as increasing battle-global integers, not necessarily +1.
Public server policy, official login and browser UI were not verified by this packet.
