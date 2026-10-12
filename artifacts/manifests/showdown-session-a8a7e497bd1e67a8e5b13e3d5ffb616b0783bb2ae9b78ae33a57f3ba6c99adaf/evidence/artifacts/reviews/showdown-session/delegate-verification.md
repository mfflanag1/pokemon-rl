# Deployment research verification — 2026-10-11

Cross-model run: `/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261011-124735-0448e7a0`.
Fable performed read-only research; opposite-model Codex QA returned revise. No
delegated implementation was integrated. The unmodified research packet is rejected
against its precise policy/implementation distinction criteria, rather than silently
accepting the coordinator's corrected summary as the candidate's output.

Actual verification:

- Web fetch of the official Smogon announcement returned HTTP 403. The in-app
  browser then opened the exact primary thread successfully, showing posts
  11159257 (Django) and 11159319 (chaos). The coordinator directly read both.
  DOM time attributes give 2026-10-09T18:13:19-0400 and 20:14:17-0400; the latter
  was edited at 20:15:30. Thus QA's continuing primary-access objection was
  resolved independently; the dates no longer require inference.
- The announcement calls for a mandatory bot registry, says details are to be
  confirmed and availability will be advertised. Chaos describes enforcement as
  unsettled. A presently usable registration process was not demonstrated.
  The candidate's instruction to obtain registration immediately, and implication
  that DMing chaos is the registration channel, overstate the evidence. The
  invitation to help development is not a verified enrollment procedure.
- The ordinary rules page was fetched directly and forbids system-gaming and
  impersonation. It contains no bot-specific deployment procedure.
- Inspected pinned server/ladders.ts:99-106, 221-230, 398-406 and
  server/monitor.ts:232-249. These implement configurable counters and certain
  suspect/forced-prefix matchmaking exclusions. They are not themselves policy,
  proof of production configuration or universal eligibility. The concurrent
  limit checks existing count >5 before preparation, illustrating why a simple
  'at most five' policy assertion is not justified from the message alone.
- No exhaustive absence claim about authentication/commands is accepted. No
  direct-challenge exemption, live production version or registration rollout
  was established. No staff was contacted and no login/game/registration occurred.

The corrected, independently sourced deployment note is docs/SHOWDOWN_DEPLOYMENT.md.
Useful leads are credited to the delegate; its uncorrected binding-rules checklist
is not approved. This records an evidence-based rejection rather than model votes.
