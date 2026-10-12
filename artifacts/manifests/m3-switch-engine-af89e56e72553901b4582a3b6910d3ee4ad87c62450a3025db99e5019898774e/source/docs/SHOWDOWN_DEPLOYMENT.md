# Public Showdown deployment status — 2026-10-11

The client has passed local transport tests. No public account login, direct
challenge, ladder search or staff communication has been performed. Account and
opponent names have not been supplied; passwords should remain local in the
`SHOWDOWN_PASSWORD` environment variable.

## Current official announcement

Smogon's [October 9 bot announcement](https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/)
calls for mandatory registration of bots, with registry details and availability
still to be announced. It prioritizes human choice about playing bots and protecting
human competitions and usage statistics. It also prohibits AI assistance during
tournaments. In the second post, chaos says enforcement is still being settled.
The announcement provides no explicit challenge-only exemption or currently usable
enrollment form. Its invitation to help development is not a confirmed registration
procedure. Before public play, establish the actual rollout and applicable account
requirements; do not infer authorization from technical connectivity.

Both posts were read directly in the in-app browser after the text-fetch service
returned 403. Their DOM timestamps are October 9, 2026 at 18:13:19 and 20:14:17
Eastern; the second was edited at 20:15:30. These facts describe the announcement
as inspected, not a claim that all later implementation details have been published.

The ordinary [Showdown rules](https://pokemonshowdown.com/rules) prohibit cheating,
system-gaming and staff impersonation; they do not describe a bot registry procedure.

## Local implementation versus production

The pinned server's [ladders.ts](https://github.com/smogon/pokemon-showdown/blob/942cda71d14bf0618bd57991708a2e2734b6daba/server/ladders.ts)
rejects designated bots from suspect queues and certain suspect/forced-prefix
matches. Its challenge and battle-preparation counters are configurable technical
limits, not a statement of public bot policy. Neither the production version nor
its configuration has been verified. No claim that ordinary ladder or challenge
play is unconditionally permitted follows from the inspected code.

Once the account setup and requirements are established, use an explicitly named,
consenting direct-challenge opponent for the first public test. Record the server
version/protocol differences, authentication, choice validity, terminal agreement
and reconnect behavior. Keep interactive games out of training/evaluation datasets
until admitted by the existing partition/provenance gates. No ladder-search command
is exposed in the current client.

The next independent local task is a stronger frozen opponent panel. The broader
custom simulator/JAX port remains a separate prerequisite for full-format learning.
No GPU or Prime resources are needed for these deployment checks.
