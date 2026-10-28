"""Diagnostic worker-watch transport.

Additive package: it instruments the bounded request/response wire used by the
simulator workers so that stalls can be attributed to a phase (write vs read),
to the owned process group, and to a bounded cleanup result. It changes no
policy, no RNG/seed derivation and no training behaviour; it only observes and
records. The JSON wire format and the response validation are identical to
``pokemon_opponent_mix.transport``; only the default worker command differs
(``node packages/worker-watch/worker.mjs``, owned by the coordinator).
"""
from .transport import (
    CLEANUP_WAIT,
    DEFAULT_COMMAND,
    FRAME_OPS,
    FRAME_SCHEMA,
    HELLO_SCHEMA,
    PLAYER_SCHEMA,
    RESPONSE_LIMIT,
    ROOT,
    Worker,
    WorkerCleanupError,
    WorkerCleanupGroupError,
    close_workers,
)

__all__ = [
    'CLEANUP_WAIT',
    'DEFAULT_COMMAND',
    'FRAME_OPS',
    'FRAME_SCHEMA',
    'HELLO_SCHEMA',
    'PLAYER_SCHEMA',
    'RESPONSE_LIMIT',
    'ROOT',
    'Worker',
    'WorkerCleanupError',
    'WorkerCleanupGroupError',
    'close_workers',
]
