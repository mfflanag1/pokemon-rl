"""Episode boundary: actor inputs and training targets use different immutable types."""
from dataclasses import dataclass
from .contract import canonical, parse_json
from .episodes import validate_episode
from .player_history import ActorInput, PlayerHistory
from .actor_actions import ActionCodec


@dataclass(frozen=True)
class TrainingTarget:
    action_index: int | None
    control_action: str | None
    accepted: bool
    next_actor_index: int | None
    next_bootstrap: bool
    terminal: bool
    truncated: bool
    reward_numerator: int
    reward_denominator: int


@dataclass(frozen=True)
class PlayerTrajectory:
    viewer: str
    actor_inputs: tuple[ActorInput, ...]
    targets: tuple[TrainingTarget, ...]
    bootstrap: ActorInput | None


def project_player_views(payload, viewer):
    """Project selected views only. This does not certify a privileged episode.

    Used by the validated offline boundary below and by noninterference tests.
    Never read spec, frames, opponent views, actions, targets or outcome to encode.
    """
    history = PlayerHistory(viewer)
    inputs = []
    for decision in payload['decisions']:
        if decision['side'] == viewer:
            inputs.append(history.append(decision['view']))
    bootstrap = None
    if payload['bootstrap_views'] is not None:
        bootstrap = history.append(payload['bootstrap_views'][viewer])
    return tuple(inputs), bootstrap


def assemble_episode(bundle, manifest):
    """Validate the complete privileged artifact, then separate inputs and targets."""
    validate_episode(bundle, manifest)
    # Own the input bytes so caller mutations cannot race either projection or labels.
    payload = parse_json(canonical(bundle['episode']['payload']))
    result = []
    for side in ('p1', 'p2'):
        actors, bootstrap = project_player_views(payload, side)
        decisions = [d for d in payload['decisions'] if d['side'] == side]
        local = {d['sequence']: i for i, d in enumerate(decisions)}
        targets = []
        for d in decisions:
            transition = payload['transitions'][d['sequence']]
            codec = ActionCodec(d['view']['request'])
            action = d['action']
            index = codec.encode(action) if action['kind'] != 'pass' else None
            target = transition['target']
            targets.append(TrainingTarget(index, 'pass' if index is None else None, d['accepted'],
                local[target['sequence']] if target['kind'] == 'decision' else None,
                target['kind'] == 'bootstrap', transition['terminal'], transition['truncated'],
                transition['reward_numerator'], transition['reward_denominator']))
        if len(targets) != len(actors):
            raise ValueError('Actor/target count mismatch')
        result.append(PlayerTrajectory(side, actors, tuple(targets), bootstrap))
    return tuple(result)
