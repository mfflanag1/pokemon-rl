"""Independent integer continuation simulator for the sealed starter profile."""

from copy import deepcopy
from itertools import permutations, product

from pokemon_switch_engine.damage import damage, max_pp, stats


SPECIES = ('bulbasaur', 'squirtle', 'fennekin')
ROSTER = [
    (list(species), list(swift))
    for species in permutations(range(3), 2)
    for swift in product(range(2), repeat=2)
]
ROSTER_IDS = [
    '_'.join(
        SPECIES[species] + '-' + ('swift' if variant else 'plain')
        for species, variant in zip(pair, variants)
    )
    for pair, variants in ROSTER
]

CAPACITY = 128
MAX_TURNS = 60
MAX_GROUPS = 62
MAX_EVENTS = 122


def require(ok, message):
    if not ok:
        raise ValueError(message)


def initial(teams):
    require(isinstance(teams, (list, tuple)) and len(teams) == 2, 'Two team IDs')
    require(all(type(team) is int and 0 <= team < 24 for team in teams), 'Team ID')
    species = [ROSTER[team][0][:] for team in teams]
    swift = [ROSTER[team][1][:] for team in teams]
    return {
        'species': species,
        'swift': swift,
        'hp': [[stats(SPECIES[member])[0] for member in pair] for pair in species],
        'pp': [[max_pp(SPECIES[member]) for member in pair] for pair in species],
        'active': [0, 0],
        'phase': 0,
        'turn': 1,
        'group': 0,
        'winner': -1,
        'seen': [[1, 0], [1, 0]],
        'history': [],
    }


def masks(state):
    result = []
    for side in range(2):
        active = state['active'][side]
        need = state['phase'] != 2 and (
            state['phase'] == 0 or state['hp'][side][active] == 0
        )
        result.append(
            [
                need and state['phase'] == 0 and state['pp'][side][active][move] > 0
                for move in range(2)
            ]
            + [
                need and member != active and state['hp'][side][member] > 0
                for member in range(2)
            ]
        )
    return result


def validate_chance(chance):
    require(
        isinstance(chance, dict) and set(chance) == {'crit', 'roll', 'tie'},
        'Chance fields',
    )
    require(type(chance['tie']) is int and chance['tie'] in (0, 1), 'Tie seat')
    require(
        isinstance(chance['crit'], list)
        and len(chance['crit']) == 2
        and all(type(outcome) is bool for outcome in chance['crit']),
        'Critical outcomes',
    )
    require(
        isinstance(chance['roll'], list)
        and len(chance['roll']) == 2
        and all(type(roll) is int and 0 <= roll < 16 for roll in chance['roll']),
        'Damage rolls',
    )


def step(state, actions, chance):
    require(
        state['winner'] == -1 and state['turn'] <= MAX_TURNS,
        'Terminal or cutoff submission',
    )
    require(
        isinstance(state['history'], list) and len(state['history']) <= CAPACITY,
        'History capacity',
    )
    validate_chance(chance)
    legal = masks(state)
    require(isinstance(actions, list) and len(actions) == 2, 'Action pair')
    for side, action in enumerate(actions):
        require(
            type(action) is int
            and (
                0 <= action < 4 and legal[side][action]
                if any(legal[side])
                else action == -1
            ),
            'Illegal action or wait',
        )

    result = deepcopy(state)
    start = len(result['history'])
    speed = [
        stats(
            SPECIES[result['species'][side][result['active'][side]]],
            bool(result['swift'][side][result['active'][side]]),
        )[5]
        for side in range(2)
    ]
    if result['phase'] == 1:
        first = next(side for side in range(2) if any(legal[side]))
    elif (actions[0] >= 2) != (actions[1] >= 2):
        first = int(actions[1] >= 2)
    elif speed[0] != speed[1]:
        first = int(speed[1] > speed[0])
    else:
        first = chance['tie']

    for side in (first, 1 - first):
        action = actions[side]
        member = result['active'][side]
        other = 1 - side
        if action < 0 or result['winner'] != -1:
            continue
        if action >= 2:
            member = action - 2
            result['active'][side] = member
            result['seen'][side][member] = 1
            event = [
                result['turn'], side, 0, member, result['species'][side][member],
                -1, member, result['hp'][side][member], 0,
            ]
        elif result['hp'][side][member] > 0:
            target = result['active'][other]
            source_species = result['species'][side][member]
            amount = damage(
                SPECIES[source_species],
                SPECIES[result['species'][other][target]],
                result['hp'][side][member],
                action,
                chance['crit'][side],
                chance['roll'][side],
            )
            result['hp'][other][target] = max(
                0, result['hp'][other][target] - amount
            )
            result['pp'][side][member][action] -= 1
            event = [
                result['turn'], side, 1, member, source_species, action, target,
                result['hp'][other][target], int(chance['crit'][side]),
            ]
            if not any(result['hp'][other]):
                result['winner'] = side
        else:
            continue
        result['history'].append(event)

    require(len(result['history']) <= CAPACITY, 'History capacity')
    result['phase'] = (
        2
        if result['winner'] != -1
        else int(
            any(
                result['hp'][side][result['active'][side]] == 0
                for side in range(2)
            )
        )
    )
    if result['phase'] == 0:
        result['turn'] += 1
    result['group'] += 1
    return result, result['history'][start:]


def percent(hp, maximum):
    value = (100 * hp + maximum - 1) // maximum
    return 99 if value == 100 and hp < maximum else value


def view(state, side):
    require(type(side) is int and side in (0, 1), 'Viewer')
    require(
        isinstance(state['history'], list) and len(state['history']) <= CAPACITY,
        'History capacity',
    )
    other = 1 - side
    own = []
    opponent = []
    history = deepcopy(state['history'])
    for member in range(2):
        species = state['species'][side][member]
        member_stats = stats(SPECIES[species], bool(state['swift'][side][member]))
        own.append([
            species + 1,
            state['hp'][side][member],
            *member_stats,
            int(state['active'][side] == member),
        ])

        species = state['species'][other][member]
        seen = state['seen'][other][member]
        opponent.append([
            species + 1 if seen else 0,
            percent(state['hp'][other][member], stats(SPECIES[species])[0])
            if seen else -1,
            int(state['active'][other] == member),
            int(state['hp'][other][member] == 0) if seen else 0,
        ])

    for event in history:
        actor = event[1]
        affected = actor if event[2] == 0 else 1 - actor
        if affected != side:
            event[7] = percent(
                event[7],
                stats(SPECIES[state['species'][affected][event[6]]])[0],
            )
        event[1] = int(actor != side)

    mask = masks(state)[side]
    phase = (
        3
        if state['phase'] == 2
        else 0
        if state['phase'] == 0
        else 1
        if any(mask)
        else 2
    )
    return {
        'own': own,
        'pp': deepcopy(state['pp'][side]),
        'opponent': opponent,
        'context': [
            state['turn'], phase, state['active'][side], state['active'][other],
            len(history),
        ],
        'history': history + [[-1] * 9 for _ in range(CAPACITY - len(history))],
        'mask': mask,
    }
