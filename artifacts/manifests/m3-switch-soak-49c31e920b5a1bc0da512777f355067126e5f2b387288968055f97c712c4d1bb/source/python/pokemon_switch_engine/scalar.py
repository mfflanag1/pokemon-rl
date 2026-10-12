"""Independent integer state machine for the sealed two-member profile."""
from copy import deepcopy
from itertools import permutations, product
from .damage import damage, stats, max_pp

SPECIES = ('bulbasaur', 'squirtle', 'fennekin')
ROSTER = [(list(p), list(v)) for p in permutations(range(3), 2) for v in product(range(2), repeat=2)]
ROSTER_IDS = ['_'.join(SPECIES[s] + '-' + ('swift' if v else 'plain') for s, v in zip(p, vs)) for p, vs in ROSTER]
CAPACITY = 32


def require(ok, message):
    if not ok: raise ValueError(message)


def initial(teams):
    require(isinstance(teams, (list, tuple)) and len(teams) == 2, 'Two team IDs')
    require(all(type(t) is int and 0 <= t < 24 for t in teams), 'Team ID')
    species = [ROSTER[t][0][:] for t in teams]; swift = [ROSTER[t][1][:] for t in teams]
    return {'species': species, 'swift': swift, 'hp': [[stats(SPECIES[s])[0] for s in p] for p in species],
            'pp': [[max_pp(SPECIES[s]) for s in p] for p in species], 'active': [0, 0],
            'phase': 0, 'turn': 1, 'group': 0, 'winner': -1, 'seen': [[1, 0], [1, 0]], 'history': []}


def masks(state):
    result = []
    for side in range(2):
        a = state['active'][side]
        need = state['phase'] != 2 and (state['phase'] == 0 or state['hp'][side][a] == 0)
        result.append([need and state['phase'] == 0 and state['pp'][side][a][m] > 0 for m in range(2)] +
                      [need and m != a and state['hp'][side][m] > 0 for m in range(2)])
    return result


def validate_chance(chance):
    require(isinstance(chance, dict) and set(chance) == {'crit', 'roll', 'tie'}, 'Chance fields')
    require(type(chance['tie']) is int and chance['tie'] in (0, 1), 'Tie seat')
    require(isinstance(chance['crit'], list) and len(chance['crit']) == 2 and all(type(x) is bool for x in chance['crit']), 'Critical outcomes')
    require(isinstance(chance['roll'], list) and len(chance['roll']) == 2 and all(type(x) is int and 0 <= x < 16 for x in chance['roll']), 'Damage rolls')


def step(state, actions, chance):
    require(state['winner'] == -1 and state['turn'] <= 12, 'Terminal or cutoff submission')
    validate_chance(chance); legal = masks(state)
    require(isinstance(actions, list) and len(actions) == 2, 'Action pair')
    for side, action in enumerate(actions):
        require(type(action) is int and (0 <= action < 4 and legal[side][action] if any(legal[side]) else action == -1), 'Illegal action or wait')
    s = deepcopy(state); start = len(s['history'])
    speed = [stats(SPECIES[s['species'][p][s['active'][p]]], bool(s['swift'][p][s['active'][p]]))[5] for p in range(2)]
    if s['phase'] == 1: first = next(p for p in range(2) if any(legal[p]))
    elif (actions[0] >= 2) != (actions[1] >= 2): first = int(actions[1] >= 2)
    elif speed[0] != speed[1]: first = int(speed[1] > speed[0])
    else: first = chance['tie']
    for side in (first, 1-first):
        action = actions[side]; member = s['active'][side]; other = 1-side
        if action < 0 or s['winner'] != -1: continue
        if action >= 2:
            member = action-2; s['active'][side] = member; s['seen'][side][member] = 1
            event = [s['turn'], side, 0, member, s['species'][side][member], -1, member, s['hp'][side][member], 0]
        elif s['hp'][side][member] > 0:
            target = s['active'][other]; source_species = s['species'][side][member]
            amount = damage(SPECIES[source_species], SPECIES[s['species'][other][target]], s['hp'][side][member], action, chance['crit'][side], chance['roll'][side])
            s['hp'][other][target] = max(0, s['hp'][other][target]-amount)
            s['pp'][side][member][action] -= 1
            event = [s['turn'], side, 1, member, source_species, action, target, s['hp'][other][target], int(chance['crit'][side])]
            if not any(s['hp'][other]): s['winner'] = side
        else: continue
        s['history'].append(event)
    require(len(s['history']) <= CAPACITY, 'History capacity')
    s['phase'] = 2 if s['winner'] != -1 else int(any(s['hp'][p][s['active'][p]] == 0 for p in range(2)))
    if s['phase'] == 0: s['turn'] += 1
    s['group'] += 1
    return s, s['history'][start:]


def percent(hp, maximum):
    n = (100*hp + maximum-1)//maximum  # Pinned Showdown uses ceil(100*hp/maxhp).
    return 99 if n == 100 and hp < maximum else n


def view(state, side):
    require(type(side) is int and side in (0, 1), 'Viewer')
    other = 1-side; own = []; opponent = []; history = deepcopy(state['history'])
    for m in range(2):
        s = state['species'][side][m]; st = stats(SPECIES[s], bool(state['swift'][side][m]))
        own.append([s+1, state['hp'][side][m], *st, int(state['active'][side] == m)])
        s = state['species'][other][m]; seen = state['seen'][other][m]
        opponent.append([s+1 if seen else 0, percent(state['hp'][other][m], stats(SPECIES[s])[0]) if seen else -1,
                         int(state['active'][other] == m), int(state['hp'][other][m] == 0) if seen else 0])
    for h in history:
        actor = h[1]; affected = actor if h[2] == 0 else 1-actor
        if affected != side: h[7] = percent(h[7], stats(SPECIES[state['species'][affected][h[6]]])[0])
        h[1] = int(actor != side)
    mask = masks(state)[side]
    phase = 3 if state['phase'] == 2 else 0 if state['phase'] == 0 else 1 if any(mask) else 2
    return {'own': own, 'pp': deepcopy(state['pp'][side]), 'opponent': opponent,
            'context': [state['turn'], phase, state['active'][side], state['active'][other], len(history)],
            'history': history + [[-1]*9 for _ in range(CAPACITY-len(history))], 'mask': mask}
