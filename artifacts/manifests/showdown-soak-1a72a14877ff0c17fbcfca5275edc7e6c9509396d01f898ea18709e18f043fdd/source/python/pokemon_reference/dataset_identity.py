"""R7 pure whole-team identity and deterministic partition assignment.

Inputs are R2 ``team`` artifact payloads (``contracts/battle-v1/schema.json``,
``$defs/team``). Identity is the ordered whole team with ordered moves and
resolved forms; bookkeeping (``name``, ``shiny``, ``role``, ``speciesId``), the
generator seed and the raw ``source_team_hash`` never enter the identity.

The R2 schema fixes absent ``nature``/``happiness`` to the pinned generator
defaults, so a payload carrying any field outside the exact schema field sets
is a different mechanical contract and is rejected rather than dropped.

IDs are trusted only after outer corpus/artifact provenance validation; this
module performs no Dex interpretation and does not verify ``source_team_hash``.
All functions are pure: inputs are never mutated and outputs own their data.
"""

from copy import deepcopy

from jsonschema import ValidationError

from .contract import hash_json, validate_schema

# Exact R2 schema field sets, required independently of the JSON Schema check.
TEAM_FIELDS = frozenset(('generator_seed', 'source_team_hash', 'pokemon', 'ids'))
ID_FIELDS = frozenset(('species', 'base_species', 'ability', 'item', 'moves', 'tera_type'))
POKEMON_FIELDS = frozenset(('name', 'species', 'speciesId', 'gender', 'shiny', 'level', 'moves',
                            'ability', 'item', 'teraType', 'role', 'evs', 'ivs'))
STAT_FIELDS = frozenset(('hp', 'atk', 'def', 'spa', 'spd', 'spe'))
GENDERS = frozenset(('M', 'F', 'N', ''))
TEAM_SIZE = 6

# Supported split policy (contracts/dataset-v1/policy.json). Any other value is
# an unknown or inconsistent policy and is rejected explicitly.
POLICY_FIELDS = frozenset(('schema_version', 'identity_version', 'hash_version', 'salt', 'denominator',
                           'ranges', 'profile', 'showdown_commit', 'holdout_unit', 'ignored_fields',
                           'legacy_policy', 'final_access'))
RANGE_FIELDS = frozenset(('split', 'lower', 'upper'))
PINNED_POLICY = {
    'schema_version': 'r7-split-policy-1',
    'identity_version': 'r7-ordered-team-1',
    'hash_version': 'r7-team-partition-1',
    'profile': 'gen9randombattle',
    'showdown_commit': '942cda71d14bf0618bd57991708a2e2734b6daba',
    'holdout_unit': 'ordered-whole-team-with-ordered-moves-and-resolved-forms',
    'ignored_fields': ['generator_seed', 'source_team_hash', 'name', 'shiny', 'role', 'speciesId'],
}
DENOMINATOR = 10000
RANGES = (('train', 0, 9000), ('development', 9000, 9500), ('final', 9500, 10000))
DIGEST_BITS = 256
DIGEST_SPACE = 1 << DIGEST_BITS


def _is_int(value):
    return type(value) is int


def _require_fields(value, fields, what):
    if type(value) is not dict:
        raise ValueError(f'{what} must be an object')
    present = set(value)
    if present != fields:
        unknown = sorted(present - fields)
        missing = sorted(fields - present)
        if unknown:
            raise ValueError(f'{what} has unknown fields {unknown}; new mechanical fields are not accepted')
        raise ValueError(f'{what} is missing fields {missing}')


def _require_stats(value, what, maximum):
    _require_fields(value, STAT_FIELDS, what)
    for stat in sorted(STAT_FIELDS):
        number = value[stat]
        if not _is_int(number) or not 0 <= number <= maximum:
            raise ValueError(f'{what}.{stat} must be an integer in [0, {maximum}]')


def _require_moves(value, what):
    if type(value) is not list or not 1 <= len(value) <= 4:
        raise ValueError(f'{what} must be a list of 1 to 4 move IDs')
    for move in value:
        if type(move) is not str or not move:
            raise ValueError(f'{what} entries must be non-empty strings')


def _validate_payload(payload):
    _require_fields(payload, TEAM_FIELDS, 'team payload')
    try:
        validate_schema(payload, 'team')
    except (ValidationError, TypeError, ValueError) as exc:
        raise ValueError(f'team payload does not match the R2 team schema: {exc}') from exc
    ids, pokemon = payload['ids'], payload['pokemon']
    if type(ids) is not list or len(ids) != TEAM_SIZE:
        raise ValueError(f'team payload ids must list exactly {TEAM_SIZE} entries')
    if type(pokemon) is not list or len(pokemon) != TEAM_SIZE:
        raise ValueError(f'team payload pokemon must list exactly {TEAM_SIZE} entries')
    for slot, entry in enumerate(ids):
        what = f'ids[{slot}]'
        _require_fields(entry, ID_FIELDS, what)
        for field in ('species', 'base_species', 'ability', 'tera_type'):
            if type(entry[field]) is not str or not entry[field]:
                raise ValueError(f'{what}.{field} must be a non-empty string')
        if type(entry['item']) is not str:
            raise ValueError(f'{what}.item must be a string')
        _require_moves(entry['moves'], f'{what}.moves')
    for slot, mon in enumerate(pokemon):
        what = f'pokemon[{slot}]'
        _require_fields(mon, POKEMON_FIELDS, what)
        if not _is_int(mon['level']) or not 1 <= mon['level'] <= 100:
            raise ValueError(f'{what}.level must be an integer in [1, 100]')
        if type(mon['gender']) is not str or mon['gender'] not in GENDERS:
            raise ValueError(f'{what}.gender must be one of M, F, N or empty')
        _require_stats(mon['evs'], f'{what}.evs', 252)
        _require_stats(mon['ivs'], f'{what}.ivs', 31)
        _require_moves(mon['moves'], f'{what}.moves')


def canonical_team(payload):
    """Return the ordered mechanical identity of a validated R2 team payload.

    Each slot pairs ``ids[i]`` (resolved species, ability, item, ordered moves,
    tera type) with ``pokemon[i]`` (level, gender, EVs, IVs) by position. The
    result is a fresh structure that shares no containers with ``payload``.
    """
    _validate_payload(payload)
    team = []
    for entry, mon in zip(payload['ids'], payload['pokemon']):
        team.append({
            'species': entry['species'],
            'ability': entry['ability'],
            'item': entry['item'],
            'moves': list(entry['moves']),
            'tera_type': entry['tera_type'],
            'level': mon['level'],
            'gender': mon['gender'],
            'evs': {stat: mon['evs'][stat] for stat in sorted(STAT_FIELDS)},
            'ivs': {stat: mon['ivs'][stat] for stat in sorted(STAT_FIELDS)},
        })
    return deepcopy(team)


def _validate_policy(policy):
    _require_fields(policy, POLICY_FIELDS, 'split policy')
    for field, pinned in PINNED_POLICY.items():
        if type(policy[field]) is not type(pinned) or policy[field] != pinned:
            raise ValueError(f'unsupported split policy {field}: {policy[field]!r}')
    if type(policy['salt']) is not str or not policy['salt']:
        raise ValueError('split policy salt must be a non-empty string')
    for field in ('legacy_policy', 'final_access'):
        if type(policy[field]) is not str:
            raise ValueError(f'split policy {field} must be a string')
    if not _is_int(policy['denominator']) or policy['denominator'] != DENOMINATOR:
        raise ValueError(f'split policy denominator must be exactly {DENOMINATOR}')
    ranges = policy['ranges']
    if type(ranges) is not list or len(ranges) != len(RANGES):
        raise ValueError(f'split policy must declare exactly {len(RANGES)} ranges')
    for index, (declared, expected) in enumerate(zip(ranges, RANGES)):
        _require_fields(declared, RANGE_FIELDS, f'ranges[{index}]')
        split, lower, upper = expected
        if (type(declared['split']) is not str or declared['split'] != split
                or not _is_int(declared['lower']) or declared['lower'] != lower
                or not _is_int(declared['upper']) or declared['upper'] != upper):
            raise ValueError(f'split policy ranges[{index}] must be exactly {split} [{lower}, {upper})')


def team_key(payload, policy):
    """Deterministic identity key of the team under ``policy.identity_version``."""
    _validate_policy(policy)
    return hash_json([policy['identity_version'], canonical_team(payload)])


def partition(payload, policy):
    """Assign the team to a split using the full unsigned 256-bit salted digest.

    Thresholds are compared as exact integers: ``digest * denominator`` is
    placed against ``bound * 2**256``. No floats and no modulo reduction.
    """
    _validate_policy(policy)
    team = canonical_team(payload)
    hexdigest = hash_json([policy['hash_version'], policy['salt'], team])
    if len(hexdigest) * 4 != DIGEST_BITS:
        raise ValueError('unexpected digest width')
    digest = int(hexdigest, 16)
    if not 0 <= digest < DIGEST_SPACE:
        raise ValueError('digest outside the unsigned 256-bit range')
    scaled = digest * policy['denominator']
    for declared in policy['ranges']:
        if declared['lower'] * DIGEST_SPACE <= scaled < declared['upper'] * DIGEST_SPACE:
            return declared['split']
    raise ValueError('split policy ranges do not cover the digest space')


def pair_partition(left, right, policy):
    """Partition of a pair of teams; rejects pairs that straddle partitions."""
    left_split = partition(left, policy)
    right_split = partition(right, policy)
    if left_split != right_split:
        raise ValueError(f'cross-partition pair: {left_split} vs {right_split}')
    return left_split
