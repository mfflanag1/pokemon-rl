"""Small test-only JSONL bridge; every operation uses the independent Python API."""
import sys
from pokemon_reference.contract import (
    parse_json, canonical, hash_json, validate_artifact, validate_schema,
    action_mask, admissible_actions, action_from_index, action_index,
)

for line in sys.stdin.buffer:
    try:
        request = parse_json(line)
        op, value = request['op'], request['value']
        if op == 'artifact':
            result = hash_json(validate_artifact(value))
        elif op == 'fragment':
            validate_schema(value, request['fragment'])
            result = True
        elif op == 'request':
            result = {'actions': admissible_actions(value), 'mask': action_mask(value)}
        elif op == 'index':
            result = {'action': action_from_index(value), 'index': action_index(action_from_index(value))}
        elif op == 'wire':
            result = {'canonical': canonical(parse_json(value)), 'hash': hash_json(parse_json(value))}
        else:
            raise ValueError('Unknown bridge operation')
        print(canonical({'ok': True, 'result': result}))
    except Exception as exc:
        print(canonical({'ok': False, 'error': type(exc).__name__}))
