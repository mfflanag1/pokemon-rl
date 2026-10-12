"""Frozen public vocabularies and fixed integer feature layout for R5."""
from .contract import CONTRACT, hash_json, parse_json

DIRECTORY = CONTRACT.parent / 'player-features-v1'
CATALOG = parse_json((DIRECTORY / 'catalog.json').read_bytes())
if hash_json({k: v for k, v in CATALOG.items() if k != 'content_hash'}) != CATALOG['content_hash']:
    raise ValueError('R5 catalog checksum')
if CATALOG['vocabulary_hash'] != hash_json(parse_json((CONTRACT / 'vocabulary.json').read_bytes())):
    raise ValueError('R5/R2 vocabulary mismatch')
LAYOUT = parse_json((DIRECTORY / 'layout.json').read_bytes())
LAYOUT_HASH = hash_json(LAYOUT)
INDEX = {k: {v: i for i, v in enumerate(values)} for k, values in CATALOG['domains'].items()}


def category(domain, name):
    """None = unknown (0); empty string = explicitly absent (1). No guessed aliases."""
    if name is None:
        return 0
    if name == '':
        return 1
    key = name.lower()
    if key not in CATALOG['names'][domain]:
        raise ValueError(f'Unsupported known {domain} name: {name!r}')
    return INDEX[domain][CATALOG['names'][domain][key]]


def category_id(domain, name):
    index = category(domain, name)
    return None if index == 0 else '' if index == 1 else CATALOG['domains'][domain][index]
