"""Conservative memory of visible encounters, never inferred opponent roster slots."""
from copy import deepcopy
import re
from .feature_spec import LAYOUT, category

_METADATA = {'', 't:', 'player', 'gametype', 'gen', 'tier', 'rule', 'teamsize', 'start', 'upkeep',
             'turn', 'message', 'hint', '-hint', '-message', 'debug'}
_STRUCTURED = {'switch', 'drag', 'replace', 'detailschange', '-formechange', '-transform',
               '-damage', '-heal', '-sethp', '-status', '-curestatus', 'faint', '-item', '-enditem',
               '-ability', '-endability', '-activate', '-terastallize', 'move', '-weather', '-fieldstart', '-fieldend',
               '-sidestart', '-sideend'}
_STATUS = {name: i for i, name in enumerate(LAYOUT['statuses'])}
_ABSORB_HEALS = {'Water Absorb', 'Volt Absorb', 'Dry Skin', 'Earth Eater'}
_SUBJECT_EVENTS = _STRUCTURED - {'switch', 'drag', 'replace', '-weather', '-fieldstart', '-fieldend', '-sidestart', '-sideend'}


def details(text):
    parts = text.split(', ')
    species = category('species', parts[0])
    level = 100; gender = 3
    for value in parts[1:]:
        if re.fullmatch(r'L[0-9]+', value):
            level = int(value[1:])
        elif value in ('M', 'F'):
            gender = 1 if value == 'M' else 2
        elif value == 'shiny' or value.startswith('tera:'):
            continue
        else:
            raise ValueError('Unsupported visible details: ' + text)
    return species, level, gender


def health(text, *, percent=False, previous_max=0):
    if text == '0 fnt':
        return (0, 100 if percent else previous_max, 1, int(percent or previous_max > 0), 1, _STATUS['fnt'])
    match = re.fullmatch(r'([0-9]+)/([0-9]+)(?: ([a-z]+))?', text)
    if not match:
        raise ValueError('Unsupported visible HP: ' + text)
    numerator, denominator = int(match[1]), int(match[2])
    if not 0 <= numerator <= denominator or denominator <= 0 or (percent and denominator != 100):
        raise ValueError('HP precision differs from pinned percent profile')
    status = match[3] or '<none>'
    if status not in _STATUS:
        raise ValueError('Unknown visible status: ' + status)
    return numerator, denominator, 1, 1, 0, _STATUS[status]


class PublicHistory:
    def __init__(self, viewer):
        self.viewer = viewer
        self.encounters = {'p1': [], 'p2': []}
        self.events = []
        self.facts = {}
        self.unstructured = 0
        self.turn = 0

    def current(self, side):
        return self.encounters[side][-1] if self.encounters[side] else None

    def subject(self, ident):
        # Bench mentions cannot be safely joined to an encounter by nickname.
        if not re.match(r'^p[12]a: ', ident):
            return None
        row = self.current(ident[:2])
        return row if row is not None and row['ident'] == ident else None

    def ingest(self, line):
        parts = line.split('|')
        if len(parts) < 2 or parts[0]:
            raise ValueError('Malformed visible protocol line')
        kind = parts[1]
        if kind == 'request':
            return
        if kind == 'turn':
            self.turn = int(parts[2]); return
        if kind in _METADATA:
            return
        if kind in ('win', 'tie'):
            raise ValueError('Terminal/future outcome is not an actor decision view')
        if kind == 'split':
            raise ValueError('Unprojected split channel is not player history')
        self.events.append(line)
        if kind not in _STRUCTURED:
            self.unstructured += 1
        row = self.subject(parts[2]) if len(parts) > 2 else None
        if kind in _SUBJECT_EVENTS and row is None:
            self.unstructured += 1  # e.g. a bench Heal Bell target that cannot be joined safely
        if kind in ('switch', 'drag'):
            ident = parts[2]
            if not re.match(r'^p[12]a: ', ident):
                raise ValueError('Only pinned singles active positions are supported')
            side = ident[:2]
            old = self.current(side)
            if old is not None:
                old['active'] = 0
            row = {key: 0 for key in LAYOUT['appearance_columns']}
            row.update(present=1, encounter=len(self.encounters[side]) + 1, active=1, ident=ident,
                       percent=(side != self.viewer))
            row['apparent_species'], row['apparent_level'], row['gender'] = details(parts[3])
            row['entry_species'], row['entry_level'] = row['apparent_species'], row['apparent_level']
            for value in parts[3].split(', '):
                if value.startswith('tera:'): row['revealed_tera'] = category('types', value[5:])
            self.encounters[side].append(row)
            self._hp(row, parts[4])
        elif kind == 'replace':
            # A reveal can change the nickname: match the occupied public position only.
            if not re.match(r'^p[12]a: ', parts[2]):
                raise ValueError('Invalid replacement position')
            row = self.current(parts[2][:2])
            if row is None:
                raise ValueError('Illusion reveal without a visible encounter')
            row['ident'] = parts[2]
            row['apparent_species'], row['apparent_level'], row['gender'] = details(parts[3])
            row['illusion_broken'] = 1
            if len(parts) > 4 and re.match(r'^(?:[0-9]+/|0 fnt)', parts[4]):
                self._hp(row, parts[4])
        elif row is not None:
            if kind in ('detailschange', '-formechange'):
                if kind == 'detailschange':
                    row['apparent_species'], row['apparent_level'], row['gender'] = details(parts[3])
                else:
                    row['apparent_species'] = category('species', parts[3])
                row['appearance_changed'] = 1
            elif kind == '-transform':
                target = self.subject(parts[3])
                row['apparent_species'] = target['apparent_species'] if target else 0
                row['transformed'] = 1
                row['appearance_changed'] = 1
                # No target-private set or stats are copied, including our own private request.
            elif kind in ('-damage', '-heal'):
                self._hp(row, parts[3])
            elif kind in ('-status', '-curestatus'):
                status = parts[3] if kind == '-status' else '<none>'
                if status not in _STATUS:
                    raise ValueError('Unknown status')
                row['status'] = _STATUS[status]
            elif kind == 'faint':
                self._hp(row, '0 fnt')
            elif kind in ('-item', '-enditem'):
                self._reveal(row, 'item', parts[3]); row['item_removed'] = int(kind == '-enditem')
            elif kind == '-ability':
                self._reveal(row, 'ability', parts[3])
            elif kind == '-endability':
                row['ability_suppression_seen'] = 1
            elif kind == '-activate' and parts[3].startswith('ability: '):
                self._reveal(row, 'ability', parts[3][9:])
            elif kind == 'move':
                row['last_used_move'] = category('moves', parts[3])
                row['last_used_move_known'] = 1
            elif kind == '-terastallize':
                row['revealed_tera'] = category('types', parts[3])
        if kind == '-sethp':
            for i in range(2, len(parts) - 1, 2):
                target = self.subject(parts[i])
                if target is None:
                    break
                self._hp(target, parts[i + 1])
        # Source attribution is scoped. Trace/Skill Swap [of] tags have other meanings.
        # Damage's source is the effect owner. Heal's source can be the attacker!
        # Only audited absorb callbacks justify attributing a heal with [of].
        # Other message kinds retain their tags as raw events, without attribution.
        if kind in ('-damage', '-heal'):
            source = next((p[5:] for p in parts if p.startswith('[of] ')), None)
            owner = self.subject(source) if source else row
            for token in parts:
                for domain in ('item', 'ability'):
                    prefix = '[from] ' + domain + ': '
                    if not token.startswith(prefix): continue
                    value = token[len(prefix):]
                    evidence_owner = owner
                    if kind == '-heal':
                        evidence_owner = row if not source or (domain == 'ability' and value in _ABSORB_HEALS) else None
                    if evidence_owner is not None:
                        self._reveal(evidence_owner, domain, value)
                        if domain == 'item': evidence_owner['item_removed'] = 0
        if kind == '-weather':
            self.facts['weather'] = '|'.join(parts[2:])
        elif kind in ('-fieldstart', '-fieldend'):
            key = 'field:' + parts[2].removeprefix('move: ')
            if kind == '-fieldstart': self.facts[key] = line
            else: self.facts.pop(key, None)
        elif kind in ('-sidestart', '-sideend'):
            key = 'side:' + parts[2][:2] + ':' + parts[3].removeprefix('move: ')
            if kind == '-sidestart': self.facts[key] = line
            else: self.facts.pop(key, None)

    def _reveal(self, row, domain, value):
        row['last_seen_' + domain] = category({'item': 'items', 'ability': 'abilities'}[domain], value)
        row[domain + '_evidence_known'] = 1

    def _hp(self, row, condition):
        hp, denominator, known, _, fainted, status = health(condition, percent=row['percent'], previous_max=row['shown_denominator'])
        row.update(shown_hp=hp, shown_denominator=denominator, hp_known=known,
                   hp_is_percent=int(row['percent']), fainted=fainted, status=status)

    def appearances(self, side):
        return deepcopy(self.encounters[side])
