"""Explicit public Dex metadata; no teacher action, scores, rewards or hidden sets."""
import numpy as np
from pokemon_full.features import encode
TYPES=('Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy','Stellar')
EXTRA=847

def to_id(s):return ''.join(c.lower() for c in s if c.isascii() and c.isalnum())

def current_types(mon,catalog):
    if mon is None:return []
    tera=mon.get('tera')
    if tera and tera!='Stellar':return [tera]
    return mon.get('types') or catalog['species'][to_id(mon['species'])]['types']

def extras(player,catalog):
    encode(player) # Strict player-only schema and public opponent field boundary.
    if catalog['schema_version']!='f17-public-metadata-v1' or tuple(catalog['types'])!=TYPES:raise ValueError('Metadata schema/types changed')
    foe=next((p for p in player['opponent'] if p and p['active']),None);ft=current_types(foe,catalog)
    def effect(attack,defense):
        return np.prod([catalog['effectiveness'][attack][d] for d in defense]) if defense and attack in TYPES else 0.
    out=[]
    for mon in player['own']:
        mt=current_types(mon,catalog)
        out.extend(float(t in mt) for t in TYPES)
        out.extend(effect(t,mt)/4 for t in TYPES)
        moves=player['active_moves'] if mon['active'] and player['phase']=='move' else [catalog['moves'][to_id(m)] for m in mon['moves'][:4]]
        for i in range(4):
            if i>=len(moves):out.extend([0.]*25);continue
            m=moves[i];typ=m['type'];category=m['category']
            out.extend(float(t==typ) for t in TYPES)
            out.extend([float(category==c) for c in ['Physical','Special','Status']])
            out.extend([m['base_power']/250,float(typ in mt),effect(typ,ft)/4])
    out.extend(float(t in ft) for t in TYPES)
    x=np.asarray(out,np.float32)
    if x.shape!=(EXTRA,) or not np.isfinite(x).all():raise ValueError('Invalid explicit features')
    return x
