"""Four semantic actions for the closed two-member request profile."""
import json
import re
from pathlib import Path

PROFILE=json.loads((Path(__file__).resolve().parents[2]/'contracts/switch-v1/profile.json').read_text())
SPECIES={s['species']:s for s in PROFILE['species']}
MON_KEYS={'ident','details','condition','active','stats','moves','baseAbility','item','pokeball','ability','commanding','reviving','teraType','terastallized'}


def require(ok,message):
    if not ok:raise ValueError(message)


def obj(value,allowed):
    require(isinstance(value,dict) and not set(value)-allowed,'Unknown request fields or shape')


def ident(text):return re.sub('[^a-z0-9]','',text.lower())


def mask(request):
    obj(request,{'wait','side','active','forceSwitch','rqid','noCancel'})
    if 'wait' in request:
        require(request['wait'] is True and 'active' not in request and 'forceSwitch' not in request,'Wait phase')
        if 'side' not in request:return [False]*4
    side=request.get('side');obj(side,{'id','name','pokemon'});sid=side.get('id')
    require(sid in ('p1','p2'),'Side identity');mons=side.get('pokemon')
    require(isinstance(mons,list) and len(mons)==2,'Team size')
    for m in mons:
        obj(m,MON_KEYS)
        require(type(m.get('active')) is bool and not any(m.get(k) for k in ('reviving','commanding','terastallized')),'Pokemon flags')
        require(m.get('ident') in (sid+': A',sid+': B'),'Member identity')
        p=next((p for p in SPECIES.values() if m.get('details')==f"{p['species']}, L50, M"),None)
        require(p is not None,'Member species/level/gender')
        require(m.get('moves')==[ident(x) for x in p['moves']] and m.get('item')=='','Member moves/item')
        require(m.get('ability')==m.get('baseAbility')==ident(p['ability']),'Member ability')
        c=m.get('condition');require(isinstance(c,str),'HP condition')
        h=re.fullmatch(r'([0-9]+)/([0-9]+)',c)
        require(c=='0 fnt' or h is not None and 0<int(h[1])<=int(h[2]),'HP condition')
        stats=m.get('stats');obj(stats,{'atk','def','spa','spd','spe'})
        require(set(stats)=={'atk','def','spa','spd','spe'} and all(type(v) is int and v>0 for v in stats.values()),'Own stats')
    require({m['ident'] for m in mons}=={sid+': A',sid+': B'} and len({m['details'] for m in mons})==2,'Distinct members')
    require(sum(m['active'] for m in mons)==1,'One active member')
    if request.get('wait'):return [False]*4
    switches=[next(not m['active'] and m['condition']!='0 fnt' for m in mons if m['ident']==sid+': '+n) for n in ('A','B')]
    if 'forceSwitch' in request:
        require(request['forceSwitch']==[True] and type(request['forceSwitch'][0]) is bool and 'active' not in request and any(switches),'Replacement phase')
        return [False,False]+switches
    active=request.get('active');require(isinstance(active,list) and len(active)==1,'Singles request');slot=active[0];obj(slot,{'moves'})
    mon=next(m for m in mons if m['active']);require(mon['condition']!='0 fnt','Fainted active move')
    moves=slot.get('moves');require(isinstance(moves,list) and len(moves)==2,'Move count');legal=[]
    for i,m in enumerate(moves):
        obj(m,{'move','id','pp','maxpp','target','disabled'})
        require(m.get('id')==mon['moves'][i] and isinstance(m.get('move'),str) and ident(m['move'])==m['id'],'Move identity')
        require(type(m.get('pp')) is int and type(m.get('maxpp')) is int and 0<=m['pp']<=m['maxpp'] and m['maxpp']>0,'PP')
        require(type(m.get('disabled')) is bool,'Disabled flag');legal.append(m['pp']>0 and not m['disabled'])
    require(any(legal),'PP exhaustion out of scope')
    return legal+switches


def command(request,index):
    require(type(index) is int and 0<=index<4,'Action index');require(mask(request)[index],'Illegal action')
    if index<2:return f'move {index+1}'
    sid=request['side']['id'];name=('A','B')[index-2]
    return 'switch '+str(next(i+1 for i,m in enumerate(request['side']['pokemon']) if m['ident']==sid+': '+name))
