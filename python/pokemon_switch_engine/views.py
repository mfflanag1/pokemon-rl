"""Encode compact observations solely from one player's received reference wire."""
from copy import deepcopy
import json
import re
from pokemon_switch.contract import mask as request_mask
from pokemon_switch.views import actor as checked_actor
from .scalar import SPECIES, require

PP = [[40,56],[40,56],[24,56]]
MOVES = [['Vine Whip','Tackle'],['Water Gun','Tackle'],['Incinerate','Scratch']]
NAMES = ['Bulbasaur','Squirtle','Fennekin']


def encode(view):
    require(isinstance(view,dict) and set(view)=={'side','request_id','turn','request','protocol'}, 'Player-only view boundary')
    require(view['side'] in ('p1','p2'), 'Player side')
    lines=view['protocol']; terminal=any(l.startswith('|win|') or l=='|tie|' for l in lines)
    if not terminal: checked_actor(view)
    observer=int(view['side'][1])-1
    own=None; own_pp=None; opponent=[[0,-1,0,0],[0,-1,0,0]]; active=[0,0]
    history=[]; turn=0; pending=None; request=None; requests=0

    def identity(text):
        match=re.fullmatch(r'p([12])a?: ([AB])',text)
        require(match is not None, 'Member identity')
        return int(match[1])-1,ord(match[2])-ord('A')

    def health(text,side):
        if text=='0 fnt': return 0
        pair=text.split('/');require(len(pair)==2 and all(p.isdecimal() for p in pair), 'Unadmitted HP/status')
        if side!=observer: require(pair[1]=='100','Exact opponent HP leak')
        return int(pair[0])

    for line in lines:
        fields=line.split('|'); kind=fields[1] if len(fields)>1 else ''
        if kind=='request':
            request=json.loads(line[9:]);requests+=1
            if request and 'side' in request:
                require(request['side']['id']==view['side'], 'Foreign request')
                if own is None: own=[[0]*9 for _ in range(2)];own_pp=[[0,0],[0,0]]
                for mon in request['side']['pokemon']:
                    side,member=identity(mon['ident']);species=NAMES.index(mon['details'].split(',')[0]);st=mon['stats']
                    maximum=int(mon['condition'].split('/')[1]) if '/' in mon['condition'] else own[member][2]
                    if own[member][0]==0: own_pp[member]=PP[species][:]
                    own[member]=[species+1,health(mon['condition'],side),maximum,*[st[k] for k in ('atk','def','spa','spd','spe')],int(mon['active'])]
                    if mon['active']: active[side]=member
                if 'active' in request:
                    require(own_pp[active[observer]]==[m['pp'] for m in request['active'][0]['moves']], 'Own PP reconstruction')
        elif kind=='turn': turn=int(fields[2])
        elif kind=='switch':
            side,member=identity(fields[2]);species=NAMES.index(fields[3].split(',')[0]);hp=health(fields[4],side);active[side]=member
            if side==observer and own is not None:
                own[member][1]=hp
                for m in range(2): own[m][8]=int(m==member)
            if side!=observer:
                opponent[member]=[species+1,hp,1,int(hp==0)]
                opponent[1-member][2]=0
            if turn>0: history.append([turn,int(side!=observer),0,member,species,-1,member,hp,0])
        elif kind=='move':
            require(pending is None and turn>0, 'Incomplete move history')
            side,member=identity(fields[2]);target,target_member=identity(fields[4]);require(target==1-side,'Move target')
            species=(own[member][0] if side==observer else opponent[member][0])-1
            require(0<=species<3 and fields[3] in MOVES[species], 'Known revealed move')
            move=MOVES[species].index(fields[3]);pending=[turn,int(side!=observer),1,member,species,move,target_member,None,0]
            if side==observer: own_pp[member][move]-=1
        elif kind=='-crit':
            require(pending is not None, 'Crit without move');pending[8]=1
        elif kind=='-damage':
            side,member=identity(fields[2]);hp=health(fields[3],side)
            require(pending is not None and pending[6]==member and pending[1]==int(side==observer),'Damage without matching move')
            pending[7]=hp;history.append(pending);pending=None
            if side==observer: own[member][1]=hp
            else: opponent[member][1]=hp;opponent[member][3]=int(hp==0)
        elif kind in ('-heal','-status','-boost','-unboost','-terastallize','error','split'):
            raise ValueError('Unsupported protocol effect')
    require(pending is None and own is not None and len(history)<=32, 'Complete bounded player history')
    require(request==view['request'] and requests==view['request_id'] and turn==view['turn'], 'Player counters/request linkage')
    mask=[False]*4 if terminal else request_mask(request)
    phase=3 if terminal else 2 if request.get('wait') else 1 if request.get('forceSwitch') else 0
    return {'own':own,'pp':own_pp,'opponent':opponent,'context':[turn,phase,active[observer],active[1-observer],len(history)],
            'history':history+[[-1]*9 for _ in range(32-len(history))], 'mask':mask}
