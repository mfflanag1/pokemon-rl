"""Gen9 fixed Rattata duel: two attacks, explicit semantic chance inputs.

This independent CPU specification is the next JAX port's test oracle, not a
replacement for the pinned Showdown reference or an official random-battle engine.
"""
from dataclasses import dataclass
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROFILE=json.loads((ROOT/'contracts/slice-v1/profile.json').read_text())
MOVES=('tackle','quickattack')


def require(ok,message):
    if not ok:raise ValueError(message)


def damage(crit,roll):
    require(type(crit) is bool and type(roll) is int and 85<=roll<=100,'Unsupported chance outcome')
    base=((2*50//5+2)*40*76//55)//50+2
    if crit:base=base*3//2
    base=base*roll//100
    return (base*6144+2047)//4096  # Gen9 STAB rounds exact halves down.


@dataclass(frozen=True)
class State:
    turn:int=1
    hp:tuple=(105,105)
    pp:tuple=((56,48),(56,48))
    winner:int|None=None

    def snapshot(self):
        return {'turn':self.turn,'phase':'terminal' if self.winner is not None else 'move','winner':self.winner,
            'hp':list(self.hp),'pp':[list(p) for p in self.pp],
            'requests':[] if self.winner is not None else [{'own_hp':self.hp[i],
                'moves':[{'id':m,'pp':self.pp[i][j],'maxpp':(56,48)[j],'disabled':False} for j,m in enumerate(MOVES)]} for i in (0,1)]}


def initial(profile):
    require(profile==PROFILE,'Unsupported profile/configuration')
    return State()


def step(state,actions,events,tie_first):
    require(state.winner is None,'Terminal state')
    require(1<=state.turn<=4 and all(0<h<=105 for h in state.hp),'Unsupported state')
    require(len(actions)==2 and all(type(a) is int and a in (0,1) for a in actions),'Unsupported action')
    require(all(state.pp[i][a]>0 for i,a in enumerate(actions)),'PP exhausted outside reachable slice')
    if actions[0]==actions[1]:
        require(type(tie_first) is int and tie_first in (0,1),'Tied order needs explicit chance outcome');first=tie_first
    else:
        require(tie_first is None,'Unexpected order chance');first=0 if actions[0]>actions[1] else 1
    hp=list(state.hp);pp=[list(p) for p in state.pp];used=0;winner=None
    for actor in (first,1-first):
        require(used<len(events),'Missing semantic chance event');event=events[used];used+=1
        require(set(event)=={'actor','move','crit','roll','damage'} and event['actor']==actor and event['move']==MOVES[actions[actor]],'Wrong chance event identity/order')
        amount=damage(event['crit'],event['roll']);require(event['damage']==amount,'Damage parity mismatch')
        pp[actor][actions[actor]]-=1;hp[1-actor]=max(0,hp[1-actor]-amount)
        if hp[1-actor]==0:winner=actor;break
    require(used==len(events),'Unused semantic events after KO')
    require(winner is not None or state.turn<4,'Proven duration bound violated')
    return State(state.turn if winner is not None else state.turn+1,tuple(hp),tuple(tuple(p) for p in pp),winner)
