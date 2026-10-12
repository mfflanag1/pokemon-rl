"""Predeclared exact-binomial cells with a joint Bonferroni family gate."""
import numpy as np
from scipy.stats import binomtest
from pokemon_slice.engine import require


def law_tests(crit,roll,tie,turns,tie_present=None):
    crit=np.asarray(crit);roll=np.asarray(roll);tie=np.asarray(tie);turns=np.asarray(turns)
    n=len(tie);valid=np.ones(n,bool) if tie_present is None else np.asarray(tie_present,bool)
    rows=[]
    def cell(name, outcomes, p):
        count=int(np.count_nonzero(outcomes));size=len(outcomes)
        rows.append({'name':name,'n':size,'count':count,'probability':p,
                     'p_value':float(binomtest(count,size,p).pvalue) if size else 1.0,
                     'expected_count':size*p,'zero_count_p_value':float(binomtest(0,size,p).pvalue) if size else 1.0})
    for a in (0,1):
        cell(f'crit/{a}',crit[:,a],1/24)
        for r in range(85,101):
            cell(f'roll/{a}/{r}',roll[:,a]==r,1/16)
            for c in (0,1):cell(f'crit-roll/{a}/{c}/{r}',(crit[:,a]==c)&(roll[:,a]==r),(1 if c else 23)/384)
    cell('tie',tie[valid]==0,.5)
    for r in range(85,101):
        for s in range(85,101):cell(f'roll-pair/{r}/{s}',(roll[:,0]==r)&(roll[:,1]==s),1/256)
    for c in (0,1):
        for d in (0,1):cell(f'crit-pair/{c}/{d}',(crit[:,0]==c)&(crit[:,1]==d),(1 if c else 23)*(1 if d else 23)/576)
    for a in (0,1):
        for t in (0,1):
            for c in (0,1):cell(f'crit-tie/{a}/{c}/{t}',((crit[:,a]==c)&(tie==t))[valid],(1 if c else 23)/48)
            for r in range(85,101):cell(f'roll-tie/{a}/{r}/{t}',((roll[:,a]==r)&(tie==t))[valid],1/32)
        # Disjoint pairs avoid overlapping Bernoulli trials for lag tests.
        c0=crit[:n//2*2:2,a];c1=crit[1:n//2*2:2,a]
        for c in (0,1):
            for d in (0,1):cell(f'crit-lag/{a}/{c}/{d}',(c0==c)&(c1==d),(1 if c else 23)*(1 if d else 23)/576)
    ts=tie[valid];k=len(ts)//2*2
    for t in (0,1):
        for u in (0,1):cell(f'tie-lag/{t}/{u}',(ts[:k:2]==t)&(ts[1:k:2]==u),.25)
    for turn in range(1,5):
        mask=turns==turn
        for a in (0,1):
            cell(f'turn/{turn}/crit/{a}',crit[mask,a],1/24)
            for r in range(85,101):cell(f'turn/{turn}/roll/{a}/{r}',roll[mask,a]==r,1/16)
        cell(f'turn/{turn}/tie',tie[mask&valid]==0,.5)
    low=roll<93
    for name,events,p in [
        ('crit-equal',crit[:,0]==crit[:,1],(23*23+1)/576),
        ('roll-equal',roll[:,0]==roll[:,1],1/16),
        ('both-low',low[:,0]&low[:,1],.25),('both-high',~low[:,0]&~low[:,1],.25),
        ('crit0-roll1',crit[:,0]&low[:,1],1/48),('crit1-roll0',crit[:,1]&low[:,0],1/48)]:cell(name,events,p)
    cell('tie-roll0',((tie==0)&low[:,0])[valid],.25)
    cell('tie-roll1',((tie==0)&low[:,1])[valid],.25)
    cell('crit-both-tie',((tie==0)&crit[:,0]&crit[:,1])[valid],1/1152)
    require(len(rows)==592 and len({r['name'] for r in rows})==592,'Predeclared test-family mismatch')
    return rows
