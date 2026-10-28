"""Cluster continuation contrasts by scenario, keeping repeated suffix keys together."""
import numpy as np
from .common import outcome_summary

def interval(g):
    if g['outcome']['kind']=='terminal':
        x=g['outcome']['rewards'][g['learner_side']];return x,x
    if g['outcome']['kind']!='administrative_cutoff':raise ValueError('Unknown outcome')
    return 0.,1.

def compare(initial,final,settings):
    if len(initial)!=len(final) or not initial:raise ValueError('Unpaired scenarios')
    grouped={};seen=set()
    for a,b in zip(initial,final):
        for key in ['episode','learner_side','prefix','provenance']:
            if a[key]!=b[key]:raise ValueError('Scenario pair mismatch')
        key=(a['prefix']['scenario_hash'],a['episode'])
        if key in seen:raise ValueError('Duplicate repeat')
        seen.add(key);al,ah=interval(a);bl,bh=interval(b)
        grouped.setdefault(key[0],[]).append((bl-ah,bh-al))
    if len(grouped)!=12 or any(len(v)!=4 for v in grouped.values()):raise ValueError('Expected12 scenarios with4 paired repeats')
    bounds=np.asarray([np.mean(v,axis=0) for _,v in sorted(grouped.items())]);rng=np.random.default_rng(settings['seed'])
    samples=[]
    for first in range(0,settings['resamples'],256):
        ids=rng.integers(0,len(bounds),(min(256,settings['resamples']-first),len(bounds)));samples.extend(bounds[ids].mean(1))
    samples=np.asarray(samples);tail=settings['alpha']/(2*settings['family_cells'])
    ci=[float(np.quantile(samples[:,0],tail)),float(np.quantile(samples[:,1],1-tail))]
    radius=float(np.sqrt(2*np.log(2*settings['family_cells']/settings['alpha'])/len(bounds)))
    point=bounds.mean(0).tolist()
    return {'scenarios':len(grouped),'games':len(initial),'initial':outcome_summary(initial),'final':outcome_summary(final),
        'point_interval':point,'approximate_bootstrap_interval':ci,'hoeffding_interval':[max(-1,point[0]-radius),min(1,point[1]+radius)],
        'gate':'improvement' if ci[0]>0 else 'regression' if ci[1]<0 else 'inconclusive'}
