"""Paired, cutoff-aware development estimates; exploratory bootstrap coverage."""
import numpy as np
from pokemon_full_train.campaign import score_interval, value_summary


def paired_bounds(initial, final):
    if len(initial) != len(final) or len(initial)%2 or not initial:
        raise ValueError('Incomplete paired evaluation')
    lows,highs=[],[]
    for i in range(0,len(initial),2):
        lo=hi=0.
        for swap,(a,b) in enumerate(zip(initial[i:i+2],final[i:i+2])):
            for key in ['episode','swap','learner_side','opponent','provenance']:
                if a[key]!=b[key]: raise ValueError('Pair identity mismatch')
            if a['swap'] is not bool(swap) or a['learner_side']!=('p2' if swap else 'p1'):
                raise ValueError('Seat orientation mismatch')
            if a['episode']!=i//2: raise ValueError('Missing or duplicate pair')
            al,ah=score_interval(a);bl,bh=score_interval(b)
            lo+=(bl-ah)/2;hi+=(bh-al)/2
        lows.append(lo);highs.append(hi)
    return np.asarray(lows,np.float64),np.asarray(highs,np.float64)


def compare(initial,final,settings):
    lo,hi=paired_bounds(initial,final)
    n=len(lo);rng=np.random.default_rng(settings['seed'])
    lower,upper=[],[]
    for first in range(0,settings['resamples'],256):
        ids=rng.integers(0,n,(min(256,settings['resamples']-first),n))
        lower.extend(lo[ids].mean(1));upper.extend(hi[ids].mean(1))
    tail=settings['alpha']/(2*settings['family_cells'])
    interval=[float(np.quantile(lower,tail)),float(np.quantile(upper,1-tail))]
    radius=float(np.sqrt(2*np.log(2*settings['family_cells']/settings['alpha'])/n))
    return {'pairs':n,'cutoff_aware_point_interval':[float(lo.mean()),float(hi.mean())],
        'approximate_familywise_bootstrap_interval':interval,
        'bounded_hoeffding_interval':[max(-1.,float(lo.mean())-radius),min(1.,float(hi.mean())+radius)],
        'bootstrap_resamples':settings['resamples'],
        'gate':'improvement' if interval[0]>0 else 'regression' if interval[1]<0 else 'inconclusive',
        'initial':value_summary(initial),'final':value_summary(final)}


def learning_gate(results,settings):
    seeds=[]
    for result in results:
        cells=result['evaluation'];base=cells['base_power']
        checks={'base_power_confidence_positive':base['approximate_familywise_bootstrap_interval'][0]>0,
            'base_power_point_gain_at_least_five_points':base['cutoff_aware_point_interval'][0]>=settings['practical_improvement'],
            'no_detected_opponent_regression':all(c['gate']!='regression' for c in cells.values()),
            'final_cutoff_rates_admitted':all(c['final']['cutoffs']/c['final']['games']<=settings['max_final_cutoff_rate'] for c in cells.values())}
        seeds.append({'seed':result['seed'],'checks':checks,'passed':all(checks.values())})
    return {'passed':all(s['passed'] for s in seeds),'per_seed':seeds,
            'scope':'Exploratory approximate-bootstrap gate; not a finite-sample or public-strength certificate.'}
