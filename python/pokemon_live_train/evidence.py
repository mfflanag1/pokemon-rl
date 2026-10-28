"""Source/runtime admission and immutable evidence for M7."""
import json
from pathlib import Path
import jax
import numpy as np
from pokemon_learner.checkpoint import digest, wire, file_hash
from pokemon_switch_engine.scalar import require
from pokemon_soak.campaign import write, utc
from pokemon_live import audit as scalar_audit
from pokemon_continuation import engine
from . import learner as l, checkpoint as cp

ROOT = l.ROOT


def read(path): return json.loads(Path(path).read_bytes())


def once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: f.write(wire(value) + b'\n')


def predecessors():
    checks = 0; result = []
    for name in ('m3-switch-screen', 'm4-bounded-value', 'm5-continuation', 'm6-live-learner'):
        path = ROOT / 'artifacts/reports' / (name + '.json'); doc = read(path)
        require(digest({k:v for k,v in doc.items() if k != 'content_hash'}) == doc['content_hash'], 'Predecessor receipt')
        if name == 'm6-live-learner': require(doc['content_hash'] == l.PLAN['predecessor'], 'M6 identity')
        archive = ROOT / 'artifacts/manifests' / (name + '-' + doc['content_hash'])
        require((archive/'receipt.json').read_bytes() == path.read_bytes(), 'Predecessor archived receipt')
        for kind, key in (('source', 'source_files'), ('evidence', 'evidence_files')):
            for filename, sha in doc[key].items():
                for target in (ROOT/filename, archive/kind/filename):
                    require(file_hash(target) == sha, 'Predecessor drift: ' + str(target)); checks += 1
        result.append({'name':name, 'sha256':file_hash(path)})
    return {'receipts':result, 'live_archive_hash_checks':checks}


def admission():
    # Hash reservation bytes without decoding teams or generating their episodes.
    reservation = ROOT/'artifacts/runs/r7-dataset-v1/teams/final.json'
    return {**cp.admission(), 'predecessors':predecessors(),
            'final_reservation_sha256':file_hash(reservation)}


def counters(s):
    return dict(windows=int(s.windows), updates=int(s.updates), adam_steps=int(s.step), next_game=int(s.next_game))


def fingerprints(result):
    return {k:cp.fingerprint(v) for k,v in zip(('session','buffer','final_state','metrics'), result)}


def replay(out, name, s, admitted, *, scalar=False):
    path = out/name; cp.save(path, s, admitted); loaded = cp.load(path, int(s.seed), admitted)
    require(cp.fingerprint(loaded) == cp.fingerprint(s), 'Checkpoint state differs')
    original = l.transaction(s, int(s.seed)); repeated = l.transaction(loaded, int(s.seed))
    require(fingerprints(original) == fingerprints(repeated), 'Original-next update mismatch')
    audit = None
    if scalar:
        ns,b,final,_ = original
        before = engine.initial_kernel(l.teams(ns.ids, ns.seed)) if int(s.windows)%5 == 0 else s.environment
        audit = scalar_audit.collection(before, ns.ids, int(s.seed), l.PLAN['boundaries'][int(s.windows)%5], b, final)
    return {'file':name,'sha256':file_hash(path),'state':cp.fingerprint(s),'counters':counters(s),
            'next':fingerprints(original),'discarded_transactions':2,'scalar':audit}


def window_summary(before, result):
    s,b,final,metrics = result; valid = np.asarray(b.valid); active = bool(valid.any())
    first = int(before.windows)%5 == 0; last = int(s.windows)%5 == 0
    a = np.asarray(b.actions)[valid]
    new_end=np.asarray(b.terminal).reshape(l.GROUPS,l.BATCH,2).any(axis=(0,2))
    return {'windows':1,'updates':int(active),'empty_windows':int(not active),
        'games':l.BATCH if first else 0,'decisions':int(valid.sum()),
        'terminal':int(b.terminal.sum())//2,'cutoff':int((np.asarray(final.winner)==-1).sum()) if last else 0,
        'post12_decisions':int((valid & (np.asarray(b.features)[:,30]*128>12)).sum()),
        'terminal_after12':int((new_end & (np.asarray(final.turn)>12)).sum()),
        'history_over32_decisions':int((valid & (np.asarray(b.features)[:,34]*128>32)).sum()),
        'actions':np.bincount(a,minlength=4).tolist(),
        'value_min':float(np.asarray(b.values)[valid].min()) if active else None,
        'value_max':float(np.asarray(b.values)[valid].max()) if active else None,
        'return_min':float(np.asarray(b.returns)[valid].min()) if active else None,
        'return_max':float(np.asarray(b.returns)[valid].max()) if active else None,
        'metrics':{k:float(np.mean(np.asarray(v))) for k,v in metrics.items()}}


class Totals:
    def __init__(self):
        self.counts = dict.fromkeys(('windows','updates','empty_windows','games','decisions','terminal','cutoff',
                                    'post12_decisions','terminal_after12','history_over32_decisions'),0)
        self.actions = np.zeros(4,np.int64); self.minimum = {}; self.maximum = {}; self.metric_sums = {}
    def add(self, row):
        for key in self.counts: self.counts[key] += row[key]
        self.actions += row['actions']
        for key in ('value_min','return_min'):
            if row[key] is not None: self.minimum[key] = min(self.minimum.get(key,1.),row[key])
        for key in ('value_max','return_max'):
            if row[key] is not None: self.maximum[key] = max(self.maximum.get(key,0.),row[key])
        for key,v in row['metrics'].items(): self.metric_sums[key] = self.metric_sums.get(key,0.) + v*row['decisions']
    def document(self):
        return {**self.counts,'actions':self.actions.tolist(),**self.minimum,**self.maximum,
            'decision_weighted_metrics':{k:v/max(1,self.counts['decisions']) for k,v in self.metric_sums.items()}}


def closed_counts(totals, initial, final):
    t = totals if isinstance(totals, dict) else totals.document()
    require(t['games'] == t['terminal'] + t['cutoff'], 'Cohort outcomes do not close')
    require(t['decisions'] == sum(t['actions']) and t['windows'] == t['updates'] + t['empty_windows'], 'Decision/window accounting')
    require(final['windows'] - initial['windows'] == t['windows'] and
            final['updates'] - initial['updates'] == t['updates'] and
            final['adam_steps'] - initial['adam_steps'] == 2*t['updates'] and
            final['next_game'] - initial['next_game'] == t['games'], 'Session counter linkage')
