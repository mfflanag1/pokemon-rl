"""Admit only the sealed F6 probe; model inputs are encoded player views only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pokemon_full.features import encode
from pokemon_full_train.storage import ROOT, canonical, digest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.asarray(arrays[key])
        h.update(canonical([key, str(a.dtype), list(a.shape)])); h.update(a.tobytes())
    return h.hexdigest()


def predecessor(plan, all_files=False):
    path = ROOT/plan['predecessor_receipt']; receipt = json.loads(path.read_text())
    body = dict(receipt); expected = body.pop('content_hash')
    require(expected == plan['predecessor_content_hash'] == hashlib.sha256(canonical(body)).hexdigest(), 'F6 receipt content hash')
    archive = ROOT/'artifacts/manifests'/f'f6-decision-canary-{expected}'/'receipt.json'
    require(path.read_bytes() == archive.read_bytes(), 'F6 receipt archive mismatch')
    require(digest(ROOT/plan['dataset']) == receipt['evidence_files'][plan['dataset']], 'F6 dataset digest mismatch')
    if all_files:
        for rel, sha in receipt['source_files'].items():
            require(digest(ROOT/rel) == sha and digest(ROOT/receipt['source_snapshot_root']/rel) == sha, f'F6 source drift: {rel}')
        for rel, sha in receipt['evidence_files'].items():
            require(digest(ROOT/rel) == sha, f'F6 evidence drift: {rel}')
    return receipt


def prepare(records, plan):
    require(len(records) == 512, 'Expected all 512 F6 probe states')
    seen = set(); encoded = []; labels_by_input = {}
    for row in records:
        spec = row['spec']; context, episode, panel = spec['context'], spec['episode'], spec['panel']
        require(set(spec) == {'context','episode','panel','run_seed'} and
                type(context) is int and 0 <= context < 16 and type(episode) is int and 0 <= episode < 16 and
                panel in ['train','variant'] and spec['run_seed'] == 8622027, 'Invalid probe coordinates')
        key = (panel, episode, context); require(key not in seen, 'Duplicate probe coordinates'); seen.add(key)
        require(row['player']['side'] == ('p2' if context % 2 else 'p1'), 'Context seat mismatch')
        x, mask = encode(row['player'])
        scores = row['scores']
        require(set(scores) == {str(i) for i in np.flatnonzero(mask)}, 'Missing or illegal oracle branches')
        require(all(type(v) in [int,float] and np.isfinite(v) and 0 <= v <= 1 for v in scores.values()), 'Invalid oracle score')
        ordered = sorted(((value, int(action)) for action, value in scores.items()), reverse=True)
        require(len(ordered) > 1 and ordered[0][0] > ordered[1][0], 'Nonunique oracle optimum')
        target = ordered[0][1]
        require(type(row['target']) is int and row['target'] == target, 'Oracle target disagrees with branches')
        input_key = fingerprint({'features':x, 'mask':mask})
        require(input_key not in labels_by_input or labels_by_input[input_key] == target, 'Conflicting targets share model inputs')
        labels_by_input[input_key] = target
        encoded.append((key, x, mask, target, input_key))
    expected = {(panel, episode, context) for panel in ['train','variant'] for episode in range(16) for context in range(16)}
    require(seen == expected, 'Incomplete probe coordinate grid')
    encoded.sort(key=lambda row: row[0])
    splits, inventory, keys = {}, {}, {}
    for name, selection in plan['splits'].items():
        rows = [r for r in encoded if r[0][0] == selection['panel'] and selection['episode_lower'] <= r[0][1] < selection['episode_upper']]
        require(len(rows) == selection['rows'], 'Split row count mismatch')
        split = {'features': np.stack([r[1] for r in rows]), 'mask': np.stack([r[2] for r in rows]),
                 'targets': np.asarray([r[3] for r in rows], np.int32), 'contexts': np.asarray([r[0][2] for r in rows], np.int32)}
        splits[name] = split; keys[name] = {r[4] for r in rows}
        inventory[name] = {'rows':len(rows), 'unique_encoded_inputs':len(keys[name]), 'array_fingerprint':fingerprint(split),
                           'coordinates': [list(r[0]) for r in rows]}
    require(set(splits) == {'fit_train','probe_same','probe_variant'}, 'Wrong split names')
    inventory['overlaps'] = {f'{a}/{b}':len(keys[a] & keys[b]) for i,a in enumerate(keys) for b in list(keys)[i+1:]}
    inventory['all_unique_encoded_inputs'] = len(labels_by_input)
    return splits, inventory


def load_data(plan):
    receipt = predecessor(plan)
    probe = json.loads((ROOT/plan['dataset']).read_text())
    require(probe['status'] == 'passed' and probe['contexts'] == 512 and probe['branches'] == 3584, 'Invalid completed probe')
    splits, inventory = prepare(probe['records'], plan)
    inventory.update(dataset_sha256=digest(ROOT/plan['dataset']), predecessor_receipt_sha256=digest(ROOT/plan['predecessor_receipt']))
    return splits, inventory
