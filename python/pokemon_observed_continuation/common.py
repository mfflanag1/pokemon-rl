"""Fixed F32 replacement, observed transport admission and complete prior history."""
import hashlib
from pokemon_full_train.storage import ROOT, digest, canonical
from pokemon_nonlinear_continuation.common import (
    read, initial, budget, BUDGETS, EXPECTED, team_keys, save, load, metadata,
    same_state, evaluation_specs, diagnostic_specs, check_collection, selfplay_value,
    tactical, shapes, check_state, load_policy, parameter_hash, analysis)
from pokemon_nonlinear_continuation.common import admit as prior_admit

PLAN = ROOT/'contracts/observed-continuation-v1/plan.json'
BASE_PLAN = ROOT/'contracts/nonlinear-continuation-v1/plan.json'
CHANGES = {'schema_version': 'f32-observed-continuation-plan-1', 'evaluation_seed': 1027263201, 'selfplay_evaluation_seed': 1027263202, 'statistics': {'method': 'paired-cluster-percentile-bootstrap', 'family_cells': 27, 'alpha': 0.05, 'resamples': 20000, 'seed': 1027263203, 'practical_improvement': 0.05, 'max_final_cutoff_rate': 0.05}, 'evaluation': 'Fresh fixed F32 development panels at absolute cohorts256,512,1024. Reevaluate all six F29 continuation starts on new fixed schedules distinct from F29/F30; all other evaluation counts and gates unchanged. Admit partial F30 and F31 engineering exact-team histories. No F30 outcomes reused or cells stitched.', 'version': 'observed-continuation-f32-v1', 'replacement': 'New independently executed replacement for the incomplete F30 workflow, not an independent training replication. Retain F29 training/selection/optimizer schedules and all six same-arm states. No resume, partial-cell reuse, increased deadline, retries, budget changes or checkpoint selection. F30 failure remains preserved and its cause unresolved.', 'worker_qualification_receipt': 'artifacts/reports/f31-worker-watch.json', 'worker_qualification_sha256': '1e6414181c2fb0a731e165bd922ceaeaa96870db5cbc7d0774a6058da977db02', 'failed_execution_receipt': 'artifacts/reports/f30-nonlinear-continuation-failure.json', 'failed_execution_sha256': '47504c3563d5b4bbf93d39a1ef1c4e6a63e73f46658c1162f126352bddeb3d66', 'observed_transport': 'pokemon_worker_watch.collector.Pool; unchanged 60-second combined write/read deadline and zero retries, coordinator-only phase/resource evidence'}


def validate_plan(plan):
    expected = {**read(BASE_PLAN), **CHANGES}
    if plan != read(PLAN) or plan != expected:
        raise ValueError('Fixed F32 replacement plan: only registered identity, evaluation and admission changes')


def preserved_receipt(path, expected_sha, prefix):
    if digest(path) != expected_sha:
        raise ValueError('Preserved receipt digest: '+str(path))
    receipt = read(path); body = dict(receipt); content_hash = body.pop('content_hash')
    if hashlib.sha256(canonical(body)).hexdigest() != content_hash:
        raise ValueError('Preserved receipt content hash')
    archive = ROOT/'artifacts/manifests'/(prefix+'-'+content_hash)
    if (archive/'receipt.json').read_bytes() != path.read_bytes():
        raise ValueError('Preserved receipt archive mismatch')
    for rel, sha in receipt['source_files'].items():
        frozen = archive/'source'/rel if prefix == 'f31-worker-watch' else ROOT/receipt['source_snapshot_root']/rel
        if digest(ROOT/rel) != sha or digest(frozen) != sha:
            raise ValueError('Preserved source drift: '+rel)
    for rel, sha in receipt['evidence_files'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError('Preserved evidence drift: '+rel)
        if prefix == 'f31-worker-watch' and digest(archive/'evidence'/rel) != sha:
            raise ValueError('Preserved evidence archive drift: '+rel)
    return receipt


def add_history(run, training, development):
    """Read persisted games, including partial F30; never claim its completion."""
    for path in sorted(run.glob('*/*/update-*.json')):
        training.update(team_keys(read(path)['ledger']['games']))
    for pattern in ['*/*/*/games/*.json', '*/*/*/selfplay/*.json']:
        for path in sorted(run.glob(pattern)):
            development.update(team_keys(read(path)['games']))


def admit(plan):
    validate_plan(plan)
    watched = preserved_receipt(ROOT/plan['worker_qualification_receipt'], plan['worker_qualification_sha256'], 'f31-worker-watch')
    failed = preserved_receipt(ROOT/plan['failed_execution_receipt'], plan['failed_execution_sha256'], 'f30-nonlinear-continuation-failure')
    if watched['status'] != 'passed' or watched['summary']['tests']['total'] != 37:
        raise ValueError('F31 engineering qualification missing')
    if failed['status'] != 'failed' or failed['scientific_comparison_completed']:
        raise ValueError('F30 failed-execution identity changed')
    previous, oracle, training, development = prior_admit(read(BASE_PLAN))
    add_history(ROOT/failed['run'], training, development)
    f31 = ROOT/'artifacts/diagnostics/f31-worker-timeout/parity'
    evidence = sorted(f31.glob('*/*/original/result.json'))
    if len(evidence) != 6:
        raise ValueError('F31 engineering history incomplete')
    for path in evidence:
        training.update(team_keys(read(path)['ledger']['games']))
    preflight = ROOT/'artifacts/reviews/f32-observed-continuation/preflight'
    if (preflight/'report.json').exists():
        evidence = sorted(preflight.glob('*/*/result.json'))
        if len(evidence) != 6 or read(preflight/'report.json')['status'] != 'passed':
            raise ValueError('F32 engineering history incomplete')
        for path in evidence:
            training.update(team_keys(read(path)['ledger']['games']))
    if training & development:
        raise ValueError('Historical role collision')
    return previous, oracle, training, development
