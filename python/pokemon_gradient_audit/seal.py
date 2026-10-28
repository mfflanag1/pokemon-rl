"""Small diagnostic receipt: verified updates, tested sources and immutable evidence."""
import argparse
import hashlib
from pathlib import Path
from pokemon_full_train.storage import ROOT, canonical, digest, admission, atomic_json
from pokemon_diversity.common import read
from .campaign import PLAN, validate_plan, predecessor

DOCS = ['docs/F27_GRADIENT_AUDIT.md', 'README.md', 'docs/IMPLEMENTATION_PLAN.md',
        'docs/ACCEPTANCE_GATES.md', 'docs/REVIEW_RECORD.md']


def validate_records(report, verification, qualification, manifest, exits):
    plan = read(PLAN)
    validate_plan(plan)
    if report['status'] != 'passed' or report['plan'] != plan or report['counts'] != plan['expected_counts']:
        raise ValueError('Incomplete audit report')
    if verification['status'] != 'passed' or verification['cohorts'] != 18 or verification['input_hash'] != report['input_hash']:
        raise ValueError('Incomplete update verification')
    if (qualification['status'] != 'passed' or qualification['tests_passed'] != 13
            or qualification['sources'] != manifest['sources'] or qualification['runtime'] != manifest['runtime']):
        raise ValueError('Tested source/runtime mismatch')
    if any(x['exit_code'] != 0 for x in exits):
        raise ValueError('Unclean execution exit')


def seal(run, review):
    report = read(run/'report.json'); manifest = read(run/'inputs/manifest.json')
    verification = read(review/'verification.json'); qualification = read(review/'qualification.json')
    exits = [read(review/name) for name in ['production-exit.json','verification-exit.json','tests-exit.json','build-exit.json']]
    validate_records(report,verification,qualification,manifest,exits)
    if digest(run/'report.json') != verification['report_sha256'] or digest(run/'inputs/manifest.json') != report['input_hash']:
        raise ValueError('Verification/report/input digest mismatch')
    for rel, sha in qualification['check_files'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError('Qualification evidence changed: '+rel)
    admission(run,manifest);predecessor(report['plan'])
    for rel in DOCS:
        snapshot = review/'documents'/rel
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists():
            if snapshot.read_bytes() != (ROOT/rel).read_bytes():
                raise ValueError('Completion document drift: '+rel)
        else:
            with snapshot.open('xb') as out:out.write((ROOT/rel).read_bytes())
    paths = [p for p in run.rglob('*') if p.is_file() and not p.is_relative_to(run/'inputs/source')]
    paths += [p for p in review.rglob('*') if p.is_file()]
    archive = ROOT/'artifacts/manifests/f27-gradient-audit-evidence'
    evidence = {}
    for p in sorted(paths):
        rel = p.relative_to(ROOT); dest = archive/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != p.read_bytes():raise ValueError('Evidence archive drift: '+str(rel))
        else:
            with dest.open('xb') as out:out.write(p.read_bytes())
        evidence[str(rel)] = digest(p)
        evidence[str(dest.relative_to(ROOT))] = digest(dest)
    body = {'schema_version':'f27-gradient-audit-receipt-v1','status':'passed',
        'run':str(run.relative_to(ROOT)), 'report_sha256':digest(run/'report.json'),
        'input_hash':report['input_hash'], 'counts':report['counts'],'tests_passed':qualification['tests_passed'],
        'predecessor_receipt':report['plan']['predecessor_receipt'],
        'predecessor_sha256':report['plan']['predecessor_sha256'],
        'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'source_files':manifest['sources'], 'evidence_files':evidence,
        'interpretation':report['plan']['interpretation'], 'new_games':0,
        'strength_evaluation':False,'promotion_authorized':False,
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    h = hashlib.sha256(canonical(body)).hexdigest()
    result = {**body,'content_hash':h}
    content = canonical(result)+b'\n'
    target = ROOT/'artifacts/reports/f27-gradient-audit.json'
    archived = ROOT/'artifacts/manifests'/('f27-gradient-audit-'+h)/'receipt.json'
    for path in [archived,target]:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_bytes()!=content:raise ValueError('Receipt differs on repeat sealing')
        else:
            with path.open('xb') as out:out.write(content)
    print({'status':'passed','receipt_sha256':digest(target),'source_files':len(manifest['sources']),
           'evidence_hashes':len(evidence)},flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True);a=parser.parse_args()
    seal(a.run.resolve(),a.review.resolve())
