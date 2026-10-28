#!/usr/bin/env python3
"""Freeze one F37 input inventory after qualification; never launch science."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'artifacts/reviews/f37-teacher-legality'
PLAN = ROOT / 'contracts/teacher-legality-v1/plan.json'
ADMISSION = REVIEW / 'admission'
spec = importlib.util.spec_from_file_location('f37_admission_checker', ROOT / 'scripts/check-teacher-legality.py')
CHECKER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECKER)


def read(path):
    return json.loads(path.read_text())


def inventory(plan):
    """Keep required F35 inputs; bind F36 history through its unchanged archive."""
    previous = read(ROOT / CHECKER.BINDINGS['failed_predecessor'])
    CHECKER.verify_predecessor(previous)
    original_path = ROOT / 'artifacts/reports/f35-teacher-assisted-failure.json'
    if previous['source_files'].get(str(original_path.relative_to(ROOT))) != CHECKER.sha(original_path):
        raise ValueError('F35 receipt is not bound by preserved F36 failure')
    original = read(original_path)
    paths = set(original['source_files'])
    for relative, expected in original['source_files'].items():
        if previous['source_files'].get(relative) != expected or CHECKER.sha(ROOT / relative) != expected:
            raise ValueError('Original F35 input drift: ' + relative)
    for name, relative in CHECKER.BINDINGS.items():
        paths.add(relative)
        if name.endswith('qualification'):
            record = read(ROOT / relative)
            paths.update(record['source_files'])
            if name == 'legality_qualification':
                paths.update(record.get('evidence_files', {}))
    paths.update(CHECKER.NEW_SOURCES)
    paths.add(CHECKER.PLAN_RELATIVE)
    return sorted(paths)


def main(*, freeze=False):
    if not freeze:
        raise ValueError('Explicit --freeze is required; admission does not launch the workflow')
    sys.path.insert(0, str(ROOT / 'python'))
    from pokemon_teacher_legality.deadline import install_deadline
    install_deadline()
    from pokemon_teacher_assisted.common import before_deadline, bind_inputs, verify_inputs
    before_deadline()
    if ADMISSION.exists() or (REVIEW / 'launch-admission.json').exists():
        raise ValueError('F37 admission already exists; no replacement or overwrite')
    absent = [ROOT / p for p in CHECKER.OUTPUT_PATHS.values()]
    absent += [ROOT / 'artifacts/reports/f37-teacher-legality.json', REVIEW / 'launcher-claim.json',
               REVIEW / 'workflow-claim.json', REVIEW / 'driver-launch.json', REVIEW / 'driver-exit.json']
    if any(p.exists() for p in absent):
        raise ValueError('F37 execution already entered; cannot admit or restart')
    plan = read(PLAN)
    for name, relative in CHECKER.BINDINGS.items():
        entry = plan['execution'][name]
        if entry['path'] != relative:
            raise ValueError('F37 execution evidence path: ' + name)
        actual = CHECKER.sha(ROOT / relative)
        if entry.get('sha256') not in (None, actual):
            raise ValueError('F37 execution evidence was changed after plan creation: ' + name)
        entry['sha256'] = actual
    # The qualification excludes this plan; finalizing its external evidence hashes
    # before source admission avoids a circular content dependency.
    PLAN.write_text(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + '\n')
    paths = inventory(plan)
    sources = {p: CHECKER.sha(CHECKER.bound_path(p)) for p in paths}
    CHECKER.verify_execution({'plan': plan, 'source_files': sources})
    qualification = {'status': 'passed', 'legality_qualification_sha256': plan['execution']['legality_qualification']['sha256'],
        'runtime_qualification_sha256': plan['execution']['runtime_qualification']['sha256'],
        'preserved_f36_source_files': 2182,
        'scope': 'New legality policy qualified; unchanged scientific and runtime evidence admitted through immutable predecessors.'}
    input_hash = bind_inputs(ADMISSION, plan, paths, qualification)
    admitted = verify_inputs(ADMISSION)
    CHECKER.verify_execution(admitted)
    result = {'schema_version': 'f37-launch-admission-v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_hash': input_hash,
        'plan_sha256': CHECKER.sha(PLAN), 'source_files': len(sources),
        'preserved_f36_source_files': 2182, 'original_unchanged_sources': len(read(ROOT / 'artifacts/reports/f35-teacher-assisted-failure.json')['source_files']),
        'scientific_plan_fields_identical_except_approved_deadline': True, 'teacher_policy': plan['execution']['teacher_policy'],
        'deadline_utc': plan['deadline_utc'], 'original_deadline_utc': CHECKER.ORIGINAL_DEADLINE,
        'deadline_authorization_sha256': plan['execution']['deadline_authorization']['sha256'],
        'no_scientific_stage_started': True,
        'legality_qualification_sha256': plan['execution']['legality_qualification']['sha256']}
    with (REVIEW / 'launch-admission.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    main(freeze=args.freeze)
