"""Exact, explicitly authorized F37 cutoff; frozen F35/F36 files stay intact.

This is not a configurable clock extension. The sole admitted deadline is bound
by immutable user-approval bytes. Scientific coordinates, budgets, original start
and checkpoint remain identical to the frozen F35 plan.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

from pokemon_teacher_runtime.stage import STAGES
from . import TEACHER_POLICY

ROOT = Path(__file__).resolve().parents[2]
DEADLINE_UTC = '2026-10-28T18:00:00Z'
DEADLINE = datetime(2026, 10, 28, 18, 0, 0, tzinfo=timezone.utc).timestamp()
ORIGINAL_DEADLINE_UTC = '2026-10-28T07:36:08Z'
ORIGINAL_DEADLINE = datetime(2026, 10, 28, 7, 36, 8, tzinfo=timezone.utc).timestamp()
AUTHORIZATION_RELATIVE = 'artifacts/reviews/f37-teacher-legality/deadline-authorization-v1.json'
AUTHORIZATION_SHA256 = 'edd891d6d8f354eb0a8842540cb2eb910fbbb6c529cad70a67a611031c37277c'
ORIGINAL_PLAN_RELATIVE = 'contracts/teacher-assisted-v1/plan.json'
ORIGINAL_PLAN_SHA256 = '25791644ca835144f761f003ed9767a9d7ff81372bd3d063d25745a8220fb8c9'
_original_validator = None
_original_before_deadline = None
_evidence = None


def _verified_json(relative, expected):
    path = ROOT / relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('F37 deadline evidence path redirected: ' + relative)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('F37 deadline evidence hash changed: ' + relative)
    return json.loads(data)


def _authorization():
    approval = _verified_json(AUTHORIZATION_RELATIVE, AUTHORIZATION_SHA256)
    expected = {'schema_version': 'f37-deadline-authorization-v1', 'approved': True,
        'approved_deadline_utc': DEADLINE_UTC, 'new_attempt': 'F37', 'clock_reset': False,
        'original_start_utc': '2026-10-27T23:36:08Z',
        'original_checkpoint_utc': '2026-10-28T03:36:08Z',
        'original_deadline_utc': ORIGINAL_DEADLINE_UTC,
        'scientific_seeds_or_budgets_changed': False, 'single_fresh_comparison_only': True,
        'github_publication_authorized': False}
    if any(type(approval.get(k)) is not type(v) or approval[k] != v for k, v in expected.items()):
        raise ValueError('F37 exact deadline authorization required')
    return approval


def _validate_extended_plan(plan):
    _authorization()
    if not isinstance(plan, dict) or plan.get('deadline_utc') != DEADLINE_UTC:
        raise ValueError('F37 permits only the explicitly approved fixed deadline')
    execution = plan.get('execution', {})
    if (not isinstance(execution, dict)
            or execution.get('schema_version') != 'f37-teacher-legality-execution-v1'
            or execution.get('teacher_policy', {}).get('policy_id') != TEACHER_POLICY
            or execution.get('deadline_authorization') != {
                'path': AUTHORIZATION_RELATIVE, 'sha256': AUTHORIZATION_SHA256}):
        raise ValueError('F37 plan lacks exact deadline approval binding')
    normalized = deepcopy(plan)
    normalized['deadline_utc'] = ORIGINAL_DEADLINE_UTC
    _original_validator(normalized)
    original = _verified_json(ORIGINAL_PLAN_RELATIVE, ORIGINAL_PLAN_SHA256)
    if {k: v for k, v in normalized.items() if k != 'execution'} != original:
        raise ValueError('F37 changed frozen scientific plan, budget, start or checkpoint')
    return plan


def _stale_aliases(common):
    stale = []
    for name, module in list(sys.modules.items()):
        if not isinstance(module, ModuleType) or module is common or name == __name__:
            continue
        for attribute, value in list(vars(module).items()):
            if value is common.validate_plan or (attribute == 'DEADLINE' and value is common.DEADLINE):
                stale.append(name + '.' + attribute)
    return stale


def _check_installed(common):
    if (common.DEADLINE != DEADLINE or common.validate_plan is not _validate_extended_plan
            or common.before_deadline is not _original_before_deadline):
        raise RuntimeError('F37 deadline overlay bindings changed')
    for name in STAGES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if ('DEADLINE' in vars(module) and module.DEADLINE != DEADLINE
                or 'before_deadline' in vars(module) and module.before_deadline is not common.before_deadline
                or 'validate_plan' in vars(module) and module.validate_plan is not common.validate_plan):
            raise RuntimeError('Stale F37 stage deadline binding: ' + name)


def install_deadline():
    """Install once before stage imports; repeat only verifies unchanged bindings."""
    global _original_validator, _original_before_deadline, _evidence
    _authorization()
    from pokemon_teacher_assisted import common
    if _evidence is not None:
        _check_installed(common)
        return deepcopy(_evidence)
    loaded = [name for name in STAGES if name in sys.modules]
    if loaded:
        raise RuntimeError('F37 deadline must precede stage imports: ' + ', '.join(loaded))
    if common.DEADLINE != ORIGINAL_DEADLINE or common.validate_plan.__module__ != common.__name__:
        raise RuntimeError('Unexpected original F35 deadline or validator binding')
    stale = _stale_aliases(common)
    if stale:
        raise RuntimeError('Stale F35 deadline/validator aliases: ' + ', '.join(stale))
    _verified_json(ORIGINAL_PLAN_RELATIVE, ORIGINAL_PLAN_SHA256)
    _original_validator = common.validate_plan
    _original_before_deadline = common.before_deadline
    common.DEADLINE = DEADLINE
    common.validate_plan = _validate_extended_plan
    _evidence = {'schema_version': 'f37-fixed-deadline-overlay-v1',
        'deadline_utc': DEADLINE_UTC, 'deadline_epoch': DEADLINE,
        'original_deadline_utc': ORIGINAL_DEADLINE_UTC,
        'authorization': {'path': AUTHORIZATION_RELATIVE, 'sha256': AUTHORIZATION_SHA256},
        'original_plan': {'path': ORIGINAL_PLAN_RELATIVE, 'sha256': ORIGINAL_PLAN_SHA256},
        'clock_reset': False, 'scientific_plan_unchanged': True,
        'stage_modules': list(STAGES), 'stale_bindings_checked': True}
    _check_installed(common)
    return deepcopy(_evidence)
