"""Exact, source-bound exceptions for the generic lifecycle wrapper only."""
from pathlib import Path
import sys

from pokemon_full_train.storage import digest
from pokemon_online_continuation.common import ROOT, read
from pokemon_online_continuation.verification import FORBIDDEN_MODULE_PREFIXES

ADMISSION = ROOT / 'artifacts/reviews/f39-online-continuation/admission/manifest.json'
ADMISSION_SHA256 = '2921e7aaed8dfba6f4826ba4e329027172760d5818c944f9fb518bb67e3d937e'
RUNTIME_EXCEPTIONS = {
    'pokemon_teacher_runtime': 'python/pokemon_teacher_runtime/__init__.py',
    'pokemon_teacher_runtime.lifecycle': 'python/pokemon_teacher_runtime/lifecycle.py',
}


def snapshot():
    """Record loaded names; exact exceptions require matching origin and admitted bytes."""
    if digest(ADMISSION) != ADMISSION_SHA256:
        raise ValueError('F40 original admission drift')
    manifest = read(ADMISSION)
    forbidden, allowed = [], []
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.')
                   for prefix in FORBIDDEN_MODULE_PREFIXES):
            continue
        if name not in RUNTIME_EXCEPTIONS:
            forbidden.append(name)
            continue
        relative = RUNTIME_EXCEPTIONS[name]
        expected_path = (ROOT / relative).resolve()
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        filename = getattr(module, '__file__', None)
        if (not isinstance(origin, str) or not isinstance(filename, str)
                or Path(origin).resolve() != expected_path
                or Path(filename).resolve() != expected_path):
            raise ValueError('F40 runtime exception origin mismatch: ' + name)
        expected_hash = manifest['source_files'][relative]
        if (digest(expected_path) != expected_hash
                or digest(ADMISSION.parent / 'source' / relative) != expected_hash):
            raise ValueError('F40 runtime exception source mismatch: ' + name)
        allowed.append({'name': name, 'path': relative, 'sha256': expected_hash})
    return {'forbidden': sorted(forbidden),
        'allowed_runtime_modules': sorted(allowed, key=lambda item: item['name']),
        'forbidden_prefixes': list(FORBIDDEN_MODULE_PREFIXES),
        'exact_runtime_exceptions': dict(RUNTIME_EXCEPTIONS),
        'original_admission_sha256': ADMISSION_SHA256,
        'scope': 'Before/after loaded-module snapshots only; not a transient-import or arbitrary subprocess execution audit.'}
