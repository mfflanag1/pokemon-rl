"""Install the F36 external transport before entering an unchanged F35 stage."""
import runpy
import sys
from types import ModuleType

STAGES = (
    'pokemon_teacher_assisted.schedules',
    'pokemon_teacher_assisted.demonstrations',
    'pokemon_teacher_assisted.campaign',
    'pokemon_teacher_assisted.external_evaluation',
    'pokemon_teacher_assisted.verification',
    'pokemon_teacher_assisted.reporting',
)
CONSUMERS = ('pokemon_teacher_assisted.external', 'pokemon_teacher_assisted.verification')


def _stale_references(original, old_class):
    references = []
    for name, module in list(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        for attribute, value in list(vars(module).items()):
            if value is old_class and not (module is original and attribute == 'JsonWorker'):
                references.append(name + '.' + attribute)
    return references


def install_overlay():
    """Replace only the external JSON transport binding in this fresh process."""
    loaded = [name for name in (*CONSUMERS, *STAGES) if name in sys.modules]
    if loaded:
        raise RuntimeError('F36 overlay must precede stage imports: ' + ', '.join(loaded))
    from pokemon_external_benchmark import transport as original
    from .transport import JsonWorker
    old_class = original.JsonWorker
    if old_class is JsonWorker:
        raise RuntimeError('F36 overlay is already installed')
    stale = _stale_references(original, old_class)
    if stale:
        raise RuntimeError('Stale external transport bindings before overlay: ' + ', '.join(stale))
    original.JsonWorker = JsonWorker
    stale = _stale_references(original, old_class)
    if stale:
        original.JsonWorker = old_class
        raise RuntimeError('Stale external transport bindings after overlay: ' + ', '.join(stale))
    return {'schema_version': 'f36-runtime-overlay-v1',
        'target': 'pokemon_external_benchmark.transport.JsonWorker',
        'implementation': 'pokemon_teacher_runtime.transport.JsonWorker',
        'stage_modules': list(STAGES), 'stale_bindings_checked': True}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == '--':
        arguments.pop(0)
    if not arguments or arguments[0] not in STAGES:
        raise ValueError('F36 stage must be one of the six unchanged F35 workflow modules')
    install_overlay()
    old_argv = sys.argv
    sys.argv = arguments
    try:
        runpy.run_module(arguments[0], run_name='__main__', alter_sys=True)
    finally:
        sys.argv = old_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
