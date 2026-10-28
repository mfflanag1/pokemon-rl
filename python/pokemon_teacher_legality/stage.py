"""Install F36 transport and F37 teacher legality before unchanged stage imports."""
import runpy
import sys
from types import ModuleType

from pokemon_teacher_runtime.stage import STAGES, install_overlay as install_transport
from . import TEACHER_POLICY
from .deadline import install_deadline


def install_overlay():
    deadline = install_deadline()
    transport = install_transport()
    from pokemon_teacher_assisted import external as original
    from . import external as replacement
    old_run_game = original.run_game
    stale = []
    for name, module in list(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        for attribute, value in list(vars(module).items()):
            if value is old_run_game and not (module is original and attribute == 'run_game'):
                stale.append(name + '.' + attribute)
    if stale:
        raise RuntimeError('Stale external game bindings before F37 overlay: ' + ', '.join(stale))
    original.run_game = replacement.run_game
    return {'schema_version': 'f37-legality-overlay-v1', 'transport': transport, 'deadline': deadline,
            'teacher_policy': TEACHER_POLICY, 'target': 'pokemon_teacher_assisted.external.run_game',
            'implementation': 'pokemon_teacher_legality.external.run_game',
            'stage_modules': list(STAGES), 'stale_bindings_checked': True}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == '--':
        arguments.pop(0)
    if not arguments or arguments[0] not in STAGES:
        raise ValueError('F37 stage must be one of the six unchanged F35 workflow modules')
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
