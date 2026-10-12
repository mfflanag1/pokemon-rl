import {readFileSync} from 'node:fs';
import {parseJSON, canonical} from './canonical.mjs';
import {validateSchema, contractURL} from './schema.mjs';

const table = parseJSON(readFileSync(new URL('actions.json', contractURL))).actions;
export const ACTION_COUNT = table.length;

export function actionIndex(action) {
  validateSchema(action, 'action');
  const index = table.findIndex(candidate => canonical(candidate) === canonical(action));
  if (index < 0) throw new TypeError('Control actions have no policy index');
  return index;
}

export function actionFromIndex(index) {
  if (!Number.isInteger(index) || index < 0 || index >= ACTION_COUNT) throw new TypeError('Invalid action index');
  return structuredClone(table[index]);
}

export function requestPhase(request) {
  validateSchema(request, 'request');
  return request.wait ? 'wait' : request.forceSwitch ? 'switch' : 'move';
}

export function admissibleActions(request) {
  const phase = requestPhase(request);
  if (phase === 'wait') return [];
  const pokemon = request.side.pokemon;
  const fainted = mon => mon.condition.endsWith(' fnt');
  if (phase === 'switch') {
    if (!request.forceSwitch[0]) return [{kind: 'pass'}];
    const reviving = Boolean(pokemon[0].reviving);
    const switches = pokemon.flatMap((mon, slot) => (
      (slot > 0 || reviving) && (reviving ? fainted(mon) : !fainted(mon)) ? [{kind: 'switch', slot}] : []
    ));
    return switches.length ? switches : [{kind: 'pass'}];
  }
  const active = request.active[0];
  if (active.moves.every(move => move.disabled)) throw new TypeError('No advertised move; expected a Struggle request');
  const actions = active.moves.flatMap((move, slot) => move.disabled ? [] : [
    {kind: 'move', slot, terastallize: false},
    ...(active.canTerastallize ? [{kind: 'move', slot, terastallize: true}] : []),
  ]);
  if (!active.trapped) pokemon.forEach((mon, slot) => {
    if (!mon.active && !fainted(mon)) actions.push({kind: 'switch', slot});
  });
  return actions;
}

export function actionMask(request) {
  const mask = Array(ACTION_COUNT).fill(false);
  for (const action of admissibleActions(request)) if (action.kind !== 'pass') mask[actionIndex(action)] = true;
  return mask;
}

export function assertAdmissible(action, request) {
  validateSchema(action, 'action');
  if (!admissibleActions(request).some(candidate => canonical(candidate) === canonical(action))) {
    throw new TypeError('Action is forbidden by the visible request');
  }
  return action;
}

export function toProtocol(action, request) {
  assertAdmissible(action, request);
  if (action.kind === 'pass') return 'pass';
  return `${action.kind} ${action.slot + 1}${action.terastallize ? ' terastallize' : ''}`;
}

export function fromProtocol(command, request) {
  if (typeof command !== 'string') throw new TypeError('Expected a protocol choice string');
  const match = /^(move ([1-4])( terastallize)?|switch ([1-6])|pass)$/.exec(command);
  if (!match || match[0] !== command) throw new TypeError('Unsupported protocol choice syntax');
  const action = command === 'pass' ? {kind: 'pass'} : match[4] ?
    {kind: 'switch', slot: Number(match[4]) - 1} :
    {kind: 'move', slot: Number(match[2]) - 1, terastallize: Boolean(match[3])};
  return assertAdmissible(action, request);
}
