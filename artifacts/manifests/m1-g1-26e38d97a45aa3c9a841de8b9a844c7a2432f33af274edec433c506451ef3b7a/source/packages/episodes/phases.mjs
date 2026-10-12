import {hashJSON} from '../battle-contract/canonical.mjs';
import {requestPhase} from '../battle-contract/actions.mjs';

export const SIDES = ['p1', 'p2'];

// Cross-player phase labels are privileged diagnostics, never policy features.
export function phaseFor(requests, terminal = false) {
  if (terminal) return {phase: 'terminal', side_phases: {p1: 'terminal', p2: 'terminal'}, requested_sides: []};
  const side_phases = Object.fromEntries(SIDES.map(side => {
    const request = requests[side], kind = requestPhase(request);
    if (kind !== 'switch') return [side, kind];
    if (!request.forceSwitch[0]) throw new Error('Unexpected singles pass-only switch group');
    const mon = request.side.pokemon[0];
    return [side, mon.reviving ? 'revival' : mon.condition.endsWith(' fnt') ? 'faint_replacement' : 'switch_continuation'];
  }));
  const requested_sides = SIDES.filter(s => side_phases[s] !== 'wait');
  const phases = requested_sides.map(s => side_phases[s]);
  let phase;
  if (phases.length === 2 && phases.every(p => p === 'move')) phase = 'joint_move';
  else if (phases.length && phases.every(p => p === 'faint_replacement')) phase = phases.length === 2 ? 'double_replacement' : 'single_replacement';
  else if (phases.length && phases.every(p => p === 'switch_continuation')) phase = 'switch_continuation';
  else if (phases.length && phases.every(p => p === 'revival')) phase = 'revival';
  else if (phases.length && phases.every(p => p !== 'move')) phase = 'mixed_switch';
  else throw new Error('Unsupported or empty request group');
  return {phase, side_phases, requested_sides};
}

// A selected mechanics audit, not a complete restorable simulator state.
export function mechanicsAudit(battle) {
  const mechanics = {
    turn: battle.turn, rng_seed: battle.prng.getSeed(),
    weather: battle.field.weather, terrain: battle.field.terrain,
    pseudo_weather: Object.keys(battle.field.pseudoWeather).sort(),
    queue: battle.queue.list.map(a => ({choice: a.choice, actor: a.pokemon?.fullname ?? null,
      move: a.move?.id ?? a.moveid ?? null})),
    sides: battle.sides.map(side => ({id: side.id, pokemon_left: side.pokemonLeft,
      conditions: Object.keys(side.sideConditions).sort(),
      pokemon: side.pokemon.map(mon => ({ident: mon.fullname, species: mon.species.id,
        hp: mon.hp, maxhp: mon.maxhp, status: mon.status, fainted: mon.fainted, active: mon.isActive,
        item: mon.item, ability: mon.ability, tera: mon.terastallized || null,
        boosts: {...mon.boosts}, volatiles: Object.keys(mon.volatiles).sort(),
        moves: mon.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp})),
      })),
    })),
  };
  return {mechanics, mechanics_hash: hashJSON(mechanics),
    choices: Object.fromEntries(battle.sides.map(s => [s.id, s.getChoice()]))};
}

export function captureFrame(session, group, event, commitments) {
  const requests = Object.fromEntries(SIDES.map(s => [s, session.projection.players[s].request]));
  const terminal = Boolean(session.result), classified = phaseFor(requests, terminal);
  const pending_sides = terminal ? [] : SIDES.filter(s => session.projection.players[s].pending);
  if (pending_sides.some(s => !classified.requested_sides.includes(s))) throw new Error('Pending side has no request');
  return {group, event, turn: session.stream.battle.turn, ...classified, pending_sides,
    commitments: {...commitments},
    requests: Object.fromEntries(SIDES.map(side => {
      const p = session.projection.players[side];
      return [side, {request_id: p.request_id, request_hash: hashJSON(p.request), protocol_length: p.protocol.length}];
    })),
    audit: mechanicsAudit(session.stream.battle),
  };
}
