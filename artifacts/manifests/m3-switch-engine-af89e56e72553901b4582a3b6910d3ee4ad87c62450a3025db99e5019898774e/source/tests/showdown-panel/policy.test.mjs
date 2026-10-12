import test from 'node:test';
import assert from 'node:assert/strict';
import {BattleStream, seedFor} from '../../packages/reference-node/engine.mjs';
import {PlayerProjection} from '../../packages/reference-node/observations.mjs';
import {createActor, damageProxy, chooseOpponent, switchValue} from '../../packages/showdown-panel/policy.mjs';
import {TacticalView} from '../../packages/showdown-panel/view.mjs';
import {legalChoices} from '../../packages/showdown-bot/protocol.mjs';
import {playGame as baselineGame} from '../../packages/showdown-bot/campaign.mjs';
import {playGame} from '../../packages/showdown-panel/game.mjs';

const own = {details: 'Pikachu, L80', condition: '200/200', stats: {atk: 150, def: 120, spa: 150, spd: 120, spe: 200},
  ability: 'static', item: '', moves: ['thunderbolt', 'surf'], boosts: {}};
const foe = {species: 'Squirtle', level: 80, condition: '100/100', boosts: {}, moves: [], ability: null, item: null};
function state() {
  const s = new TacticalView(); s.ingest('|switch|p1a: Pikachu|Pikachu, L80|200/200');
  s.ingest('|switch|p2a: Squirtle|Squirtle, L80|100/100'); s.ingest('|turn|1'); return s;
}
function request() {
  return {side: {id: 'p1', name: 'Baseline 1', pokemon: [{...structuredClone(own), ident: 'p1: Pikachu', active: true}]},
    active: [{moves: [{id: 'thunderbolt', pp: 10}, {id: 'surf', pp: 10}]}]};
}

test('public damage proxies respond to observed stages, known items and defender typing', () => {
  const v = state(), base = damageProxy('thunderbolt', own, foe, v);
  assert(base > damageProxy('surf', own, foe, v));
  assert(damageProxy('thunderbolt', {...own, boosts: {spa: 2}}, foe, v) > base);
  assert(damageProxy('thunderbolt', own, {...foe, boosts: {spd: 2}}, v) < base);
  assert(damageProxy('thunderbolt', {...own, item: 'choicespecs'}, foe, v) > base);
  assert.equal(damageProxy('thunderbolt', own, {...foe, species: 'Dugtrio'}, v), 0);
  assert.equal(damageProxy('thunderbolt', own, {...foe, ability: 'voltabsorb'}, v), 0);
});

test('only revealed moves/items/abilities enter opponent memory; switches reset stat stages', () => {
  const s = state(); assert.deepEqual(s.active.p2.moves, []); assert.equal(s.active.p2.ability, null);
  s.ingest('|move|p2a: Squirtle|Water Gun|p1a: Pikachu'); s.ingest('|-boost|p2a: Squirtle|spd|2');
  s.ingest('|-ability|p2a: Squirtle|Torrent'); s.ingest('|-item|p2a: Squirtle|Leftovers');
  s.ingest('|switch|p2a: Other|Bulbasaur, L80|100/100'); s.ingest('|switch|p2a: Squirtle|Squirtle, L80|90/100');
  assert.deepEqual(s.active.p2.boosts, {}); assert.deepEqual(s.active.p2.moves, ['watergun']);
  assert.equal(s.active.p2.ability, 'torrent'); assert.equal(s.active.p2.item, 'leftovers');
});

test('Illusion reveal preserves observed moves and stages; public statuses and hazards update', () => {
  const s = state(); s.ingest('|-boost|p2a: Squirtle|spa|2'); s.ingest('|move|p2a: Squirtle|Dark Pulse|p1a: Pikachu');
  s.ingest('|replace|p2a: Zoroark|Zoroark, L80');
  assert.equal(s.active.p2.boosts.spa, 2); assert.deepEqual(s.active.p2.moves, ['darkpulse']);
  s.ingest('|-status|p2a: Zoroark|brn'); assert(s.active.p2.condition.endsWith(' brn'));
  s.ingest('|-curestatus|p2a: Zoroark|brn'); assert(!s.active.p2.condition.includes(' brn'));
  s.ingest('|-sidestart|p1: Baseline 1|Spikes'); s.ingest('|-sidestart|p1: Baseline 1|Spikes'); assert.equal(s.hazards.p1.spikes, 2);
  s.ingest('|-sideend|p1: Baseline 1|Spikes'); assert.equal(s.hazards.p1.spikes, 0);
  s.ingest('|switch|p2a: Squirtle|Squirtle, L80|100/100'); assert.deepEqual(s.active.p2.moves, []);
  assert.throws(() => s.ingest('|split|p2'), /Unprojected/);
});

test('selective stage clears, Psych Up and swaps follow pinned public events', () => {
  const s = state(); s.ingest('|-boost|p2a: Squirtle|atk|2'); s.ingest('|-unboost|p2a: Squirtle|spe|1');
  s.ingest('|-clearnegativeboost|p2a: Squirtle|[silent]'); assert.deepEqual(s.active.p2.boosts, {atk: 2});
  s.ingest('|-copyboost|p1a: Pikachu|p2a: Squirtle|[from] move: Psych Up');
  assert.deepEqual(s.active.p1.boosts, {atk: 2});
  s.ingest('|-clearpositiveboost|p2a: Squirtle|p1a: Pikachu|move: Spectral Thief');
  assert.deepEqual(s.active.p2.boosts, {});
  s.ingest('|-swapboost|p1a: Pikachu|p2a: Squirtle|atk, spa|[from] move: Power Swap');
  assert.equal(s.active.p1.boosts.atk, 0); assert.equal(s.active.p2.boosts.atk, 2);
});

test('Illusion keeps earlier disguise knowledge separate from the current appearance', () => {
  const s = state(); s.ingest('|move|p2a: Squirtle|Water Gun|p1a: Pikachu');
  s.ingest('|-item|p2a: Squirtle|Eviolite');
  s.ingest('|switch|p2a: Other|Bulbasaur, L80|100/100');
  s.ingest('|switch|p2a: Squirtle|Squirtle, L80|100/100');
  s.ingest('|move|p2a: Squirtle|Dark Pulse|p1a: Pikachu'); s.ingest('|replace|p2a: Zoroark|Zoroark, L80');
  assert.deepEqual(s.active.p2.moves, ['darkpulse']); assert.equal(s.active.p2.item, null);
  s.ingest('|switch|p2a: Squirtle|Squirtle, L80|100/100');
  assert.deepEqual(s.active.p2.moves, ['watergun']); assert.equal(s.active.p2.item, 'eviolite');
});

test('actual pinned Psych Up and Power Swap events update observed stages', async () => {
  const view = new TacticalView(), observations = [];
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) {
      if (type !== 'update') return;
      const projection = new PlayerProjection(); projection.ingest(type, data);
      for (const line of projection.players.p1.protocol) {
        view.ingest(line);
        if (/^\|-(copy|swap)boost\|/.test(line)) observations.push([line.split('|')[1], view.active.p1.boosts.atk, view.active.p2.boosts.atk]);
      }
    }
  }();
  try {
    await stream.write(`>start ${JSON.stringify({formatid: 'gen9anythinggoes@@@!Team Preview', seed: seedFor('panel-boost-events')})}\n` +
      `>player p1 ${JSON.stringify({name: 'A', team: [{species: 'Pikachu', ability: 'Static', moves: ['swordsdance', 'powerswap']}]})}\n` +
      `>player p2 ${JSON.stringify({name: 'B', team: [{species: 'Squirtle', ability: 'Torrent', moves: ['psychup', 'growl']}]})}`);
    await stream.write('>p1 move 1\n>p2 move 1');
    await stream.write('>p1 move 1\n>p2 move 2');
    await stream.write('>p1 move 2\n>p2 move 1');
    assert.deepEqual(observations, [['-copyboost', 2, 2], ['-swapboost', 2, 3], ['-copyboost', 2, 2]]);
    assert.equal(view.active.p1.boosts.atk, stream.battle.p1.active[0].boosts.atk);
    assert.equal(view.active.p2.boosts.atk, stream.battle.p2.active[0].boosts.atk);
  } finally { stream.battle?.destroy(); }
});

test('entry cost is charged only when entering, and first-turn moves work for incoming mons', () => {
  const s = state(); s.turn = 8; s.hazards.p1.stealthrock = 1;
  const active = switchValue(own, foe, s, 'p1', false), entering = switchValue(own, foe, s, 'p1');
  assert.equal(active - entering, 1 / 8);
  assert(damageProxy('fakeout', {...own, enteredTurn: s.turn}, foe, s) > 0);
  assert.equal(damageProxy('fakeout', {...own, enteredTurn: 0}, foe, s), 0);
});

test('accuracy uses the 3-based combined stage and always-hit moves ignore stages', () => {
  const s = state(), ordinary = damageProxy('surf', own, foe, s);
  assert.equal(damageProxy('surf', {...own, boosts: {accuracy: -1}}, foe, s), ordinary * 3 / 4);
  assert.equal(damageProxy('swift', {...own, boosts: {accuracy: -6}}, foe, s), damageProxy('swift', own, foe, s));
});

test('opponents honor masks, refrain from input mutation and choose legal Tera or forced switches', () => {
  for (const kind of ['damage-v1', 'tactical-v1']) {
    const r = request(), s = state(); r.active[0].moves[0].disabled = true; r.active[0].canTerastallize = 'Water';
    const snapshot = JSON.stringify([r, s]); const choice = chooseOpponent(kind, r, s);
    assert(legalChoices(r).includes(choice)); assert(!choice.startsWith('move 1')); assert.equal(JSON.stringify([r, s]), snapshot);
    r.forceSwitch = [true]; delete r.active; r.side.pokemon[0].condition = '0 fnt';
    r.side.pokemon.push({...own, ident: 'p1: Bench', details: 'Raichu, L80', active: false});
    assert.equal(chooseOpponent(kind, r, s), 'switch 2');
  }
});

test('known ability immunity changes choice while unknown abilities stay explicit assumptions', () => {
  const r = request(), s = state(); assert.equal(chooseOpponent('damage-v1', r, s), 'move 1');
  s.ingest('|-ability|p2a: Squirtle|Volt Absorb'); assert.equal(chooseOpponent('damage-v1', r, s), 'move 2');
});

test('the richer driver preserves the sealed baseline/random trajectories', async () => {
  const mon = (species, moves, ability) => ({species, moves, ability, level: 50});
  const teams = [
    {team: [mon('Squirtle', ['watergun', 'tackle'], 'Torrent'), mon('Bulbasaur', ['vinewhip', 'tackle'], 'Overgrow')]},
    {team: [mon('Fennekin', ['ember', 'scratch'], 'Blaze'), mon('Pikachu', ['thundershock'], 'Static')]},
  ];
  const args = {teams, seed: seedFor('panel-baseline-parity'), policySeeds: [seedFor('a'), seedFor('b')]};
  const a = await baselineGame({...args, policies: ['heuristic', 'random']});
  const b = await playGame({...args, policies: ['baseline-v1', 'random-v1']});
  assert.deepEqual(b.input, a.input); assert.deepEqual(b.outcome, a.outcome); assert.deepEqual(b.spectator, a.spectator);
  const strip = ({latency_ms, policy, ...rest}) => rest;
  assert.deepEqual(b.decisions.map(strip), a.decisions.map(strip));
});

test('both new policies have identical actual player inputs/actions under hidden-reserve mutation', async () => {
  async function initial(reserve) {
    const p = new PlayerProjection();
    const stream = new class extends BattleStream { constructor() { super({noCatch: true}); } pushMessage(t, d) { p.ingest(t, d); } }();
    const mon = (species, moves, ability) => ({species, moves, ability, level: 50});
    await stream.write(`>start ${JSON.stringify({formatid: 'gen9anythinggoes@@@!Team Preview', seed: seedFor('panel-hidden')})}\n` +
      `>player p1 ${JSON.stringify({name: 'Baseline 1', team: [mon('Squirtle', ['watergun'], 'Torrent')]})}\n` +
      `>player p2 ${JSON.stringify({name: 'Baseline 2', team: [mon('Fennekin', ['ember'], 'Blaze'), mon(reserve, ['tackle'], 'Run Away')]})}`);
    return {p, stream};
  }
  const a = await initial('Rattata'), b = await initial('Sentret');
  try {
    for (const id of ['damage-v1', 'tactical-v1']) {
      const views = [a, b].map(x => { const actor = createActor(id); x.p.players.p1.protocol.forEach(l => actor.state.ingest(l)); return {actor, request: x.p.players.p1.request}; });
      assert.deepEqual(views[0].request, views[1].request); assert.deepEqual(views[0].actor.state, views[1].actor.state);
      assert.equal(views[0].actor.choose(views[0].request, views[0].actor.state), views[1].actor.choose(views[1].request, views[1].actor.state));
      assert.notEqual(a.stream.battle.p2.pokemon[1].species.id, b.stream.battle.p2.pokemon[1].species.id);
    }
  } finally { a.stream.battle?.destroy(); b.stream.battle?.destroy(); }
});
