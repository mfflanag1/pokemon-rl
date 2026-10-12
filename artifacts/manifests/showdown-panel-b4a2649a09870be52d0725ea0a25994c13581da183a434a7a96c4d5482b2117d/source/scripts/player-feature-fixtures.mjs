import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {ReferenceSession, defaultSpec} from '../packages/reference-node/runner.mjs';
import {PRNG, seedFor, describe} from '../packages/reference-node/engine.mjs';
import {observation} from '../packages/episodes/session.mjs';
import {admissibleActions, actionMask, actionIndex, fromProtocol, toProtocol} from '../packages/battle-contract/actions.mjs';
import {canonical, hashJSON} from '../packages/battle-contract/canonical.mjs';

// Direct simulator interventions are controlled observer probes, not reachable games.
export async function featureFixtures() {
  const cases = [], pairs = [], semantics = [], requests = [];
  async function probe(viewer, label, mutate) {
    const s = await new ReferenceSession(defaultSpec('r5-observer-probes-v1')).start();
    try {
      const before = observation(s.projection.view(viewer), 0);
      const evidence = await mutate(s, s.stream.battle.getSide(viewer), s.stream.battle.getSide(viewer === 'p1' ? 'p2' : 'p1'));
      const after = observation(s.projection.view(viewer), before.protocol_end);
      const result = {id: `${viewer}/${label}`, viewer, views: [before, after], evidence};
      cases.push(result); return result;
    } finally { s.close(); }
  }
  function flush(s) {
    s.stream.battle.makeRequest('move');
    s.stream.battle.add('message', 'Constructed R5 observer probe');
    s.stream.battle.sendUpdates();
  }
  for (const viewer of ['p1', 'p2']) {
    const base = await probe(viewer, 'baseline', s => { flush(s); return null; });
    for (const kind of ['item', 'ability', 'moves', 'tera', 'stats', 'species', 'rng']) {
      const hidden = await probe(viewer, kind, (s, own, other) => {
        const mon = other.pokemon[1];
        let before, after;
        if (kind === 'rng') {
          before = s.stream.battle.prng.getSeed(); s.stream.battle.prng = new PRNG(seedFor('r5-hidden-alternate'));
          after = s.stream.battle.prng.getSeed();
        } else if (kind === 'stats') {
          before = structuredClone(mon.storedStats); mon.storedStats.atk += 17; after = structuredClone(mon.storedStats);
        } else if (kind === 'species') {
          before = mon.species.id; mon.setSpecies(s.stream.battle.dex.species.get(before === 'pikachu' ? 'eevee' : 'pikachu')); after = mon.species.id;
        } else if (kind === 'moves') {
          before = structuredClone(mon.moveSlots);
          mon.moveSlots = [{move: 'Splash', id: 'splash', pp: 1, maxpp: 40, target: 'self', disabled: false, disabledSource: '', used: false}];
          after = structuredClone(mon.moveSlots);
        } else {
          const key = kind === 'tera' ? 'teraType' : kind;
          before = mon[key]; mon[key] = kind === 'item' ? (before === 'leftovers' ? 'choicescarf' : 'leftovers') :
            kind === 'ability' ? (before === 'pressure' ? 'runaway' : 'pressure') : (before === 'Water' ? 'Fire' : 'Water');
          after = mon[key];
        }
        assert.notDeepEqual(before, after); flush(s); return {kind, before, after};
      });
      assert.deepEqual(hidden.views, base.views);
      pairs.push({a: base.id, b: hidden.id, equal: true, kind: `hidden_${kind}`});
    }
    const untouched = await probe(viewer, 'uncommitted', () => null);
    const committed = await probe(viewer, 'opponent-commitment', async (s, own, other) => {
      const v = s.projection.view(other.id), action = admissibleActions(v.request).find(a => a.kind === 'move');
      const result = await s.submit(other.id, v.request_id, toProtocol(action, v.request));
      assert.equal(result.accepted, true); assert.ok(other.choice.actions.length);
      return {command: toProtocol(action, v.request), committed_actions: other.choice.actions.length};
    });
    assert.deepEqual(untouched.views, committed.views);
    pairs.push({a: untouched.id, b: committed.id, equal: true, kind: 'hidden_commitment'});
    const hp = [];
    for (const exact of [197, 199, 201]) {
      hp.push(await probe(viewer, `hp-${exact}`, (s, own, other) => {
        const mon = other.active[0]; mon.maxhp = 400; mon.hp = exact;
        s.stream.battle.add('-damage', mon, mon.getHealth); flush(s); return {exact_hp: exact, max_hp: 400};
      }));
    }
    assert.deepEqual(hp[0].views, hp[1].views);
    assert.notDeepEqual(hp[0].views, hp[2].views);
    pairs.push({a: hp[0].id, b: hp[1].id, equal: true, kind: 'hidden_hp_bucket'},
      {a: hp[0].id, b: hp[2].id, equal: false, kind: 'visible_hp_change'});
    for (const actor of ['own', 'opponent']) {
      const result = await probe(viewer, `${actor}-transform`, (s, own, other) => {
        const mon = actor === 'own' ? own.active[0] : other.active[0], target = actor === 'own' ? other.active[0] : own.active[0];
        const before = {hp: mon.hp, maxhp: mon.maxhp, level: mon.level, base_stats: structuredClone(mon.baseStoredStats)};
        assert.equal(mon.transformInto(target, s.stream.battle.dex.abilities.get('imposter')), true);
        flush(s);
        return {actor, before, target_public_ident: target.toString(), target_species: target.species.id};
      });
      assert.ok(result.views[1].protocol_delta.some(l => l.startsWith('|-transform|') && l.split('|')[3] === result.evidence.target_public_ident));
      semantics.push({case: result.id, kind: `${actor}_transform`});
    }
    const illusion = await probe(viewer, 'illusion-replace', (s, own, other) => {
      const mon = other.active[0]; mon.illusion = other.pokemon[5];
      s.stream.battle.add('switch', mon, mon.getFullDetails);
      s.stream.battle.sendUpdates();
      const disguised = observation(s.projection.view(viewer), 0);
      const apparent = mon.illusion.species.id; mon.illusion = null;
      s.stream.battle.add('replace', mon, mon.getUpdatedDetails()); flush(s);
      return {disguised, apparent, real: mon.species.id};
    });
    semantics.push({case: illusion.id, kind: 'illusion_replace'});
    for (const [ability, move] of [['waterabsorb', 'watergun'], ['voltabsorb', 'thundershock'], ['dryskin', 'watergun'], ['eartheater', 'mudslap']]) {
      const result = await probe(viewer, `absorb-${ability}`, (s, own, other) => {
        const target = other.active[0], source = own.active[0];
        target.ability = ability; target.abilityState.id = ability;
        source.ability = 'pressure'; source.abilityState.id = 'pressure';
        target.hp = Math.floor(target.maxhp / 2);
        // Invoke the pinned event dispatcher and ability callback; keep this labelled constructed.
        s.stream.battle.runEvent('TryHit', target, source, s.stream.battle.dex.getActiveMove(move));
        flush(s);
        return {ability, target: target.toString(), attacker: source.toString()};
      });
      assert.ok(result.views[1].protocol_delta.some(l => l.startsWith(`|-heal|${result.evidence.target}|`) && l.includes(`[of] ${result.evidence.attacker}`)));
      semantics.push({case: result.id, kind: 'absorb_heal'});
    }
    const r = base.views[0].request;
    requests.push({label: `${viewer}/move`, request: r});
    const tera = structuredClone(r); tera.active[0].canTerastallize = tera.side.pokemon[0].teraType;
    requests.push({label: `${viewer}/tera`, request: tera});
    for (const mode of ['wait', 'pass', 'switch', 'revival']) {
      const x = structuredClone(r); delete x.active;
      if (mode === 'wait') x.wait = true;
      else x.forceSwitch = [mode !== 'pass'];
      if (mode === 'revival') { x.side.pokemon[0].condition = '0 fnt'; x.side.pokemon[0].reviving = true; }
      requests.push({label: `${viewer}/${mode}`, request: x});
    }
    for (const id of ['recharge', 'struggle']) {
      const x = structuredClone(r); x.active = [{moves: [{move: id, id}]}]; requests.push({label: `${viewer}/${id}`, request: x});
    }
  }
  for (const c of requests) {
    c.mask = actionMask(c.request);
    c.actions = admissibleActions(c.request).map(action => {
      const command = toProtocol(action, c.request);
      assert.deepEqual(fromProtocol(command, c.request), action);
      return {action, command, index: action.kind === 'pass' ? null : actionIndex(action)};
    });
  }
  const body = {version: 'r5-controlled-observer-probes-1', profile: 'constructed-observer-interventions',
    reference_manifest: describe(), cases, pairs, semantics, requests};
  return {...body, content_hash: hashJSON(body)};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw new Error('Usage: node scripts/player-feature-fixtures.mjs OUTPUT.json');
  const result = await featureFixtures(); writeFileSync(process.argv[2], canonical(result) + '\n');
  console.log(JSON.stringify({cases: result.cases.length, pairs: result.pairs.length, semantic_probes: result.semantics.length, requests: result.requests.length, content_hash: result.content_hash}));
}
