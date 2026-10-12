import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {PublicBattle, legalChoices} from '../showdown-bot/protocol.mjs';
import {chooseHeuristic} from '../showdown-bot/policy.mjs';
import {developmentTeams} from '../showdown-bot/development.mjs';

const CASES = ['before-choice', 'lost-write', 'accepted-choice', 'next-turn', 'forced-switch', 'double-drop'];
const plain = x => JSON.parse(JSON.stringify(x));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');

/** Rebuild only the receiving player's public reducer and own request from the
 * saved transport. Check every attempted action, including the deliberately lost write. */
export function auditTrace(report, trace) {
  assert.equal(report.status, 'complete'); assert.deepEqual(report.cases, CASES); assert.deepEqual(report.failures, []);
  assert.deepEqual(report.records.map(r => r.scenario), CASES);
  assert(trace.every(e => CASES.includes(e.scenario)), 'Unknown trace scenario');
  let actions = 0, faults = 0, sentchoices = 0;
  for (const record of report.records) {
    const events = trace.filter(e => e.scenario === record.scenario), connections = new Map();
    const sends = [], injected = [], restores = [], reconnects = [], decisions = [];
    let awaitingDecision = null, retries = 0;
    const get = generation => {
      assert(Number.isSafeInteger(generation) && generation > 0);
      if (!connections.has(generation)) connections.set(generation, new Map());
      return connections.get(generation);
    };
    for (const e of events) {
      if (e.type === 'receive') {
        const rooms = get(e.generation);
        for (const {room: id, line} of e.messages) {
          if (line === '|init|battle') {
            assert(!rooms.has(id), 'Repeated initialization');
            rooms.set(id, {state: new PublicBattle(), request: null, sent: new Set(), rqid: -1});
          }
          const room = rooms.get(id); assert(room, 'Missing initial history'); room.state.ingest(line);
          if (line.startsWith('|request|')) {
            room.request = JSON.parse(line.slice(9));
            if (room.request) {
              assert(Number.isSafeInteger(room.request.rqid) && room.request.rqid >= room.rqid, 'Request order');
              room.rqid = room.request.rqid;
              if (!room.request.wait) assert.equal(room.request.side.name, 'RecoveryA');
            }
          }
          if (line.startsWith('|sentchoice|')) {
            assert(room.request); room.sent.add(JSON.stringify(room.request)); sentchoices++;
          }
          if (line.startsWith('|error|')) {
            assert(line.startsWith('|error|[Unavailable choice]')); room.request = null; retries++;
          }
        }
      } else if (e.type === 'send') {
        assert.equal(awaitingDecision, null, 'Missing preceding decision event');
        const match = /^(battle-[a-z0-9-]+)\|\/choose ([^|\r\n]+)\|(\d+)$/.exec(e.data); assert(match, 'Choice syntax');
        const [, id, choice, rqid] = match, room = get(e.generation).get(id); assert(room?.request && !room.request.wait);
        assert(!room.state.ended); assert.equal(Number(rqid), room.request.rqid);
        const key = JSON.stringify(room.request); assert(!room.sent.has(key), 'Duplicate or accepted pending choice replayed');
        assert(legalChoices(room.request).includes(choice));
        assert.equal(choice, chooseHeuristic(structuredClone(room.request), structuredClone(room.state)));
        room.sent.add(key);
        awaitingDecision = {generation: e.generation, room: id, rqid: Number(rqid), choice, turn: room.state.turn,
          actor_input: plain({request: room.request, state: room.state})};
        assert.equal(typeof e.delivered, 'boolean'); sends.push({...awaitingDecision, delivered: e.delivered});
      } else if (e.type === 'decision') {
        assert(awaitingDecision, 'Decision without attempted send');
        const {scenario, type, session_generation, ...actual} = e;
        assert.deepEqual(actual, awaitingDecision, 'Actor input/action differs from player-visible transport');
        decisions.push(e); awaitingDecision = null; actions++;
      } else if (e.type === 'fault') {
        injected.push(e); faults++;
        if (record.scenario === 'next-turn' || record.scenario === 'forced-switch') {
          const rooms = [...get(e.generation).values()];
          assert(rooms.some(r => record.scenario === 'next-turn' ? r.state.turn >= 2 : r.request?.forceSwitch?.[0]), 'Fault phase');
        }
      } else if (e.type === 'restored') {
        const room = get(e.generation).get(e.room); assert(room);
        assert.equal(e.rqid, room.rqid); assert.equal(e.turn, room.state.turn);
        assert.equal(e.committed, Boolean(room.request && room.sent.has(JSON.stringify(room.request))));
        assert.equal(e.ended, room.state.ended); const {scenario, type, generation, ...data} = e; restores.push(data);
      } else if (e.type === 'reconnecting') {
        const {scenario, type, ...data} = e; reconnects.push(data);
      } else assert.fail('Unknown event type');
    }
    assert.equal(awaitingDecision, null);
    assert.equal(record.decisions, decisions.length); assert.equal(record.results[0].decisions, decisions.length);
    assert.equal(record.results[0].retries, retries);
    assert.equal(record.faults, injected.length); assert.equal(injected.length, record.scenario === 'double-drop' ? 2 : 1);
    assert.deepEqual(record.restores, restores); assert.deepEqual(record.reconnects, reconnects);
    assert.equal(restores.length, injected.length); assert.equal(reconnects.length, injected.length);
    assert.equal(record.connections, connections.size); assert.equal(record.connections, injected.length + 1);
    assert.deepEqual([...connections.keys()], Array.from({length: record.connections}, (_, i) => i + 1));
    for (const result of record.results) {
      const final = connections.get(record.connections).get(result.room).state;
      assert(final.ended); assert.equal(result.winner, final.winner); assert.equal(result.turns, final.turn);
    }
    if (record.scenario === 'lost-write') {
      assert.equal(sends.filter(s => !s.delivered).length, 1); assert.equal(sends[0].delivered, false);
      assert.equal(sends[1].generation, 2); assert.equal(sends[1].rqid, sends[0].rqid); assert.equal(sends[1].choice, sends[0].choice);
    } else assert(sends.every(s => s.delivered));
    if (record.scenario === 'accepted-choice') {
      assert(restores[0].committed); assert.equal(sends[0].generation, 1);
      assert(sends.find(s => s.generation === 2).rqid > sends[0].rqid);
    }
    if (record.scenario === 'before-choice') assert.equal(sends[0].generation, 2);
    if (record.scenario === 'double-drop') assert.equal(sends[0].generation, 3);
  }
  assert(sentchoices >= 1, 'Accepted-choice evidence missing');
  return {status: 'pass', games: CASES.length, faults, actor_inputs_and_actions_checked: actions, sentchoice_frames: sentchoices};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const directory = process.argv[2]; if (!directory) throw new Error('Usage: audit.mjs RUN_DIR');
  const read = name => JSON.parse(readFileSync(`${directory}/${name}.json`, 'utf8'));
  assert.deepEqual(read('teams'), developmentTeams(2, 'showdown-session-network-v1'), 'Development team admission');
  const result = {...auditTrace(read('report'), read('trace')), report_sha256: hash(`${directory}/report.json`),
    trace_sha256: hash(`${directory}/trace.json`), teams_sha256: hash(`${directory}/teams.json`)};
  writeFileSync(`${directory}/audit.json`, JSON.stringify(result, null, 2), {flag: 'wx'});
  console.log(JSON.stringify(result));
}
