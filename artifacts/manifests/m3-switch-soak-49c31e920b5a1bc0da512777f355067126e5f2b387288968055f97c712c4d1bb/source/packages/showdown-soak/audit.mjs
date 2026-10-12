import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, existsSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';
import {Teams} from '../reference-node/engine.mjs';
import {sha256} from '../reference-node/provenance.mjs';
import {DEV_FORMAT, PublicBattle, legalChoices, toID} from '../showdown-bot/protocol.mjs';
import {chooseHeuristic} from '../showdown-bot/policy.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';
import {chooseOpponent} from '../showdown-panel/policy.mjs';
import {planFor, gameSpec, expectedCoverage} from './schedule.mjs';
import {freshTeams, teamReceiptLine} from './teams.mjs';
import {actorInput} from './wire.mjs';
import {formatSource} from './local-server.mjs';

const plain = x => JSON.parse(JSON.stringify(x));
const roles = ['tactical', 'baseline'];
const hash = p => sha256(readFileSync(p));

/** Stateful across rematches, isolated by player and by fresh connection. */
export class WireAudit {
  constructor(smoke, teams) {
    this.smoke = smoke; this.plan = planFor(smoke); this.teams = teams; this.index = 0;
    this.connections = {}; this.known = {}; this.roomsSeen = new Set(); this.lastTime = -Infinity;
    this.counts = {games: 0, sessions: 0, rematches: 0, faults: 0, tactical_connections: 0, baseline_connections: 0,
      tactical_p1: 0, tactical_p2: 0, scenarios: {normal: 0, 'before-choice': 0, 'lost-write': 0, 'accepted-choice': 0}};
    this.actions = {tactical: 0, baseline: 0}; this.retries = {tactical: 0, baseline: 0};
    this.latencies = {tactical: [], baseline: []}; this.peakRss = 0; this.peakHeap = 0;
  }
  check(game) {
    const spec = gameSpec(this.index, this.smoke); assert.deepEqual(game.spec, spec, 'Schedule mismatch');
    assert.equal(game.results.length, 2); const roomID = game.results[0].room;
    assert.equal(game.results[1].room, roomID); assert(!this.roomsSeen.has(roomID), 'Duplicate battle'); this.roomsSeen.add(roomID);
    if (!spec.within_session) {
      this.connections = Object.fromEntries(roles.map(r => [r, new Map()]));
      this.known = Object.fromEntries(roles.map(r => [r, new Map()])); this.counts.sessions++;
    } else this.counts.rematches++;
    const per = Object.fromEntries(roles.map(r => [r, {sends: [], decisions: [], ends: [], faults: [], restores: [], reconnects: [], uploads: [], commands: [], queries: [], proofs: 0, retries: 0, awaiting: null, held: null}]));
    const conn = (role, generation) => {
      const c = this.connections[role].get(generation); assert(c, 'Unknown connection'); return c;
    };
    for (const e of game.events) {
      assert(roles.includes(e.role)); assert(Number.isFinite(e.at_ms) && e.at_ms >= this.lastTime, 'Event clock order'); this.lastTime = e.at_ms;
      const role = e.role, p = per[role], connections = this.connections[role];
      if (e.type === 'connection') {
        assert.equal(e.generation, connections.size + 1, 'Connection sequence');
        connections.set(e.generation, {rooms: new Map(), identity: false, ready: false}); this.counts[`${role}_connections`]++;
      } else if (e.type === 'identity') {
        const c = conn(role, e.generation), parts = e.line.split('|');
        assert.equal(parts[1], 'updateuser');
        if (parts[3] === '1' && toID(parts[2].replace(/^[^A-Za-z0-9]+/, '').split('@')[0]) === toID(spec.names[role])) c.identity = true;
      } else if (e.type === 'ready') {
        const c = conn(role, e.generation); assert(c.identity && !c.ready, 'Login readiness'); c.ready = true;
        assert.equal(e.reconnects, e.generation - 1);
      } else if (e.type === 'receive') {
        const c = conn(role, e.generation); assert(c.identity, 'Battle before named login');
        for (const {room: id, line} of e.messages) {
          if (line === '|init|battle') {
            assert.equal(id, roomID); assert(!c.rooms.has(id), 'Repeated initialization');
            c.rooms.set(id, {state: role === 'tactical' ? new TacticalView() : new PublicBattle(), request: null, rqid: -1, sent: new Set(), undelivered: new Set()});
          }
          const room = c.rooms.get(id); assert(room, 'History missing'); room.state.ingest(line);
          if (line.startsWith('|soakteam|')) {
            assert(p.queries.some(q => q.generation === e.generation), 'Unrequested team receipt');
            assert.equal(line, teamReceiptLine(this.teams[spec[`${role}_team`]].team), 'Received simulator team'); p.proofs++;
          }
          if (line.startsWith('|bigerror|') || line.startsWith('|noinit|')) assert.fail('Server battle failure');
          if (line.startsWith('|request|')) {
            room.request = JSON.parse(line.slice(9)); room.request_at = e.at_ms;
            if (room.request) {
              assert(Number.isSafeInteger(room.request.rqid) && room.request.rqid >= room.rqid, 'Request order'); room.rqid = room.request.rqid;
              if (!room.request.wait) {
                const seat = role === 'tactical' ? spec.tactical_seat : spec.tactical_seat === 'p1' ? 'p2' : 'p1';
                assert.equal(room.request.side.name, spec.names[role]); assert.equal(room.request.side.id, seat, 'Policy seat');
                assert.equal(room.request.side.pokemon.length, 6); room.seat = seat;
              }
            }
          }
          if (line.startsWith('|sentchoice|')) {
            assert(room.request); const key = JSON.stringify(room.request); assert(!room.undelivered.has(key), 'Undelivered choice confirmed'); room.sent.add(key);
          }
          if (line.startsWith('|error|')) {
            assert(line.startsWith('|error|[Unavailable choice]'), 'Unexpected choice error'); room.request = null; p.retries++;
          }
          const known = this.known[role].get(id) ?? {rqid: -1, turn: 0};
          this.known[role].set(id, {rqid: Math.max(known.rqid, room.rqid), turn: Math.max(known.turn, room.state.turn)});
        }
      } else if (e.type === 'team') {
        assert(conn(role, e.generation).ready); assert.equal(e.packed, Teams.pack(this.teams[spec[`${role}_team`]].team), 'Uploaded team'); p.uploads.push(e);
      } else if (e.type === 'team-query') {
        assert.equal(e.room, roomID); assert(conn(role, e.generation).rooms.has(roomID));
        assert(!p.queries.some(q => q.generation === e.generation)); p.queries.push(e);
      } else if (e.type === 'command') {
        const c = conn(role, e.generation); assert(c.identity);
        const target = spec.names[role === 'tactical' ? 'baseline' : 'tactical'];
        assert([`|/challenge ${target}, ${DEV_FORMAT}`, `|/accept ${toID(target)}`, `|/join ${roomID}`].includes(e.data), 'Unexpected challenge/join');
        if (!e.data.startsWith('|/join ')) assert(p.uploads.some(u => u.generation === e.generation), 'Team must precede challenge/accept');
        p.commands.push(e.data);
      } else if (e.type === 'send') {
        assert.equal(p.awaiting, null, 'Unpaired send'); assert.equal(e.generation, connections.size, 'Stale socket send');
        const c = conn(role, e.generation); assert(c.ready);
        const match = /^(battle-[a-z0-9-]+)\|\/choose ([^|\r\n]+)\|(\d+)$/.exec(e.data); assert(match, 'Choice syntax');
        const [, id, choice, rqid] = match; assert.equal(id, roomID);
        const room = c.rooms.get(id); assert(room?.request && !room.request.wait && !room.state.ended);
        assert.equal(room.state.tier, DEV_FORMAT); assert.equal(room.state.gameType, 'singles');
        assert.equal(Number(rqid), room.request.rqid); const key = JSON.stringify(room.request);
        assert(!room.sent.has(key), 'Duplicate or accepted choice replayed'); assert(legalChoices(room.request).includes(choice));
        const chosen = role === 'tactical' ? chooseOpponent('tactical-v1', structuredClone(room.request), structuredClone(room.state)) : chooseHeuristic(structuredClone(room.request), structuredClone(room.state));
        assert.equal(choice, chosen, 'Frozen policy action'); room.sent.add(key);
        p.awaiting = {room: id, rqid: Number(rqid), turn: room.state.turn, choice, generation: e.generation,
          actor_input: plain(actorInput(room))};
        assert.equal(typeof e.delivered, 'boolean'); assert.equal(typeof e.held, 'boolean');
        if (!e.delivered) room.undelivered.add(key);
        if (e.held) {
          assert.equal(role, 'baseline'); assert.equal(spec.scenario, 'accepted-choice'); assert(!e.delivered && !p.held);
          assert.equal(per.tactical.restores.length, 0); p.held = {generation: e.generation, data: e.data, key};
        }
        if (role === 'baseline' && spec.scenario === 'accepted-choice' && e.delivered) {
          assert.equal(per.tactical.restores.length, 1, 'Baseline delivered before tactical restoration');
          assert(per.tactical.restores[0].committed);
        }
        p.sends.push({...p.awaiting, delivered: e.delivered, held: e.held});
        const latency = e.at_ms - room.request_at; assert(latency >= 0); this.latencies[role].push(latency);
      } else if (e.type === 'deliver') {
        assert.equal(role, 'baseline'); assert.equal(spec.scenario, 'accepted-choice'); assert.equal(per.tactical.restores.length, 1);
        assert(per.tactical.restores[0].committed && p.held && p.awaiting === null);
        assert.equal(e.data, p.held.data); assert.equal(e.generation, p.held.generation);
        conn(role, e.generation).rooms.get(roomID).undelivered.delete(p.held.key); p.held = null;
      } else if (e.type === 'decision') {
        assert(p.awaiting, 'Decision without send');
        const {role: ignoredRole, type, at_ms, session_generation, ...actual} = e;
        assert.equal(session_generation, 2 * e.generation - 1); assert.deepEqual(actual, p.awaiting, 'Actor input/action mismatch');
        p.decisions.push(e); p.awaiting = null;
      } else if (e.type === 'fault') {
        assert.equal(role, 'tactical'); assert.equal(e.generation, connections.size); p.faults.push(e);
      } else if (e.type === 'reconnecting') {
        assert.equal(role, 'tactical'); assert.equal(e.attempt, connections.size); assert.deepEqual(e.rooms, [roomID]);
        assert.equal(e.delayMs, Math.min(30000, this.plan.retry_delay_ms * 2 ** (e.attempt - 1))); p.reconnects.push(e);
      } else if (e.type === 'restored') {
        const room = conn(role, e.generation).rooms.get(e.room); assert(room && e.room === roomID);
        assert.equal(e.rqid, room.rqid); assert.equal(e.turn, room.state.turn); assert.equal(e.ended, room.state.ended);
        assert.equal(e.committed, Boolean(room.request && room.sent.has(JSON.stringify(room.request))));
        const known = this.known[role].get(roomID); assert(e.rqid >= known.rqid && e.turn >= known.turn, 'Recovery moved backwards'); p.restores.push(e);
      } else if (e.type === 'end') {
        const room = conn(role, e.generation).rooms.get(e.room); assert(room && room.state.ended);
        assert.equal(e.room, roomID); assert.equal(e.winner, room.state.winner); assert.equal(e.turns, room.state.turn); p.ends.push(e);
      } else assert.fail('Unknown event type');
    }
    for (const [i, role] of roles.entries()) {
      const p = per[role], result = game.results[i]; assert.equal(p.awaiting, null); assert(p.decisions.length > 0);
      assert.equal(p.held, null); assert(p.proofs >= 1 && p.proofs <= p.queries.length, 'Missing server team receipt');
      assert.equal(p.ends.length, 1); const {role: ignoredRole, type, at_ms, generation, ...terminal} = p.ends[0];
      assert.deepEqual(result, terminal); assert.equal(result.decisions, p.decisions.length); assert.equal(result.retries, p.retries);
      assert.equal(p.uploads.length, 1); const target = spec.names[role === 'tactical' ? 'baseline' : 'tactical'];
      const control = role === spec.challenger ? `|/challenge ${target}, ${DEV_FORMAT}` : `|/accept ${toID(target)}`;
      assert.equal(p.commands.filter(x => x === control).length, 1); assert(p.commands.every(x => x === control || x === `|/join ${roomID}`));
      const expectedFault = role === 'tactical' && spec.scenario !== 'normal' ? 1 : 0;
      assert.equal(p.faults.length, expectedFault); assert.equal(p.reconnects.length, expectedFault); assert.equal(p.restores.length, expectedFault);
      assert.equal(p.commands.filter(x => x === `|/join ${roomID}`).length, expectedFault);
      const observed = game.telemetry.roles[role], cs = this.connections[role];
      assert.deepEqual(observed, {rooms: cs.get(cs.size).rooms.size, finished: spec.within_session + 1, generations: cs.size, reconnects: cs.size - 1});
      assert(observed.rooms <= this.plan.games_per_session); this.actions[role] += p.decisions.length; this.retries[role] += p.retries;
    }
    assert.equal(game.results[0].winner, game.results[1].winner); assert.equal(game.results[0].turns, game.results[1].turns);
    const p = per.tactical;
    if (spec.scenario === 'lost-write') {
      assert.equal(p.sends.filter(s => !s.delivered).length, 1); assert.equal(p.sends[0].delivered, false);
      assert.equal(p.sends[1].generation, p.sends[0].generation + 1); assert.equal(p.sends[1].rqid, p.sends[0].rqid); assert.equal(p.sends[1].choice, p.sends[0].choice);
    } else assert(p.sends.every(s => s.delivered));
    assert(per.baseline.sends.every(s => s.delivered || s.held));
    if (['before-choice', 'lost-write'].includes(spec.scenario)) assert.equal(p.restores[0].committed, false);
    if (spec.scenario === 'accepted-choice') {
      assert(p.restores[0].committed); const next = p.sends.find(s => s.generation > p.sends[0].generation);
      assert(next && next.rqid > p.sends[0].rqid);
    }
    if (spec.scenario === 'before-choice') assert.equal(p.sends[0].generation, p.faults[0].generation + 1);
    for (const [key, value] of Object.entries(game.telemetry)) if (key !== 'roles') assert(Number.isFinite(value) && value > 0, 'Invalid telemetry');
    assert(game.telemetry.elapsed_ms >= this.lastTime); assert(game.telemetry.rss <= this.plan.max_client_rss_mib * 2 ** 20, 'RSS budget');
    this.peakRss = Math.max(this.peakRss, game.telemetry.rss); this.peakHeap = Math.max(this.peakHeap, game.telemetry.heap_used);
    this.counts.games++; this.counts.faults += p.faults.length; this.counts[`tactical_${spec.tactical_seat}`]++; this.counts.scenarios[spec.scenario]++; this.index++;
  }
  finish() {
    assert.deepEqual(this.counts, expectedCoverage(this.smoke), 'Complete coverage');
    const latency = Object.fromEntries(roles.map(role => {
      const x = this.latencies[role].sort((a, b) => a - b), at = q => x[Math.ceil(q * x.length) - 1];
      return [role, {median: at(.5), p95: at(.95), p99: at(.99), max: x.at(-1)}];
    }));
    return {status: 'pass', coverage: this.counts, decisions: this.actions, unavailable_retries: this.retries,
      peak_client_rss: this.peakRss, peak_client_heap: this.peakHeap, request_to_attempt_ms: latency};
  }
}

export function audit(directory, {write = true} = {}) {
  assert(!existsSync(`${directory}/failure.json`), 'Failed campaign');
  const read = name => JSON.parse(readFileSync(`${directory}/${name}.json`, 'utf8'));
  const registration = read('preregistration'), report = read('report'), teams = read('teams');
  assert.equal(typeof registration.smoke, 'boolean'); const plan = planFor(registration.smoke);
  assert.deepEqual(registration.plan, plan); assert.deepEqual(report.plan, plan); assert.equal(report.smoke, registration.smoke);
  assert.deepEqual(teams, freshTeams(registration.smoke), 'Official development admission');
  assert.equal(readFileSync(`${directory}/server/dist/config/custom-formats.js`, 'utf8'), formatSource(teams), 'Exact admitted server format');
  assert.equal(report.status, 'complete'); assert.deepEqual(report.failures, []); assert.deepEqual(report.expected, expectedCoverage(registration.smoke));
  for (const key of ['training_performed', 'public_server_used', 'final_evaluation_used']) assert.equal(report[key], false);
  assert.deepEqual(read('progress'), {completed: plan.games, total: plan.games}); assert.equal(report.records.length, plan.games);
  const checker = new WireAudit(registration.smoke, teams);
  for (const record of report.records) {
    const spec = gameSpec(checker.index, registration.smoke); assert.equal(record.index, spec.index); assert.equal(record.file, spec.file);
    const bytes = readFileSync(`${directory}/${record.file}`); assert.equal(sha256(bytes), record.sha256, 'Game hash');
    const game = JSON.parse(gunzipSync(bytes)); assert.deepEqual(record.results, game.results); assert.deepEqual(record.telemetry, game.telemetry);
    checker.check(game);
  }
  assert(report.elapsed_ms >= checker.lastTime);
  const result = {...checker.finish(), teams: teams.length, final_teams_used: 0,
    report_sha256: hash(`${directory}/report.json`), preregistration_sha256: hash(`${directory}/preregistration.json`), teams_sha256: hash(`${directory}/teams.json`)};
  if (write) writeFileSync(`${directory}/audit.json`, JSON.stringify(result, null, 2), {flag: 'wx'});
  return result;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw new Error('Usage: audit.mjs RUN');
  console.log(JSON.stringify(audit(process.argv[2], {write: !process.argv.includes('--check')})));
}
