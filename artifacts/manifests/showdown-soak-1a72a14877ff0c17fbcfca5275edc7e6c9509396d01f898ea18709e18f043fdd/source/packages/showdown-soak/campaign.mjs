import assert from 'node:assert/strict';
import {mkdirSync, writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {performance} from 'node:perf_hooks';
import {setTimeout as pause} from 'node:timers/promises';
import {gzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';
import {Teams} from '../reference-node/engine.mjs';
import {ROOT, verifyBuild, sha256} from '../reference-node/provenance.mjs';
import {DEV_FORMAT, parseFrame} from '../showdown-bot/protocol.mjs';
import {startLocalServer} from './local-server.mjs';
import {RecoveringBot} from '../showdown-session/client.mjs';
import {TacticalBot} from '../showdown-confirmation/client.mjs';
import {planFor, gameSpec, expectedCoverage} from './schedule.mjs';
import {freshTeams} from './teams.mjs';
import {actorInput} from './wire.mjs';

export async function campaign(directory, {smoke = false, port = 18030} = {}) {
  const plan = planFor(smoke), records = [], failures = [], started = performance.now();
  mkdirSync(directory);
  let server, bots, current, spec, faults, generations, held, released, queried, teamSeen;
  const save = (name, value) => writeFileSync(`${directory}/${name}`, JSON.stringify(value, null, 2));
  const event = (role, type, detail = {}) => current.events.push({role, type, at_ms: performance.now() - started, ...detail});
  const socketFactory = role => url => {
    assert.equal(url, server.url);
    const socket = new WebSocket(url), generation = ++generations[role], send = socket.send.bind(socket), session = spec.session;
    event(role, 'connection', {generation});
    let scheduled = false;
    const drop = reason => { faults++; event(role, 'fault', {generation, reason}); socket.close(); };
    socket.send = data => {
      if (data.includes('|/choose ')) {
        const lost = role === 'tactical' && spec.scenario === 'lost-write' && !faults;
        const hold = role === 'baseline' && spec.scenario === 'accepted-choice' && !released;
        event(role, 'send', {generation, data, delivered: !lost && !hold, held: hold});
        if (hold) { assert(!held, 'Multiple held baseline actions'); held = {send, data, generation, socket}; return; }
        if (lost) { drop('discarded-before-transport'); return; }
        send(data);
        if (role === 'tactical' && spec.scenario === 'accepted-choice' && !faults) drop('closed-after-action-write');
      } else {
        if (data.startsWith('|/utm ')) event(role, 'team', {generation, packed: data.slice(6)});
        if (data.startsWith('|/challenge ') || data.startsWith('|/accept ') || data.startsWith('|/join ')) event(role, 'command', {generation, data});
        if (data.endsWith('|/soakteam')) event(role, 'team-query', {generation, room: data.split('|')[0]});
        send(data);
      }
    };
    socket.addEventListener('message', e => {
      if (spec.session !== session || !bots) return;
      try {
      const parsed = parseFrame(e.data), messages = parsed.filter(m => m.room.startsWith('battle-'));
      for (const m of parsed) if (!m.room && m.line.startsWith('|updateuser|')) event(role, 'identity', {generation, line: m.line});
      if (!messages.length) return;
      event(role, 'receive', {generation, messages});
      if (messages.some(m => m.line.startsWith('|soakteam|'))) teamSeen[role] = true;
      if (!teamSeen[role] && !queried[role].has(generation)) {
        const first = messages.find(m => m.line.startsWith('|request|') && m.line !== '|request|null' && !JSON.parse(m.line.slice(9)).wait);
        if (first) { queried[role].add(generation); socket.send(`${first.room}|/soakteam`); }
      }
      if (role === 'tactical' && spec.scenario === 'before-choice' && !faults && !scheduled &&
          messages.some(m => m.line.startsWith('|request|') && m.line !== '|request|null' && !JSON.parse(m.line.slice(9)).wait)) {
        scheduled = true; queueMicrotask(() => drop('closed-before-scheduled-action'));
      }
      } catch (error) { bots[role].fail(error); }
    });
    return socket;
  };
  try {
    const admission = JSON.parse(execFileSync(`${ROOT}/.venv/bin/python`, [`${ROOT}/scripts/seal-showdown-soak.py`, '--admission'],
      {cwd: ROOT, encoding: 'utf8', maxBuffer: 4_000_000}));
    const teams = freshTeams(smoke);
    save('preregistration.json', {plan, smoke, admission, frozen_at: new Date().toISOString()}); save('teams.json', teams);
    server = await startLocalServer(`${directory}/server`, port, teams);
    for (let index = 0; index < plan.games; index++) {
      spec = gameSpec(index, smoke); current = {spec, events: []}; faults = 0; held = null; released = false;
      queried = {tactical: new Set(), baseline: new Set()}; teamSeen = {tactical: false, baseline: false};
      if (!spec.within_session) {
        generations = {tactical: 0, baseline: 0};
        bots = Object.fromEntries(['tactical', 'baseline'].map(role => [role, new (role === 'tactical' ? TacticalBot : RecoveringBot)({
          url: server.url, name: spec.names[role], format: DEV_FORMAT, accept: [spec.names[role === 'tactical' ? 'baseline' : 'tactical']],
          decisionDelayMs: plan.decision_delay_ms, retryDelayMs: plan.retry_delay_ms, socketFactory: socketFactory(role),
        })]));
        for (const [role, bot] of Object.entries(bots)) {
          bot.on('reconnecting', detail => event(role, 'reconnecting', detail));
          bot.on('ready', detail => event(role, 'ready', {...detail, generation: generations[role]}));
          bot.on('restored', detail => {
            event(role, 'restored', {generation: generations[role], ...detail});
            if (role === 'tactical' && spec.scenario === 'accepted-choice') {
              released = true;
              if (held) {
                assert.equal(held.socket.readyState, 1); event('baseline', 'deliver', {generation: held.generation, data: held.data});
                held.send(held.data); held = null;
              }
            }
          });
          bot.on('decision', detail => {
            const room = bot.connection.rooms.get(detail.room);
            event(role, 'decision', {...detail, generation: generations[role], session_generation: detail.generation,
              actor_input: actorInput(room)});
          });
          bot.on('battleEnd', detail => event(role, 'end', {...detail, generation: generations[role]}));
          bot.on('failure', error => { failures.push({index: spec.index, role, error: error.message}); event(role, 'failure', {error: error.message}); });
        }
        await Promise.all(Object.values(bots).map(bot => bot.connect()));
      }
      for (const role of ['tactical', 'baseline']) {
        const packed = Teams.pack(teams[spec[`${role}_team`]].team);
        bots[role].options.team = packed; bots[role].connection.options.team = packed;
      }
      let timedOut = false;
      const controller = new AbortController(), timer = setTimeout(() => { timedOut = true; controller.abort(); }, plan.game_timeout_ms);
      const finish = Object.values(bots).map(bot => Promise.race([
        once(bot, 'battleEnd', {signal: controller.signal}).then(([result]) => result),
        once(bot, 'failure', {signal: controller.signal}).then(([error]) => { throw error; }),
      ]));
      let results;
      try {
        bots[spec.challenger].challenge(spec.names[spec.challenger === 'tactical' ? 'baseline' : 'tactical']);
        results = await Promise.all(finish);
      } catch (error) {
        controller.abort(); await Promise.allSettled(finish);
        throw timedOut ? new Error(`Game ${index} exceeded ${plan.game_timeout_ms} ms timeout`, {cause: error}) : error;
      } finally { clearTimeout(timer); controller.abort(); }
      assert.equal(results[0].room, results[1].room); assert.equal(results[0].winner, results[1].winner); assert.equal(results[0].turns, results[1].turns);
      assert.equal(faults, Number(spec.scenario !== 'normal'), 'Planned fault did not execute');
      const last = spec.within_session === plan.games_per_session - 1;
      if (last) Object.values(bots).forEach(bot => bot.close());
      await pause(plan.intergame_delay_ms);
      assert.equal(failures.length, 0, 'Client failure');
      assert.equal(held, null, 'Unreleased baseline action');
      assert(teamSeen.tactical && teamSeen.baseline, 'Missing server team receipts');
      for (const role of ['tactical', 'baseline']) for (const type of ['fault', 'restored', 'reconnecting']) {
        assert.equal(current.events.filter(e => e.role === role && e.type === type).length, role === 'tactical' ? faults : 0, 'Unplanned or incomplete recovery');
      }
      if (spec.scenario === 'accepted-choice') assert(current.events.find(e => e.type === 'restored').committed, 'Accepted action was not confirmed');
      const memory = process.memoryUsage();
      assert(memory.rss <= plan.max_client_rss_mib * 2 ** 20, 'Client RSS budget exceeded');
      current.results = results; current.telemetry = {rss: memory.rss, heap_used: memory.heapUsed,
        elapsed_ms: performance.now() - started, roles: Object.fromEntries(Object.entries(bots).map(([role, bot]) => [role,
          {rooms: bot.connection.rooms.size, finished: bot.finished.size, generations: generations[role], reconnects: bot.reconnects}]))};
      const bytes = gzipSync(JSON.stringify(current)); writeFileSync(`${directory}/${spec.file}`, bytes, {flag: 'wx'});
      records.push({index, file: spec.file, sha256: sha256(bytes), results, telemetry: current.telemetry});
      save('progress.json', {completed: records.length, total: plan.games});
      if (last) bots = null;
    }
    verifyBuild();
    save('report.json', {status: 'complete', plan, smoke, expected: expectedCoverage(smoke), records, failures,
      elapsed_ms: performance.now() - started, public_server_used: false, training_performed: false, final_evaluation_used: false});
    console.log(JSON.stringify({status: 'complete', games: records.length}));
  } catch (error) {
    save('failure.json', {status: 'failed', error: error.stack, completed: records.length, failures});
    if (current) writeFileSync(`${directory}/failure-game.json.gz`, gzipSync(JSON.stringify(current)));
    throw error;
  } finally { if (bots) Object.values(bots).forEach(bot => bot.close()); await server?.stop(); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, ...flags] = process.argv.slice(2);
  if (!directory || flags.some(flag => flag !== '--smoke')) throw new Error('Usage: campaign.mjs NEW_RUN [--smoke]');
  await campaign(directory, {smoke: flags.includes('--smoke')});
}
