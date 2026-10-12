import {mkdirSync, writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {once} from 'node:events';
import {setTimeout as pause} from 'node:timers/promises';
import {Teams} from '../reference-node/engine.mjs';
import {ROOT, verifyBuild} from '../reference-node/provenance.mjs';
import {developmentTeams} from '../showdown-bot/development.mjs';
import {ShowdownBot} from '../showdown-bot/client.mjs';
import {DEV_FORMAT, parseFrame} from '../showdown-bot/protocol.mjs';
import {startLocalServer} from '../showdown-bot/local-server.mjs';
import {RecoveringBot} from './client.mjs';

export const CASES = ['before-choice', 'lost-write', 'accepted-choice', 'next-turn', 'forced-switch', 'double-drop'];
const [directory, port = '18010'] = process.argv.slice(2);
if (!directory) throw new Error('Usage: node packages/showdown-session/network.mjs NEW_OUTPUT_DIR [PORT]');
mkdirSync(directory);
const provenance = JSON.parse(execFileSync(`${ROOT}/.venv/bin/python`, [`${ROOT}/scripts/seal-showdown-session.py`, '--admission'],
  {cwd: ROOT, encoding: 'utf8', maxBuffer: 4_000_000}));
const teams = developmentTeams(2, 'showdown-session-network-v1');
writeFileSync(`${directory}/admission.json`, JSON.stringify(provenance));
writeFileSync(`${directory}/teams.json`, JSON.stringify(teams));
const records = [], failures = [], trace = [];
let server, pair = [];
try {
  server = await startLocalServer(`${directory}/server`, Number(port));
  for (const scenario of CASES) {
    let faults = 0, socketNumber = 0;
    const restores = [], reconnects = [], decisions = [];
    const entry = (type, detail) => trace.push({scenario, type, ...detail});
    const first = new RecoveringBot({url: server.url, name: 'RecoveryA', format: DEV_FORMAT,
      accept: ['RecoveryB'], team: Teams.pack(teams[0].team), decisionDelayMs: 30, retryDelayMs: 80,
      socketFactory: url => {
        const socket = new WebSocket(url), generation = ++socketNumber;
        let turn = 0;
        const send = socket.send.bind(socket);
        const drop = reason => {
          faults++; entry('fault', {generation, reason}); socket.close();
        };
        socket.send = data => {
          if (data.includes('|/choose ')) {
            const lost = scenario === 'lost-write' && !faults;
            entry('send', {generation, data, delivered: !lost});
            if (lost) { drop('outbound action discarded before transport'); return; }
            send(data);
            if (scenario === 'accepted-choice' && !faults) drop('transport closed after action write');
          } else send(data);
        };
        socket.addEventListener('message', event => {
          const messages = parseFrame(event.data).filter(x => x.room.startsWith('battle-'));
          if (!messages.length) return;
          entry('receive', {generation, messages});
          for (const {line} of messages) {
            if (line.startsWith('|turn|')) turn = Number(line.slice(6));
            if (!line.startsWith('|request|') || line === '|request|null') continue;
            const r = JSON.parse(line.slice(9));
            if (r.wait) continue;
            if (!faults && (scenario === 'before-choice' || scenario === 'double-drop' ||
                scenario === 'next-turn' && turn >= 2 || scenario === 'forced-switch' && r.forceSwitch?.[0])) {
              // Let the client observe the request before the real transport closes.
              queueMicrotask(() => drop('transport closed before scheduled decision'));
            }
          }
        });
        return socket;
      }});
    const second = new ShowdownBot({url: server.url, name: 'RecoveryB', format: DEV_FORMAT,
      accept: ['RecoveryA'], team: Teams.pack(teams[1].team), policy: 'random', seed: [91, 82, 73, 64],
      decisionDelayMs: scenario === 'accepted-choice' ? 800 : 40});
    pair = [first, second];
    first.on('reconnecting', event => { reconnects.push(event); entry('reconnecting', event); });
    first.on('restored', event => {
      restores.push(event); entry('restored', {generation: socketNumber, ...event});
      second.options.decisionDelayMs = 40;
      if (scenario === 'double-drop' && faults === 1) {
        faults++; entry('fault', {generation: socketNumber, reason: 'second transport loss during restoration'});
        first.connection.socket.close();
      }
    });
    first.on('decision', event => {
      const room = first.connection.rooms.get(event.room);
      decisions.push(event); entry('decision', {...event, generation: socketNumber, session_generation: event.generation,
        actor_input: {request: structuredClone(room.request), state: structuredClone(room.state)}});
    });
    for (const bot of pair) bot.on('failure', error => failures.push({scenario, error: error.message}));
    await Promise.all(pair.map(bot => bot.connect()));
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 120_000);
    const finish = pair.map(bot => Promise.race([
      once(bot, 'battleEnd', {signal: controller.signal}).then(([result]) => result),
      once(bot, 'failure', {signal: controller.signal}).then(([error]) => { throw error; }),
    ]));
    first.challenge('RecoveryB');
    let results;
    try { results = await Promise.all(finish); } finally { clearTimeout(timer); controller.abort(); }
    if (failures.length || results[0].winner !== results[1].winner || results[0].turns !== results[1].turns) throw new Error('Client failure or terminal disagreement');
    if (faults !== (scenario === 'double-drop' ? 2 : 1) || restores.length !== faults || reconnects.length !== faults) throw new Error('Fault scenario did not execute and restore exactly');
    if (scenario === 'accepted-choice' && !restores[0].committed) throw new Error('Server did not confirm accepted pending action');
    records.push({scenario, faults, connections: socketNumber, results, restores, reconnects, decisions: decisions.length});
    writeFileSync(`${directory}/progress.json`, JSON.stringify(records, null, 2));
    pair.forEach(bot => bot.close()); pair = []; await pause(150);
  }
  verifyBuild();
  writeFileSync(`${directory}/report.json`, JSON.stringify({status: 'complete', cases: CASES, records, failures,
    format: DEV_FORMAT, team_source: 'two development-partition teams, explicitly supplied',
    public_server_used: false, training_performed: false}, null, 2));
  console.log(JSON.stringify({status: 'complete', games: records.length, faults: records.reduce((n, r) => n + r.faults, 0)}));
} catch (error) {
  writeFileSync(`${directory}/failure.json`, JSON.stringify({status: 'failed', error: error.stack, records, failures}, null, 2));
  throw error;
} finally {
  pair.forEach(bot => bot.close()); await server?.stop();
  writeFileSync(`${directory}/trace.json`, JSON.stringify(trace));
}
