import {mkdirSync, writeFileSync} from 'node:fs';
import {once} from 'node:events';
import {Teams} from '../reference-node/engine.mjs';
import {verifyBuild} from '../reference-node/provenance.mjs';
import {baselineAdmission, developmentTeams} from './development.mjs';
import {ShowdownBot} from './client.mjs';
import {DEV_FORMAT, parseFrame} from './protocol.mjs';
import {startLocalServer} from './local-server.mjs';

const [directory, port = '18000'] = process.argv.slice(2);
if (!directory) throw new Error('Usage: node packages/showdown-bot/network-smoke.mjs NEW_OUTPUT_DIR [PORT]');
mkdirSync(directory);
const records = [], decisions = [], failures = [], requestOrder = [];
let server; const bots = [];
try {
  const provenance = baselineAdmission();
  const teams = developmentTeams(2, 'playable-baseline-network-v1');
  writeFileSync(`${directory}/teams.json`, JSON.stringify(teams));
  server = await startLocalServer(`${directory}/server`, Number(port));
  const landing = await fetch(`http://127.0.0.1:${port}/`);
  if (!landing.ok || !(await landing.text()).includes('.psim.us')) throw new Error('Stock web-client landing page unavailable');
  let pair;
  for (let round = 0; round < 3; round++) {
    if (round !== 1) pair = ['BaselineA', 'BaselineB'].map((name, i) => {
      const wire = new Map();
      const bot = new ShowdownBot({url: server.url, name, format: DEV_FORMAT, decisionDelayMs: 0,
        team: Teams.pack(teams[i].team), policy: i === (round === 2 ? 1 : 0) ? 'heuristic' : 'random',
        seed: [11, 22, 33, 44], accept: [i ? 'BaselineA' : 'BaselineB'], socketFactory: url => {
          const socket = new WebSocket(url);
          socket.addEventListener('message', event => {
            for (const {room, line} of parseFrame(event.data)) {
              if (!room) continue;
              const state = wire.get(room) ?? {turn: 0, tier: false, singles: false}; wire.set(room, state);
              if (line.startsWith('|turn|')) state.turn = Number(line.slice(6));
              if (line.startsWith('|tier|')) state.tier = true;
              if (line === '|gametype|singles') state.singles = true;
              if (line.startsWith('|request|') && line !== '|request|null') {
                const r = JSON.parse(line.slice(9));
                requestOrder.push({room, player: name, rqid: r.rqid, wait: Boolean(r.wait), ...state});
              }
            }
          });
          return socket;
        }});
      bot.on('decision', d => {
        if (d.turn !== wire.get(d.room)?.turn) failures.push({room: d.room, error: 'Decision used stale public turn'});
        decisions.push({player: name, ...d});
      });
      bot.on('failure', e => failures.push({round, player: name, error: e.message}));
      bots.push(bot); return bot;
    });
    if (round !== 1) await Promise.all(pair.map(bot => bot.connect()));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);
    const {signal} = controller;
    const finishes = pair.map(bot => Promise.race([
      once(bot, 'battleEnd', {signal}).then(([result]) => result),
      once(bot, 'failure', {signal}).then(([error]) => { throw error; }),
    ]));
    pair[0].challenge('BaselineB');
    let result;
    try { result = await Promise.all(finishes); } finally { clearTimeout(timer); controller.abort(); }
    if (result[0].winner !== result[1].winner || result[0].turns !== result[1].turns) throw new Error('Players disagree on terminal result');
    records.push(result);
    if (round !== 0) pair.forEach(bot => bot.close());
    // Round 1 repeats a challenge on the same connections; round 2 uses fresh transports.
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (failures.length) throw new Error('Network client failure');
  if (requestOrder.some(r => !r.wait && (!r.tier || !r.singles || r.turn < 1))) throw new Error('Request preceded initial public state');
  verifyBuild();
  writeFileSync(`${directory}/report.json`, JSON.stringify({status: 'complete', games: records.length, records, decisions, provenance, requestOrder,
    same_connection_rematch: true, fresh_connection_match: true,
    format: DEV_FORMAT, stock_rules: 'gen9randombattle', team_source: 'development partition, explicitly supplied',
    public_server_used: false, failures}, null, 2));
  console.log(JSON.stringify({status: 'complete', games: records.length, decisions: decisions.length, records}));
} catch (error) {
  writeFileSync(`${directory}/failure.json`, JSON.stringify({status: 'failed', error: error.stack, records, decisions, failures}, null, 2));
  throw error;
} finally { bots.forEach(bot => bot.close()); await server?.stop(); }
