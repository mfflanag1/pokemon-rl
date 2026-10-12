import {cpSync, mkdirSync, symlinkSync, writeFileSync, readFileSync, openSync, closeSync} from 'node:fs';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {developmentFormat} from './teams.mjs';
import {Teams} from '../reference-node/engine.mjs';
import {UPSTREAM, verifyBuild} from '../reference-node/provenance.mjs';

export function formatSource(teams) {
  if (!Array.isArray(teams) || !teams.length) throw new Error('Admitted teams are required');
  const packed = teams.map(entry => Teams.pack(Teams.unpack(Teams.pack(entry.team))));
  return `const allowed = new Set(${JSON.stringify(packed)});\n` +
    `const format = ${JSON.stringify(developmentFormat())};\n` +
    `format.validateTeam = team => Array.isArray(team) && team.length === 6 && allowed.has(Teams.pack(team)) ? null : ['Team is outside the admitted official development pool'];\n` +
    `exports.Formats = [format];\n`;
}

export async function startLocalServer(directory, port, teams) {
  if (!Number.isSafeInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid local port');
  verifyBuild();
  directory = path.resolve(directory);
  mkdirSync(directory);
  // The pinned oracle remains untouched. All writable server paths live in this disposable copy.
  for (const name of ['dist', 'config', 'data', 'translations']) cpSync(path.join(UPSTREAM, name), path.join(directory, name), {recursive: true});
  cpSync(path.join(UPSTREAM, 'server/static'), path.join(directory, 'server/static'), {recursive: true});
  symlinkSync(path.join(UPSTREAM, 'node_modules'), path.join(directory, 'node_modules'), 'dir');
  writeFileSync(path.join(directory, 'package.json'), '{"type":"commonjs"}\n');
  const settings = {port, bindaddress: '127.0.0.1', subprocesses: 0, noguestsecurity: true, noipchecks: true, nothrottle: true,
    watchconfig: false, repl: false, crashguard: false, usesqlite: false, wsdeflate: null, trustedproxies: [],
    loginserver: 'http://127.0.0.1:9/', serverid: 'local-baseline', logladderip: false, reportjoins: false, potd: '',
    consoleips: [], backdoor: false, nofswriting: true};
  writeFileSync(path.join(directory, 'config/config.js'), `Object.assign(exports, ${JSON.stringify(settings, null, 2)});\n`);
  // Random formats bypass constructed-team legality. Admit only the exact
  // official development pool while preserving all battle rules and set values.
  writeFileSync(path.join(directory, 'dist/config/custom-formats.js'), formatSource(teams));
  cpSync(new URL('./team-receipt.cjs', import.meta.url), path.join(directory, 'dist/server/chat-plugins/soak-team.js'));
  const fd = openSync(path.join(directory, 'server.log'), 'wx');
  // Explicit positional port avoids cloud-env's default IP overriding the loopback bind.
  const child = spawn(process.execPath, ['dist/server/index.js', String(port)], {cwd: directory, stdio: ['ignore', fd, fd],
    env: {PATH: process.env.PATH, TMPDIR: process.env.TMPDIR ?? '/tmp'}});
  closeSync(fd);
  const url = `ws://127.0.0.1:${port}/showdown/websocket`;
  const stop = async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill('SIGTERM');
    await new Promise(resolve => { const timer = setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 3000); child.once('exit', () => { clearTimeout(timer); resolve(); }); });
  };
  try {
    let exited = false, spawnError = null;
    child.once('error', error => { spawnError = error; });
    child.once('exit', () => { exited = true; });
    for (let i = 0; i < 100; i++) {
      if (spawnError) throw spawnError;
      if (exited) throw new Error(`Local server exited; inspect ${directory}/server.log`);
      try {
        if (readFileSync(path.join(directory, 'server.log'), 'utf8').includes(`now listening on 127.0.0.1:${port}`)) {
          const response = await fetch(`http://127.0.0.1:${port}/showdown/info`, {signal: AbortSignal.timeout(300)});
          if (response.ok && !exited) return {child, url, directory, stop};
        }
      } catch {}
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`Local server did not start; inspect ${directory}/server.log`);
  } catch (error) { await stop(); throw error; }
}
