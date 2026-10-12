import {cpSync, mkdirSync, symlinkSync, writeFileSync, readFileSync, openSync, closeSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {Dex} from '../reference-node/engine.mjs';
import {UPSTREAM, verifyBuild} from '../reference-node/provenance.mjs';

export async function startLocalServer(directory, port = 8000) {
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
  // Identical battle rules, with explicit development-partition teams instead of a server-generated team.
  const format = {name: '[Gen 9] Bot Baseline Development', mod: 'gen9', gameType: 'singles',
    searchShow: false, rated: false, ruleset: [...Dex.formats.get('gen9randombattle').ruleset]};
  writeFileSync(path.join(directory, 'dist/config/custom-formats.js'), `exports.Formats = [${JSON.stringify(format)}];\n`);
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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, port] = process.argv.slice(2);
  if (!directory) throw new Error('Usage: node packages/showdown-bot/local-server.mjs NEW_SERVER_DIR [PORT]');
  const server = await startLocalServer(directory, port === undefined ? 8000 : Number(port));
  console.log(`Local Showdown ready: ${server.url}`);
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, async () => { await server.stop(); });
}
