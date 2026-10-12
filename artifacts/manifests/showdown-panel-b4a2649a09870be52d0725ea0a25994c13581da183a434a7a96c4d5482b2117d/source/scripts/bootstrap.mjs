import {execFileSync} from 'node:child_process';
import {existsSync, mkdirSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {ROOT, UPSTREAM, pin, treeHash, verifySource} from '../packages/reference-node/provenance.mjs';

const run = (command, args, cwd = UPSTREAM) => execFileSync(command, args, {cwd, stdio: 'inherit'});
if (!Set.prototype.intersection) throw new Error('Node >=22.18 is required; see .node-version for the tested runtime');
if (!existsSync(UPSTREAM)) {
  mkdirSync(path.dirname(UPSTREAM), {recursive: true});
  run('git', ['clone', '--depth', '1', '--no-checkout', pin.repository, UPSTREAM], ROOT);
  run('git', ['fetch', '--depth', '1', 'origin', pin.commit]);
  run('git', ['checkout', '--detach', pin.commit]);
}
// Keep generated receipt out of upstream status without changing its tracked files.
const excludePath = path.join(UPSTREAM, '.git/info/exclude');
const exclude = readFileSync(excludePath, 'utf8');
if (!exclude.split('\n').includes('/.reference-build.json')) writeFileSync(excludePath, `${exclude}\n/.reference-build.json\n`);
verifySource();
run('npm', ['ci', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund']);
// The upstream compiler does not remove stale outputs. Only remove its generated tree.
rmSync(path.join(UPSTREAM, 'dist'), {recursive: true, force: true});
run(process.execPath, ['build']);
verifySource();
const receipt = {
  commit: pin.commit,
  lock_sha256: pin.package_lock_sha256,
  node: process.version,
  npm: execFileSync('npm', ['--version'], {encoding: 'utf8'}).trim(),
  platform: process.platform,
  arch: process.arch,
  dist_sha256: treeHash(path.join(UPSTREAM, 'dist')),
  prng_dependency_sha256: treeHash(path.join(UPSTREAM, 'node_modules/ts-chacha20')),
  build_command: 'node build',
  dependency_install: 'npm ci --omit=dev --ignore-scripts --no-audit --no-fund',
};
writeFileSync(path.join(UPSTREAM, '.reference-build.json'), `${JSON.stringify(receipt, null, 2)}\n`);
console.log(JSON.stringify({status: 'built', ...receipt}, null, 2));
