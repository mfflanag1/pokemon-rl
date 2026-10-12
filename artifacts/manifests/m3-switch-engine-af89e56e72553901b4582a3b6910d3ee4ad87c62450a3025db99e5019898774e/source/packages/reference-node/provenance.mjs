import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {existsSync, readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export const UPSTREAM = path.join(ROOT, 'vendor/pokemon-showdown');
export const pin = JSON.parse(readFileSync(path.join(ROOT, 'showdown.pin.json'), 'utf8'));
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

// R1 internal canonical JSON. The cross-language public contract is R2 work.
export function canonical(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  throw new TypeError('Only finite JSON values can be hashed');
}
export const hashJSON = value => sha256(canonical(value));
export const fileHash = name => sha256(readFileSync(name));
export const git = (...args) => execFileSync('git', ['-C', UPSTREAM, ...args], {encoding: 'utf8'}).trim();

export function treeHash(directory) {
  const entries = [];
  function visit(dir) {
    for (const entry of readdirSync(dir, {withFileTypes: true}).sort((a, b) => a.name < b.name ? -1 : 1)) {
      const full = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`Unexpected symlink in build: ${full}`);
      if (entry.isDirectory()) visit(full);
      else entries.push([path.relative(directory, full).split(path.sep).join('/'), fileHash(full)]);
    }
  }
  visit(directory);
  return hashJSON(entries);
}

export function verifySource() {
  if (git('rev-parse', 'HEAD') !== pin.commit) throw new Error('Showdown checkout does not match showdown.pin.json');
  if (git('status', '--porcelain', '--untracked-files=all')) throw new Error('Showdown source is dirty; refusing reference results');
  if (fileHash(path.join(UPSTREAM, 'package-lock.json')) !== pin.package_lock_sha256) throw new Error('Upstream lockfile hash mismatch');
  for (const extension of ['ts', 'js']) {
    if (existsSync(path.join(UPSTREAM, `config/custom-formats.${extension}`))) throw new Error('Custom formats are not allowed in the reference profile');
  }
}

export function verifyBuild() {
  verifySource();
  const receiptPath = path.join(UPSTREAM, '.reference-build.json');
  if (!existsSync(receiptPath)) throw new Error('No verified build receipt. Run npm run setup.');
  const receipt = JSON.parse(readFileSync(receiptPath, 'utf8'));
  if (receipt.commit !== pin.commit || receipt.lock_sha256 !== pin.package_lock_sha256) throw new Error('Stale build receipt');
  if (receipt.node !== process.version) throw new Error('Node version changed; rebuild with npm run setup');
  if (treeHash(path.join(UPSTREAM, 'dist')) !== receipt.dist_sha256) throw new Error('Compiled Showdown tree changed; rebuild with npm run setup');
  if (treeHash(path.join(UPSTREAM, 'node_modules/ts-chacha20')) !== receipt.prng_dependency_sha256) throw new Error('PRNG dependency changed; rebuild with npm run setup');
  return receipt;
}
