import {existsSync, readFileSync, writeFileSync, unlinkSync} from 'node:fs';
import {hostname} from 'node:os';
import {canonical, parseJSON} from '../battle-contract/canonical.mjs';

// Same conservative local PID recovery protocol as the R3 corpus writer.
export function writerLock(directory) {
  const lock = `${directory}/.writer-lock.json`, guard = `${directory}/.writer-recovery.json`;
  const owner = canonical({pid: process.pid, host: hostname()});
  try { writeFileSync(lock, owner, {flag: 'wx'}); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    writeFileSync(guard, owner, {flag: 'wx'});
    try {
      if (existsSync(lock)) {
        const old = parseJSON(readFileSync(lock));
        if (old.host !== hostname() || !Number.isSafeInteger(old.pid) || old.pid <= 0) throw new Error('Unrecognized or remote writer');
        try { process.kill(old.pid, 0); throw new Error('Campaign writer is active'); }
        catch (error) { if (error.code !== 'ESRCH') throw error; }
        unlinkSync(lock);
      }
      writeFileSync(lock, owner, {flag: 'wx'});
    } finally { unlinkSync(guard); }
  }
  return () => unlinkSync(lock);
}
