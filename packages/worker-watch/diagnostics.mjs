import {performance} from 'node:perf_hooks';

let command = null, disabled = false, dropped = 0, sink;
try {
  sink = process.stderr;
  sink.on('error', () => { disabled = true; });
} catch { disabled = true; }
export function context(value) { command = value; }

/** Best effort only: a closed or backed-up diagnostic stream cannot fail a reply. */
export function mark(stage) {
  try {
    if (disabled || sink.destroyed || sink.writableLength > 65536) { dropped++; return; }
    const record = {schema_version: 'worker-watch-v1', stage, pid: process.pid,
      request_id: command?.id ?? null, op: command?.op ?? null,
      wall_ms: Date.now(), monotonic_ms: performance.now(), rss_bytes: process.memoryUsage.rss(),
      dropped_records: dropped};
    sink.write(`${JSON.stringify(record)}\n`, error => { if (error) disabled = true; });
  } catch { disabled = true; dropped++; }
}
