import assert from 'node:assert/strict';

export function assertJSON(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') { assert(Number.isFinite(value), 'Non-finite actor field'); return; }
  assert(typeof value === 'object' && (Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype), 'Non-JSON actor field');
  for (const entry of Object.values(value)) assertJSON(entry);
}

export function actorInput(room) {
  const input = {request: structuredClone(room.request), state: structuredClone(room.state)};
  assertJSON(input); return input;
}
