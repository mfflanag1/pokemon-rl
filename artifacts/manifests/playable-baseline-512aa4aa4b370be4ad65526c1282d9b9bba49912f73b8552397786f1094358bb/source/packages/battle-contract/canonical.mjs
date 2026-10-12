import {createHash} from 'node:crypto';

function string(value) {
  if (!value.isWellFormed()) throw new TypeError('Lone Unicode surrogate');
  return JSON.stringify(value);
}

export function canonical(value, depth = 0) {
  if (value === null || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'string') return string(value);
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new TypeError('Expected a safe integer');
    return String(value); // Includes -0 -> 0.
  }
  if (!value || typeof value !== 'object') throw new TypeError('Non-JSON value');
  if (depth >= 100) throw new TypeError('JSON nesting exceeds 100 containers');
  if (Array.isArray(value)) return `[${Array.from(value, v => canonical(v, depth + 1)).join(',')}]`;
  if (![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new TypeError('Non-JSON object');
  const keys = Object.keys(value);
  if (Reflect.ownKeys(value).length !== keys.length || keys.some(k => !('value' in Object.getOwnPropertyDescriptor(value, k)))) {
    throw new TypeError('Non-JSON object properties');
  }
  keys.forEach(string);
  keys.sort((a, b) => Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8')));
  return `{${keys.map(k => `${string(k)}:${canonical(value[k], depth + 1)}`).join(',')}}`;
}

export const hashJSON = value => createHash('sha256').update(canonical(value), 'utf8').digest('hex');

// JSON.parse alone loses duplicate keys and integer lexemes before validation.
export function parseJSON(input) {
  const text = typeof input === 'string' ? input : new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(input);
  let at = 0;
  const integerToken = /-?(0|[1-9][0-9]*)/y;
  const fail = () => { throw new SyntaxError(`Invalid contract JSON near offset ${at}`); };
  const whitespace = () => { while (/[\x20\t\r\n]/.test(text[at] ?? '\0')) at++; };
  function readString() {
    const start = at++;
    while (at < text.length) {
      const char = text[at++];
      if (char === '\\') { at++; continue; }
      if (char === '"') {
        const value = JSON.parse(text.slice(start, at));
        string(value);
        return value;
      }
    }
    return fail();
  }
  function read(depth) {
    whitespace();
    const char = text[at];
    if (char === '"') return readString();
    if (char === '[' || char === '{') {
      if (depth >= 100) return fail();
      const object = char === '{', close = object ? '}' : ']';
      const result = object ? {} : [];
      at++; whitespace();
      if (text[at] === close) { at++; return result; }
      while (true) {
        whitespace();
        if (object) {
          if (text[at] !== '"') return fail();
          const key = readString();
          if (Object.hasOwn(result, key)) throw new SyntaxError('Duplicate object key');
          whitespace();
          if (text[at++] !== ':') return fail();
          Object.defineProperty(result, key, {value: read(depth + 1), enumerable: true, writable: true, configurable: true});
        } else result.push(read(depth + 1));
        whitespace();
        const separator = text[at++];
        if (separator === close) return result;
        if (separator !== ',') return fail();
      }
    }
    for (const [literal, value] of [['true', true], ['false', false], ['null', null]]) {
      if (text.startsWith(literal, at)) { at += literal.length; return value; }
    }
    integerToken.lastIndex = at;
    const match = integerToken.exec(text);
    if (!match) return fail();
    at += match[0].length;
    const number = Number(match[0]);
    if (!Number.isSafeInteger(number)) throw new SyntaxError('Unsafe integer');
    return number;
  }
  const result = read(0);
  whitespace();
  if (at !== text.length) fail();
  return result;
}
