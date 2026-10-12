import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {ROOT, UPSTREAM, fileHash, sha256} from '../reference-node/provenance.mjs';

const require = createRequire(import.meta.url);
export const ts = require(path.join(ROOT, 'tools/mechanics-deps/node_modules/typescript'));
if (ts.version !== '5.9.3') throw new Error('R6 requires the separately locked TypeScript 5.9.3 parser');

export function parseSource(file) {
  const text = readFileSync(path.join(UPSTREAM, file), 'utf8');
  const ast = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  if (ast.parseDiagnostics.length) throw new Error(`Cannot parse pinned source ${file}`);
  const sourceHash = fileHash(path.join(UPSTREAM, file));
  function ref(node) {
    const start = node.getStart(ast), end = node.getEnd();
    return {file, file_sha256: sourceHash, start_line: ast.getLineAndCharacterOfPosition(start).line + 1,
      end_line: ast.getLineAndCharacterOfPosition(end).line + 1, start_byte: Buffer.byteLength(text.slice(0, start)),
      end_byte: Buffer.byteLength(text.slice(0, end)), fragment_sha256: sha256(text.slice(start, end))};
  }
  return {file, text, ast, ref};
}

export function walk(node, visit) { visit(node); ts.forEachChild(node, child => { walk(child, visit); }); }
export const keyName = node => node && (ts.isIdentifier(node) || ts.isStringLiteral(node) || ts.isNumericLiteral(node)) ? node.text : null;

export function tableEntries(source, variable) {
  let table;
  walk(source.ast, n => {
    if (ts.isVariableDeclaration(n) && keyName(n.name) === variable && n.initializer && ts.isObjectLiteralExpression(n.initializer)) table = n.initializer;
  });
  if (!table) throw new Error(`Missing ${variable} table`);
  return new Map(table.properties.map(n => {
    const key = keyName(n.name);
    if (!key || !ts.isPropertyAssignment(n) || !ts.isObjectLiteralExpression(n.initializer)) throw new Error(`Nonliteral ${variable} entry`);
    return [key, n];
  }));
}

export function methodNodes(source) {
  const out = new Map();
  walk(source.ast, n => { if (ts.isMethodDeclaration(n) && keyName(n.name)) out.set(keyName(n.name), n); });
  return out;
}

export function stringLiterals(node) {
  const out = new Set();
  walk(node, n => { if (ts.isStringLiteral(n) || ts.isNoSubstitutionTemplateLiteral(n)) out.add(n.text); });
  return [...out].sort();
}

const SINKS = {
  runMove: 'move-execution', useMove: 'move-execution', useMoveInner: 'move-execution', getActiveMove: 'active-move-construction',
  transformInto: 'transform-copy', formeChange: 'form-change', setSpecies: 'form-change', setAbility: 'ability-change',
  setItem: 'item-transfer', takeItem: 'item-transfer', useItem: 'item-consumption', eatItem: 'item-consumption',
  addVolatile: 'condition-install', addSideCondition: 'condition-install', addSlotCondition: 'condition-install',
  setWeather: 'condition-install', setTerrain: 'condition-install', addPseudoWeather: 'condition-install',
  runEvent: 'event-dispatch', singleEvent: 'event-dispatch', eachEvent: 'event-dispatch', fieldEvent: 'event-dispatch', residualEvent: 'event-dispatch',
  call: 'callback-invocation', apply: 'callback-invocation',
};
const STATE_FIELDS = new Set(['moveSlots', 'baseMoveSlots', 'moves', 'ability', 'baseAbility', 'item', 'lastItem', 'species', 'baseSpecies']);

export function pathways(source, root = source.ast) {
  const out = [];
  walk(root, n => {
    let kind, selector, args = [];
    if (ts.isCallExpression(n)) {
      const expr = n.expression;
      const name = ts.isPropertyAccessExpression(expr) ? keyName(expr.name) : ts.isIdentifier(expr) ? expr.text : null;
      kind = Object.hasOwn(SINKS, name) ? SINKS[name] : undefined;
      if (ts.isElementAccessExpression(expr)) kind = 'computed-call';
      const target = expr.getText(source.ast);
      if (name === 'get' && /\.(moves|abilities|items|species|conditions)\.get$/.test(target)) kind = 'dex-resolution';
      // Retain every remaining call under the enclosing source obligation.
      // Local aliases can themselves hold callbacks, so dropping ordinary
      // identifier/property calls would hide potential dynamic dispatch.
      if (!kind) kind = 'source-owned-call';
      selector = target;
      args = n.arguments.map(a => ({expression: a.getText(source.ast), literal: ts.isStringLiteral(a) ? a.text : null}));
    } else if (ts.isNewExpression(n)) {
      kind = 'source-owned-construction'; selector = n.expression.getText(source.ast);
      args = (n.arguments ?? []).map(a => ({expression: a.getText(source.ast), literal: ts.isStringLiteral(a) ? a.text : null}));
    } else if (ts.isBinaryExpression(n) && n.operatorToken.kind === ts.SyntaxKind.EqualsToken
        && ts.isPropertyAccessExpression(n.left) && STATE_FIELDS.has(keyName(n.left.name))) {
      kind = 'identity-state-write'; selector = n.left.getText(source.ast);
      args = [{expression: n.right.getText(source.ast), literal: ts.isStringLiteral(n.right) ? n.right.text : null}];
    } else return;
    out.push({kind, selector, arguments: args, source: source.ref(n)});
  });
  return out;
}

export function callbacks(source, entry) {
  const out = [];
  function visit(n, prefix) {
    if (ts.isObjectLiteralExpression(n)) {
      for (const p of n.properties) {
        const key = keyName(p.name);
        if (!key) throw new Error('Computed/spread effect property needs an explicit R6 rule');
        const route = [...prefix, key];
        if (ts.isMethodDeclaration(p) || (ts.isPropertyAssignment(p) && (ts.isArrowFunction(p.initializer) || ts.isFunctionExpression(p.initializer)))) {
          out.push({path: route.join('.'), source: source.ref(p), pathways: pathways(source, p)});
        } else if (ts.isPropertyAssignment(p)) visit(p.initializer, route);
      }
    } else if (ts.isArrayLiteralExpression(n)) n.elements.forEach((child, i) => visit(child, [...prefix, String(i)]));
  }
  visit(entry.initializer, []); return out;
}
