import {writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runEpisode, defaultSpec} from '../packages/reference-node/runner.mjs';
import {contractManifest, episodeArtifacts} from '../packages/reference-node/contract-v1.mjs';
import {admissibleActions, actionMask, toProtocol} from '../packages/battle-contract/actions.mjs';

// Explicit fixture refresh, never executed automatically by the test suite.
const bundle = episodeArtifacts(await runEpisode(defaultSpec('r1-smoke-8')), contractManifest());
const rejectedIndices = bundle.decisions.flatMap((d, i) => d.payload.accepted ? [] : [i, i + 1]);
const selected = [...new Set([0, 1, bundle.decisions.findIndex(d => d.payload.phase === 'switch'), ...rejectedIndices])];
const request = structuredClone(bundle.views[0].payload.request);
const cases = [];
function add(name, req, origin = 'constructed') {
  const actions = admissibleActions(req);
  cases.push({name, origin, request: req, actions, mask: actionMask(req), commands: actions.map(a => toProtocol(a, req))});
}
for (const i of selected) add(`captured-${i}`, bundle.views[i].payload.request, 'natural-r1-smoke-8');
add('wait', {wait: true, side: request.side});
add('forced-pass', {forceSwitch: [false], side: request.side});
const noBench = structuredClone(request.side);
noBench.pokemon.slice(1).forEach(mon => { mon.condition = '0 fnt'; });
add('no-bench-pass', {forceSwitch: [true], side: noBench});
const revival = structuredClone(noBench);
revival.pokemon[0].reviving = true;
add('revival-targets', {forceSwitch: [true], side: revival});
const activeRevival = structuredClone(revival);
activeRevival.pokemon[0].condition = '0 fnt';
add('revival-active-target', {forceSwitch: [true], side: activeRevival});
const struggle = structuredClone(request);
struggle.active = [{moves: [{move: 'Struggle', id: 'struggle', target: 'randomNormal', disabled: false}], trapped: true}];
add('struggle', struggle);
const locked = structuredClone(struggle);
locked.active[0].moves = [{move: 'Recharge', id: 'recharge'}];
add('locked-recharge', locked);
const uncertain = structuredClone(request);
Object.assign(uncertain.active[0], {maybeDisabled: true, maybeLocked: true, maybeTrapped: true});
uncertain.active[0].moves[0].disabled = 'hidden'; // Truthy field means unavailable if actually sent.
uncertain.active[0].moves[1].pp = 0; // PP alone is not a second mask rule.
add('visible-flags', uncertain);

const canonicalVectors = [
  {wire: '{"b":-0,"a":[true,false,null,9007199254740991,-9007199254740991]}', canonical: '{"a":[true,false,null,9007199254740991,-9007199254740991],"b":0}'},
  {wire: '{"\\ud800\\udc00":2,"\\ue000":1,"é":"é\\n\\u0001","z":-0}', canonical: '{"z":0,"é":"é\\n\\u0001","\ue000":1,"\u{10000}":2}'},
  {wire: '{"__proto__":{"x":1},"x":"/\\b\\t\\n\\f\\r\\\"\\\\  "}', canonical: '{"__proto__":{"x":1},"x":"/\\b\\t\\n\\f\\r\\\"\\\\  "}'},
].map(v => ({...v, sha256: createHash('sha256').update(v.canonical).digest('hex')}));
const invalidWire = ['{"a":1,"a":2}', '{"a":1,"\\u0061":2}', '9007199254740992', '-9007199254740992', '1.0', '1e0', 'NaN', 'Infinity', '01', '[1,]', '{"x":1,}', '"\\ud800"', '"\\udfff"', '\ufeff{}', '{} junk'];
const golden = {
  fixture_version: 'r2-golden-1', canonical_vectors: canonicalVectors, invalid_wire: invalidWire,
  artifacts: [bundle.manifest, ...bundle.teams, ...selected.map(i => bundle.views[i]), ...selected.map(i => bundle.decisions[i]), bundle.episode],
  cases,
};
writeFileSync(new URL('../tests/fixtures/contract-golden.json', import.meta.url), JSON.stringify(golden, null, 2) + '\n');
console.log(JSON.stringify({artifacts: golden.artifacts.length, cases: cases.length, selected_decisions: selected}));
