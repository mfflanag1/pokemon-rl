import test from 'node:test';
import assert from 'node:assert/strict';
import {TerminalRosterView} from '../../packages/terminal-matchup/view.mjs';

const roster = () => ['Zoroark','Mimikyu','Pikachu','Snorlax','Vaporeon','Espeon'].map((id,i) => ({
  ident:`p1: ${id}`,details:`${id}, L80`,condition:i===0?'235/235':'201/201',active:i===0,
  moves:['splash'],ability:i===0?'Illusion':'No Ability',item:'',stats:{},teraType:'Water'}));
const req = mons => ({wait:true,side:{id:'p1',pokemon:mons}});
function fixture() {
  const v=new TerminalRosterView('p1');
  v.ingest('|switch|p1a: Mimikyu|Mimikyu, L80|235/235');
  v.acceptRequest(req(roster()));
  return v;
}
const terminal = v => v.withOwnTypes(v.observe(null,99));
const by = (p,id) => p.own.find(m=>m.ident===id);

test('captured terminal alias shape leaves exactly the real Zoroark active', () => {
  const v=fixture(), p=terminal(v);
  assert.deepEqual(p.own.filter(m=>m.active).map(m=>m.ident),['Zoroark']);
  assert.equal(by(p,'Mimikyu').condition,'201/201');
  assert.equal(by(p,'Zoroark').condition,'235/235');
  assert(p.mask.every(m=>!m));
});

test('active alias HP, status, boosts and Tera affect actual owner only', () => {
  const v=fixture();
  for(const line of ['|-damage|p1a: Mimikyu|111/235','|-status|p1a: Mimikyu|psn',
    '|-boost|p1a: Mimikyu|spa|2','|-terastallize|p1a: Mimikyu|Water'])v.ingest(line);
  const p=terminal(v),z=by(p,'Zoroark'),m=by(p,'Mimikyu');
  assert.equal(z.condition,'111/235 psn');assert.deepEqual(z.boosts,{spa:2});assert.equal(z.tera,'Water');
  assert.equal(m.condition,'201/201');assert.deepEqual(m.boosts,{});assert.equal(m.tera,null);
});

test('Illusion reveal retains boosts then faint changes only the true member', () => {
  const v=fixture();v.ingest('|-boost|p1a: Mimikyu|spa|2');
  v.ingest('|replace|p1a: Zoroark|Zoroark, L80');
  v.ingest('|faint|p1a: Zoroark');
  const p=terminal(v);assert.equal(by(p,'Zoroark').condition,'0 fnt');
  assert.deepEqual(by(p,'Zoroark').boosts,{spa:2});assert.equal(by(p,'Mimikyu').condition,'201/201');
});

test('bare bench identity remains distinct from active disguise', () => {
  const v=fixture();v.ingest('|-status|p1: Mimikyu|brn');v.ingest('|-curestatus|p1: Mimikyu');
  v.ingest('|-heal|p1: Mimikyu|180/201');v.ingest('|-damage|p1a: Mimikyu|100/235 psn');
  const p=terminal(v);assert.equal(by(p,'Mimikyu').condition,'180/201');assert.equal(by(p,'Zoroark').condition,'100/235 psn');
});

test('submitted switch resolves terminal entrant without another private request', () => {
  const v=fixture();v.expectSwitch('Mimikyu');v.ingest('|switch|p1a: Mimikyu|Mimikyu, L80|201/201');
  v.ingest('|-damage|p1a: Mimikyu|180/201');
  const p=terminal(v);assert.deepEqual(p.own.filter(m=>m.active).map(m=>m.ident),['Mimikyu']);
  assert.equal(by(p,'Mimikyu').condition,'180/201');assert.equal(by(p,'Zoroark').condition,'235/235');
});

test('ambiguous dragged appearance waits for private request, and fails without it', () => {
  const v=fixture();v.expectSwitch('Snorlax');v.ingest('|switch|p1a: Snorlax|Snorlax, L80|201/201');
  v.ingest('|drag|p1a: Mimikyu|Mimikyu, L80|201/201');
  assert.throws(()=>terminal(v),/Unresolved own appearance/);
  const mons=roster();mons.forEach(m=>m.active=m.ident==='p1: Mimikyu');v.acceptRequest(req(mons));
  assert.deepEqual(terminal(v).own.filter(m=>m.active).map(m=>m.ident),['Mimikyu']);
});

test('private refresh supersedes stale HP/status while retaining correct boost identity', () => {
  const v=fixture();v.ingest('|-boost|p1a: Mimikyu|spa|2');v.ingest('|-damage|p1a: Mimikyu|10/235 psn');
  const mons=roster();mons[0].condition='100/235';v.acceptRequest(req(mons));
  const p=terminal(v);assert.equal(by(p,'Zoroark').condition,'100/235');assert.deepEqual(by(p,'Zoroark').boosts,{spa:2});
});

test('foreign requests and contradictory own identity are rejected', () => {
  const v=fixture();assert.throws(()=>v.acceptRequest({side:{id:'p2',pokemon:roster()}}),/Foreign/);
  assert.throws(()=>v.ingest('|replace|p1a: Pikachu|Pikachu, L80'),/identity changed/);
});

test('ambiguous entry cannot be overwritten by a later switch before resolution', () => {
  const v=fixture();v.expectSwitch('Snorlax');v.ingest('|switch|p1a: Snorlax|Snorlax, L80|201/201');
  v.ingest('|drag|p1a: Mimikyu|Mimikyu, L80|201/201');
  v.ingest('|-damage|p1a: Mimikyu|1/201');v.expectSwitch('Pikachu');
  assert.throws(()=>v.ingest('|switch|p1a: Pikachu|Pikachu, L80|201/201'),/Unresolved outgoing/);
});

test('a switch choice that never executes does not change terminal roster', () => {
  const v=fixture();v.expectSwitch('Pikachu');
  const p=terminal(v);assert.deepEqual(p.own.filter(m=>m.active).map(m=>m.ident),['Zoroark']);
});

test('bench healing survives a subsequent Illusion reveal', () => {
  const v=fixture();v.ingest('|-heal|p1: Mimikyu|190/201');v.ingest('|replace|p1a: Zoroark|Zoroark, L80');
  assert.equal(by(terminal(v),'Mimikyu').condition,'190/201');
});

test('drag cannot choose the outgoing identity even when it has Illusion', () => {
  const v=fixture();v.ingest('|drag|p1a: Mimikyu|Mimikyu, L80|201/201');
  assert.deepEqual(terminal(v).own.filter(m=>m.active).map(m=>m.ident),['Mimikyu']);
});

test('base Illusion remains a candidate after a temporary ability change', () => {
  const v=fixture(),mons=roster();mons[0].ability='Mummy';mons[0].baseAbility='Illusion';v.acceptRequest(req(mons));
  v.expectSwitch('Snorlax');v.ingest('|switch|p1a: Snorlax|Snorlax, L80|201/201');
  v.ingest('|drag|p1a: Mimikyu|Mimikyu, L80|235/235');
  assert.throws(()=>terminal(v),/Unresolved own appearance/);
  mons[0].ability='Illusion';v.acceptRequest(req(mons));
  assert.deepEqual(terminal(v).own.filter(m=>m.active).map(m=>m.ident),['Zoroark']);
});

test('non-Illusion hint must match the observed incoming appearance', () => {
  const v=fixture();v.expectSwitch('Pikachu');
  assert.throws(()=>v.ingest('|switch|p1a: Snorlax|Snorlax, L80|201/201'),/hint disagrees/);
});
