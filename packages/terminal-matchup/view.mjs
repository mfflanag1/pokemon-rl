import {DynamicRosterView} from '../dynamic-matchup/view.mjs';

const name = ident => ident?.split(': ').slice(1).join(': ');
const activeIdent = ident => /^p[12]a: /.test(ident ?? '');
const clone = value => structuredClone(value);
const canDisguise = p => [p.ability,p.baseAbility].some(a => a?.toLowerCase() === 'illusion');

/** Own events use private roster identities. Public appearances remain confined
 * to the inherited opponent/type reducers, including while Illusion is intact. */
export class TerminalRosterView extends DynamicRosterView {
  constructor(side) {
    super(side);
    this.privateRoster = null;
    this.ownEntry = null;
    this.switchHint = null;
  }

  expectSwitch(ident) { this.switchHint = ident; }

  bindEntry(ident) {
    if (!this.ownEntry) throw new Error('Own identity without entry');
    if (this.ownEntry.ident && this.ownEntry.ident !== ident) throw new Error('Own entry identity changed');
    if (!this.privateRoster?.some(p => name(p.ident) === ident)) throw new Error('Unknown own roster identity');
    if (!this.ownEntry.ident) {
      for (const event of this.ownEvents.values()) event.active = false;
      const prior = this.ownEvents.get(ident) ?? {};
      const event = {...prior, ...this.ownEntry.event};
      this.ownEvents.set(ident, event);
      this.ownEntry = {...this.ownEntry, ident, event};
    }
  }

  acceptRequest(request) {
    if (!request?.side) return;
    if (request.side.id !== this.side) throw new Error('Foreign private request rejected');
    const roster = request.side.pokemon;
    if (roster.length !== 6 || new Set(roster.map(p => name(p.ident))).size !== 6) throw new Error('Expected six distinct own identities');
    const active = roster.filter(p => p.active);
    if (active.length !== 1) throw new Error('Expected one private active member');
    this.privateRoster = clone(roster);
    if (this.ownEntry) this.bindEntry(name(active[0].ident));
    for (const p of roster) {
      const id = name(p.ident), event = this.ownEvents.get(id) ?? {};
      // Private HP/status and active flags supersede earlier projected events.
      Object.assign(event, {condition:p.condition, active:p.active});
      if (p.terastallized) event.tera = p.terastallized;
      this.ownEvents.set(id,event);
    }
    if (this.ownEntry) this.ownEntry.event = this.ownEvents.get(this.ownEntry.ident);
    this.switchHint = null;
    // Store the private roster in protocol order; terminal frames have no request.
    super.observe(request, 0);
  }

  ingest(line) {
    const [,kind,ident,value,extra] = line.split('|');
    // Keep the sealed public/type behavior, discarding its nickname-keyed own map.
    const own = this.ownEvents;
    this.ownEvents = new Map();
    try { super.ingest(line); } finally { this.ownEvents = own; }
    if (!ident?.startsWith(this.side + ': ') && !ident?.startsWith(this.side + 'a: ')) return;
    if (['switch','drag'].includes(kind)) {
      if (this.ownEntry && !this.ownEntry.ident) throw new Error('Unresolved outgoing own appearance');
      const outgoing = this.ownEntry?.ident;
      this.ownEntry = {ident:null, appearance:ident, event:{active:true, boosts:{},
        condition:extra, tera:/(?:^|, )tera:([^,]+)/.exec(value ?? '')?.[1] ?? null}};
      if (kind === 'switch' && this.switchHint) {
        const target = this.privateRoster?.find(p => name(p.ident) === this.switchHint);
        if (!target || (name(ident) !== this.switchHint && !canDisguise(target))) throw new Error('Switch hint disagrees with appearance');
        this.bindEntry(this.switchHint);
      }
      else if (this.privateRoster) {
        // A normal matching member and an Illusion user can share an appearance.
        // Defer that ambiguity to the next private request, never guess by nickname.
        const candidates = this.privateRoster.filter(p => name(p.ident) !== outgoing && !p.condition.endsWith(' fnt') &&
          (name(p.ident) === name(ident) || canDisguise(p)));
        if (candidates.length === 1) this.bindEntry(name(candidates[0].ident));
      }
      this.switchHint = null;
      return;
    }
    if (kind === 'replace') {
      if (!this.ownEntry) throw new Error('Own replacement without entry');
      this.ownEntry.appearance = ident;
      this.bindEntry(name(ident));
      if (extra !== undefined) this.ownEntry.event.condition = extra;
      return;
    }
    let event;
    if (activeIdent(ident)) {
      if (!this.ownEntry) throw new Error('Own active event without appearance');
      event = this.ownEntry.event;
    } else {
      const id = name(ident);
      if (!this.privateRoster?.some(p => name(p.ident) === id)) return;
      event = this.ownEvents.get(id) ?? {};
      this.ownEvents.set(id,event);
    }
    if (['-damage','-heal','-sethp'].includes(kind)) event.condition = value;
    if (kind === 'faint') event.condition = '0 fnt';
    if (kind === '-status') event.condition = `${(event.condition ?? '100/100').split(' ')[0]} ${value}`;
    if (kind === '-curestatus') event.condition = (event.condition ?? '100/100').split(' ')[0];
    if (kind === '-terastallize') event.tera = value;
    if (['-boost','-unboost','-setboost'].includes(kind)) {
      event.boosts ??= {};
      event.boosts[value] = Math.max(-6,Math.min(6,kind === '-setboost' ? Number(extra) :
        (event.boosts[value] ?? 0) + (kind === '-boost' ? 1 : -1)*Number(extra)));
    }
    if (kind === '-clearboost') event.boosts = {};
  }

  observe(request, requestId) {
    if (!this.lastOwn && request?.side) this.acceptRequest(request);
    if (this.ownEntry && !this.ownEntry.ident) throw new Error('Unresolved own appearance at observation');
    return super.observe(request,requestId);
  }
}
