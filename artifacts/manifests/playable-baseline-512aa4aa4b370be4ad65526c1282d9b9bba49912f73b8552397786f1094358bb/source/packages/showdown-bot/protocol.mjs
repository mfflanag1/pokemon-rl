export const FORMAT = 'gen9randombattle';
export const DEV_FORMAT = 'gen9botbaselinedevelopment';
export const toID = value => String(value).toLowerCase().replace(/[^a-z0-9]/g, '');
export const isLoopback = url => ['localhost', '127.0.0.1', '[::1]'].includes(new URL(url).hostname);

export function parseFrame(text) {
  if (typeof text !== 'string' || text.length > 2_000_000) throw new Error('Invalid or oversized protocol frame');
  let room = '';
  const result = [];
  for (const line of text.split('\n')) {
    if (line.startsWith('>')) {
      room = line.slice(1);
      if (!/^[a-z0-9-]+$/.test(room)) throw new Error('Invalid room ID');
    } else if (line) result.push({room, line});
  }
  return result;
}

export function legalChoices(request) {
  if (!request || request.wait) return [];
  const team = request.side?.pokemon;
  if (!Array.isArray(team) || team.length < 1 || team.length > 6) throw new Error('Expected a singles team');
  if (!['p1', 'p2'].includes(request.side.id) || team.some(p => !p.ident?.startsWith(`${request.side.id}: `))) {
    throw new Error('Foreign or malformed private request');
  }
  if (request.teamPreview) return [`team ${team.map((_, i) => i + 1).join('')}`];
  const fainted = p => /(?:^0(?:\/\d+)?(?: |$)| fnt$)/.test(p.condition);
  if (request.forceSwitch) {
    if (request.forceSwitch.length !== 1) throw new Error('Only singles are supported');
    if (!request.forceSwitch[0]) return ['pass'];
    const reviving = Boolean(team[0].reviving);
    const choices = team.flatMap((p, i) => (reviving ? fainted(p) : !p.active && !fainted(p)) ? [`switch ${i + 1}`] : []);
    if (!choices.length) throw new Error('Forced replacement has no eligible target');
    return choices;
  }
  if (request.active?.length !== 1) throw new Error('Only singles move requests are supported');
  const active = request.active[0];
  if (active.canDynamax || active.canMegaEvo || active.canZMove) throw new Error('Only Gen 9 mechanics are supported');
  const choices = active.moves.flatMap((m, i) => m.disabled ? [] : [
    `move ${i + 1}`, ...(active.canTerastallize ? [`move ${i + 1} terastallize`] : []),
  ]);
  if (!choices.length) throw new Error('No advertised move (expected Struggle)');
  if (!active.trapped) team.forEach((p, i) => {
    if (!p.active && !fainted(p)) choices.push(`switch ${i + 1}`);
  });
  return choices;
}

// Public events only. The request is supplied separately and never mixed with an opponent request.
export class PublicBattle {
  constructor() { this.active = {}; this.revealedTera = {}; this.turn = 0; this.ended = false; this.winner = null; this.tier = null; this.gameType = null; }
  ingest(line) {
    const [, type, ident, value, condition] = line.split('|');
    const seat = ident?.slice(0, 2);
    if (type === 'split') throw new Error('Unprojected simulator output cannot enter the bot');
    if (type === 'turn') this.turn = Number(ident);
    if (type === 'tier') this.tier = toID(ident);
    if (type === 'gametype') this.gameType = ident;
    if (['switch', 'drag', 'replace'].includes(type)) {
      const previous = this.active[seat];
      this.active[seat] = {ident, species: value.split(',')[0],
        condition: condition ?? (type === 'replace' ? previous?.condition : null) ?? null, types: null,
        tera: this.revealedTera[ident] ?? (type === 'replace' ? previous?.tera ?? null : null)};
      if (this.active[seat].tera) this.revealedTera[ident] = this.active[seat].tera;
    }
    const mon = this.active[seat];
    if (mon && ['-damage', '-heal', '-sethp'].includes(type)) mon.condition = value;
    if (mon && ['detailschange', '-formechange'].includes(type)) { mon.species = value.split(',')[0]; mon.types = null; }
    if (mon && type === '-terastallize') { mon.tera = value; this.revealedTera[ident] = value; }
    if (mon && type === '-start' && value === 'typechange') mon.types = condition.split('/');
    if (mon && type === '-start' && value === 'typeadd') mon.addedType = condition;
    if (mon && type === 'faint') mon.condition = '0 fnt';
    if (type === 'win' || type === 'tie') { this.ended = true; this.winner = type === 'win' ? ident : null; }
  }
}
