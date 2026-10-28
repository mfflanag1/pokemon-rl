import {RosterView} from '../full-showdown/view.mjs';
import {Dex} from '../reference-node/engine.mjs';
import {PublicTypes} from './types.mjs';

/** Additive public-protocol repair. The parent still supplies all non-type data. */
export class DynamicRosterView extends RosterView {
  constructor(side) {
    super(side);
    this.activeOwnAppearance = null;
    this.publicTypes = new PublicTypes(species => {
      const entry = Dex.species.get(species);
      if (!entry.exists) throw new Error(`Unknown observed species: ${species}`);
      return [...entry.types];
    });
  }

  ingest(line) {
    const [,kind,ident,details] = line.split('|');
    if (['switch','drag','replace'].includes(kind) && ident?.startsWith(this.side)) this.activeOwnAppearance = ident;
    this.publicTypes.ingest(line);
    super.ingest(line);
    if (['switch','drag'].includes(kind)) {
      const tera = /(?:^|, )tera:([^,]+)/.exec(details ?? '')?.[1] ?? null;
      if (ident?.startsWith(this.foe)) this.slots[this.active].tera = tera;
      else if (ident?.startsWith(this.side)) {
        const name = ident.split(': ').slice(1).join(': ');
        this.ownEvents.get(name).tera = tera;
      }
    }
  }

  observe(request, requestId) {
    const player = super.observe(request, requestId);
    for (const mon of player.opponent) {
      if (mon) mon.types = this.publicTypes.typesFor(`${this.foe}: ${mon.ident}`, mon.species);
    }
    // Own fields are added only after the inherited v1 validation in snapshot().
    return player;
  }

  withOwnTypes(player) {
    return {...player, schema_version: 'full-showdown-player-v2',
      own: player.own.map(mon => ({...mon,
        // Temporary types cannot persist on the bench. This also prevents an
        // active Illusion appearance from contaminating its real bench namesake.
        types: mon.active ? this.publicTypes.typesFor(
          this.activeOwnAppearance ?? `${this.side}: ${mon.ident}`, mon.species) : null}))};
  }
}
