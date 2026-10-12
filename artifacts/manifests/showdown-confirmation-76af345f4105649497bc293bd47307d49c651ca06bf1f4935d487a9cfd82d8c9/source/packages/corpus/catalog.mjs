import {readFileSync} from 'node:fs';
import {Dex} from '../reference-node/engine.mjs';
import {UPSTREAM, fileHash, pin} from '../reference-node/provenance.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';

// A declarative denominator, not a reachability or battle-mechanics inventory.
export function sourceCatalog() {
  const source = 'data/random-battles/gen9/sets.json';
  const data = JSON.parse(readFileSync(`${UPSTREAM}/${source}`, 'utf8'));
  const entries = Object.fromEntries(Object.entries(data).map(([id, entry]) => {
    const species = Dex.species.get(id);
    if (!species.exists) throw new Error(`Unknown source species ${id}`);
    return [id, {
      source_id: id, base_species: Dex.species.get(species.baseSpecies).id,
      level: entry.level,
      sets: entry.sets.map((set, index) => ({index, template_hash: hashJSON(set), role: set.role,
        moves: set.movepool.map(x => Dex.moves.get(x).id).sort(),
        abilities: set.abilities.map(x => Dex.abilities.get(x).id).sort(),
        tera_types: set.teraTypes.map(x => Dex.types.get(x).id).sort(),
      })),
    }];
  }));
  const speciesNames = [...new Set(Dex.species.all().flatMap(s => [s.name, ...s.cosmeticFormes ?? []]))];
  const resolvers = {
    species: Object.fromEntries(speciesNames.map(name => {
      const s = Dex.species.get(name);
      return [name, {id: s.id, base_species: Dex.species.get(s.baseSpecies).id}];
    })),
    ability: Object.fromEntries(Dex.abilities.all().map(x => [x.name, x.id])),
    item: {'': '', ...Object.fromEntries(Dex.items.all().map(x => [x.name, x.id]))},
    tera_type: Object.fromEntries(Dex.types.all().map(x => [x.name, x.id])),
  };
  return {catalog_version: 'r3-declarative-sets-v1', showdown_commit: pin.commit,
    source, source_sha256: fileHash(`${UPSTREAM}/${source}`), entries, resolvers};
}
