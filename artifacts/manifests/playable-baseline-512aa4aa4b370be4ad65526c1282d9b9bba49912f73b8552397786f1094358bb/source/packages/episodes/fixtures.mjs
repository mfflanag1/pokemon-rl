// Audited, named test interventions. These never run in the official campaign.
// The separate constructed profile and recipe are bound into the episode identity.
export const FIXTURES = ['single-faint-p1', 'single-faint-p2', 'double-faint', 'tie-after-first-choice',
  'live-switch-p1', 'live-switch-p2', 'mixed-switch', 'revival-p1', 'revival-p2'];

export function applyFixture(session, recipe, stage) {
  if (recipe === null) return;
  if (!FIXTURES.includes(recipe)) throw new Error('Unknown constructed fixture recipe');
  const battle = session.stream.battle;
  if (stage === 'initial' && recipe !== 'tie-after-first-choice') {
    const chosen = battle.getSide(recipe.endsWith('p2') ? 'p2' : 'p1');
    const sides = recipe === 'double-faint' ? battle.sides : [chosen];
    if (recipe.startsWith('single-faint') || recipe === 'double-faint' || recipe === 'mixed-switch') {
      for (const side of sides) battle.faint(side.active[0]);
    } else if (recipe.startsWith('revival')) battle.faint(chosen.pokemon[1]);
    battle.faintMessages();
    battle.checkFainted();
    if (recipe.startsWith('live-switch')) chosen.active[0].switchFlag = true;
    if (recipe === 'mixed-switch') battle.getSide('p2').active[0].switchFlag = true;
    if (recipe.startsWith('revival')) {
      chosen.addSlotCondition(0, 'revivalblessing', chosen.active[0], battle.dex.moves.get('revivalblessing'));
      chosen.active[0].switchFlag = 'revivalblessing';
    }
    battle.makeRequest('switch');
    // sendUpdates emits requests only when there is a new log entry. Make every
    // constructed intervention explicit, including live switches with no faint.
    battle.add('message', `Constructed R4 fixture: ${recipe}`);
    battle.sendUpdates();
  }
  if (stage === 'after-first-choice' && recipe === 'tie-after-first-choice') {
    // Exercise terminal cancellation while the opponent still has an unsubmitted
    // request. This is a constructed reference tie, not a naturally played draw.
    battle.tie();
    battle.sendUpdates();
  }
}
