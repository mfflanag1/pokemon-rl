from copy import deepcopy
from pathlib import Path
import unittest
from jsonschema.exceptions import ValidationError as SchemaValidationError

from pokemon_reference.contract import parse_json, hash_json, canonical, load_player_view, CONTRACT
from pokemon_reference.episodes import validate_episode, EpisodeValidationError

GOLDEN = parse_json((CONTRACT.parents[1] / 'tests/fixtures/r4-golden.json').read_bytes())


def reseal(bundle):
    e = bundle['episode']
    p = e['payload']
    for d in p['decisions']:
        d['decision_id'] = hash_json(['r4-decision-1', p['episode_key'],
                                      {k: v for k, v in d.items() if k != 'decision_id'}])
    for i, t in enumerate(p['transitions']):
        t['decision_id'] = p['decisions'][i]['decision_id']
        if t['target']['kind'] == 'decision':
            t['target']['id'] = p['decisions'][t['target']['sequence']]['decision_id']
    e['content_hash'] = hash_json({k: v for k, v in e.items() if k not in ('content_hash', 'artifact_id')})
    e['artifact_id'] = 'phase_episode:' + e['content_hash']
    return bundle


class EpisodeTests(unittest.TestCase):
    def bad(self, mutate):
        bundle = deepcopy(GOLDEN['bundle'])
        mutate(bundle['episode']['payload'])
        with self.assertRaises(EpisodeValidationError):
            validate_episode(reseal(bundle), GOLDEN['manifest'])

    def test_portable_golden_and_no_mutation(self):
        before = deepcopy(GOLDEN)
        result = validate_episode(GOLDEN['bundle'], GOLDEN['manifest'])
        self.assertEqual(result, GOLDEN['bundle'])
        self.assertEqual(GOLDEN, before)

    def test_cross_player_transition_rejected_after_rehash(self):
        self.bad(lambda p: p['transitions'][0].update(side='p2'))

    def test_bootstrap_cannot_have_terminal_reward(self):
        self.bad(lambda p: p['transitions'][0].update(reward_numerator=2))

    def test_phase_must_match_requests(self):
        self.bad(lambda p: p['frames'][0].update(phase='double_replacement'))

    def test_delta_must_match_saved_visible_prefix(self):
        def mutate(p):
            view = p['decisions'][0]['view']
            view['protocol_delta'][0] = '|message|injected future information'
            view['view_id'] = hash_json({k: v for k, v in view.items() if k != 'view_id'})
        self.bad(mutate)

    def test_stale_request_id_rejected(self):
        def mutate(p):
            view = p['decisions'][0]['view']
            view['request_id'] = 0
            view['view_id'] = hash_json({k: v for k, v in view.items() if k != 'view_id'})
        self.bad(mutate)

    def test_unsafe_cut_with_opponent_commitment_rejected(self):
        def mutate(p):
            frame = p['frames'][-1]
            frame['commitments']['p1'] = 0
            frame['pending_sides'] = ['p2']
        self.bad(mutate)

    def test_episode_diagnostics_are_rejected_by_actor_loader(self):
        with self.assertRaises(SchemaValidationError):
            load_player_view(canonical(GOLDEN['bundle']['episode']))

    def test_controller_and_order_are_bound_to_identity(self):
        self.bad(lambda p: p.update(submission_order='p2-first'))
        self.bad(lambda p: p.update(controller='scripted-semantic-choices-r4'))

    def test_prefix_hashes_match_canonical_full_history(self):
        p = GOLDEN['bundle']['episode']['payload']
        views = [d['view'] for d in p['decisions']] + list(p['bootstrap_views'].values())
        for v in views:
            self.assertEqual(v['protocol_prefix_hash'], hash_json(p['protocol'][v['viewer']][:v['protocol_end']]))

    def test_raw_evidence_must_project_to_stored_protocol(self):
        bundle = deepcopy(GOLDEN['bundle'])
        bundle['raw_output'][0]['data'] += '\n|message|unexpected evidence'
        bundle['raw_output_hash'] = hash_json(bundle['raw_output'])
        # The original normalized hash must fail even though the raw hash was resealed.
        with self.assertRaises(EpisodeValidationError):
            validate_episode(bundle, GOLDEN['manifest'])


if __name__ == '__main__':
    unittest.main()
