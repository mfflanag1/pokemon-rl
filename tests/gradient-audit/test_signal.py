import copy
import hashlib
import unittest
import numpy as np
from test_core import fixture
from pokemon_gradient_audit.signal import row_metadata, game_halves, between_share, audit_signal, actor_gradient
from pokemon_matchup.learning import validate
from pokemon_full_train.learning import derived_seed


def linked_fixture():
    p, opt, batch = fixture()
    batch['advantages'] = np.linspace(-.2, .3, 8, dtype=np.float32)
    batch['returns'] = batch['advantages'] + np.float32(.5)
    games = []
    for gid in range(4):
        actors = []
        for i in range(2*gid, 2*gid+2):
            actors.append({'side':'p1' if i%2==0 else 'p2', 'value':.5,
                'feature_sha256':hashlib.sha256(batch['features'][i].tobytes()).hexdigest(),
                'mask':batch['mask'][i].tolist(), 'action':int(batch['actions'][i]),
                'log_probability':float(batch['old_logp'][i])})
        games.append({'episode':gid, 'opponent':None, 'split':'train', 'learner_side':'p1',
                      'outcome':{'kind':'terminal','rewards':{'p1':1,'p2':0}}, 'actor_trace':actors})
    ids = [6,1,4,7,0,5,2,3]
    ledger = {'games':games, 'strata':[{'first_episode':0,'next_episode':4,'opponent':None,
               'selection':{'collected_rows':8,'selected_rows':8,'selected_indices':ids}}]}
    return p, {k:v[ids] for k,v in batch.items()}, ledger


class SignalTests(unittest.TestCase):
    def test_selected_identity_and_corruption(self):
        _, batch, ledger = linked_fixture()
        meta = row_metadata(ledger,batch)
        self.assertEqual([m['game'] for m in meta], [3,0,2,3,0,2,1,1])
        for field in ['features','actions','old_logp']:
            bad = {k:v.copy() for k,v in batch.items()}
            bad[field][0] += 1
            with self.assertRaises(ValueError):
                row_metadata(ledger,bad)
        bad = copy.deepcopy(ledger)
        bad['strata'][0]['selection']['selected_indices'][0] = 1
        with self.assertRaises(ValueError):
            row_metadata(bad,batch)

    def test_game_seats_never_split(self):
        _,batch,ledger = linked_fixture()
        meta = row_metadata(ledger,batch)
        gids = np.asarray([m['game'] for m in meta])
        first = game_halves(meta,142)
        second = game_halves(meta,142)
        for (a,b),(c,d) in zip(first,second):
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            self.assertFalse(set(gids[a]) & set(gids[b]))
            self.assertTrue(np.all(a|b))
            self.assertEqual(a.sum(),4)

    def test_mixed_disjoint_ranges_and_seed(self):
        p,batch,ledger = linked_fixture()
        original={k:v[np.argsort([6,1,4,7,0,5,2,3])] for k,v in batch.items()}
        for game in ledger['games'][2:]:
            game['opponent']='tactical'
            for actor in game['actor_trace']:actor['side']='p1'
        ledger['strata']=[{'first_episode':0,'next_episode':2,'opponent':None,
            'selection':{'collected_rows':4,'selected_rows':4,'selected_indices':[2,0,3,1]}},
            {'first_episode':2,'next_episode':4,'opponent':'tactical',
            'selection':{'collected_rows':4,'selected_rows':4,'selected_indices':[1,3,0,2]}}]
        batch={k:v[[2,0,3,1,5,7,4,6]] for k,v in original.items()}
        metadata=row_metadata(ledger,batch)
        self.assertEqual([m['game'] for m in metadata],[1,0,1,0,2,3,2,3])
        seed=derived_seed('f27-game-halves',7103,'mixed',0)
        self.assertIsInstance(seed,int)
        result=audit_signal(p,batch,ledger,seed)
        self.assertEqual(result['streams']['tactical']['selected_rows'],4)
        self.assertEqual(result['streams']['tactical']['won_tactical_games'],2)

    def test_half_norm_matches_compacted_row_mean(self):
        p,batch,_=linked_fixture()
        mask=np.array([True,False,False,True,True,False,False,False])
        full=actor_gradient(p,validate({**batch,'valid':mask}))
        subset=actor_gradient(p,validate({**{k:v[mask] for k,v in batch.items()},'valid':np.ones(3,bool)}))
        for k in p:np.testing.assert_allclose(full[k],subset[k],atol=1e-7,rtol=1e-5)

    def test_variance_decomposition_and_signal(self):
        self.assertEqual(between_share([0,0,1,1],np.array([0,0,1,1])),1.)
        self.assertEqual(between_share([0,1,0,1],np.array([0,0,1,1])),0.)
        self.assertIsNone(between_share([1,1],np.array([0,1])))
        p,batch,ledger = linked_fixture()
        result = audit_signal(p,batch,ledger,142)
        self.assertEqual(result['streams']['self_play']['selected_physical_games'],4)
        self.assertEqual(result['streams']['self_play']['selected_game_seat_trajectories'],8)
        self.assertIsNone(result['streams']['self_play']['won_tactical_games'])
        self.assertEqual(len(result['game_disjoint_halves']),4)
        self.assertTrue(all(-1.000001<=r['actor_cosine']<=1.000001 for r in result['game_disjoint_halves']))


if __name__ == '__main__':
    unittest.main()
