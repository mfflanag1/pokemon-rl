"""Synthetic F35 loss, sampling and accepted-teacher-data qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from pokemon_full.policy import distribution
from pokemon_structured_policy import learning as original
from pokemon_structured_policy.network import SHAPES, forward
from pokemon_structured_policy.features import FEATURES, encode_packed, empty_history
from pokemon_teacher_assisted import data, learning
from test_f34_learning import packed_fixture, player_fixture

ROOT = Path(__file__).resolve().parents[1]


def state():
    rng = np.random.default_rng(3501)
    p = {k: jnp.asarray(rng.normal(0, .01, s).astype(np.float32)) for k, s in SHAPES.items()}
    zero = jax.tree.map(jnp.zeros_like, p)
    return p, (zero, zero, jnp.int32(0))


def demo(n=256, seed=3502):
    x = packed_fixture(seed, n)
    mask = np.ones((n, 14), bool); mask[:, 13] = False
    return dict(features=x, mask=mask, actions=np.arange(n, dtype=np.int32) % 13)


def ppo(p, n=1024):
    b = demo(n)
    logits, _ = forward(p, jnp.asarray(b['features']))
    _, logs = distribution(logits, jnp.asarray(b['mask']))
    return {**b, 'old_logp': np.asarray(logs)[np.arange(n), b['actions']].astype(np.float32),
            'advantages': np.linspace(-1, 1, n, dtype=np.float32),
            'returns': np.linspace(0, 1, n, dtype=np.float32)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True) + '\n')


class AssistedLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.opt = state()
        cls.batch = ppo(cls.p)
        cls.dataset = data.Dataset([('one', demo(2)), ('two', demo(30, 3503))], split='train')

    def assert_state_equal(self, left, right):
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
            np.testing.assert_array_equal(a, b)

    def test_zero_coefficient_original_full_cohort_is_exact(self):
        expected = original.optimize_batch(self.p, self.opt, self.batch, 718)
        actual = learning.optimize_batch(self.p, self.opt, self.batch, 718, coefficient=0.)
        self.assert_state_equal(expected[:2], actual[:2])
        self.assertEqual(expected[2], actual[2])
        self.assertEqual(int(actual[1][2]), 8)

    def test_auxiliary_gradient_is_actor_only_but_reaches_shared_trunk(self):
        b = {k: jnp.asarray(v) for k, v in demo(4).items()}
        loss, gradients = jax.value_and_grad(lambda p: learning.demo_objective(p, b)[0])(self.p)
        self.assertTrue(np.isfinite(float(loss)))
        for key in ('vw', 'vb', 'matchup_value', 'sp_value_out'):
            self.assertEqual(np.count_nonzero(gradients[key]), 0, key)
        self.assertGreater(float(np.linalg.norm(gradients['w'])), 0)
        self.assertGreater(float(np.linalg.norm(gradients['sp_context_w'])), 0)
        q = {k: self.p[k] - jnp.float32(.01) * gradients[k] for k in self.p}
        _, before = forward(self.p, b['features'])
        _, after = forward(q, b['features'])
        self.assertGreater(float(np.max(np.abs(before - after))), 0.)

    def test_one_assisted_update_and_recovery_are_exact(self):
        b = {k: v[:256] for k, v in self.batch.items()}; b['valid'] = np.ones(256, bool)
        demonstration = self.dataset.sample(seed=17, cohort=0, update=0)
        q, qo, m = learning.update(self.p, self.opt, b, demonstration)
        self.assertEqual(int(qo[2]), 1)
        self.assertAlmostEqual(m['loss'], m['ppo_loss'] + .1 * m['demo_nll'], places=5)
        self.assertTrue(all(np.isfinite(v) for v in m.values()))
        from pokemon_structured_policy import checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint.save(tmp, q, qo, dict(input_hash='a'*64, seed=7103, cohorts=0, next_episode=0))
            restored, ro, _ = checkpoint.load(tmp)
            next_demo = self.dataset.sample(seed=17, cohort=0, update=1)
            expected = learning.update(q, qo, b, next_demo)
            actual = learning.update(restored, ro, b, next_demo)
            self.assert_state_equal(expected[:2], actual[:2]); self.assertEqual(expected[2], actual[2])

    def test_demo_validation_rejects_ppo_fields_bad_dtypes_masks_and_padding(self):
        for key in ('old_logp', 'advantages', 'returns', 'valid'):
            b = demo(2); b[key] = np.zeros(2, np.float32)
            with self.assertRaises(ValueError): data.validate_demo(b)
        cases = []
        b = demo(2); b['features'][0, 0] = np.nan; cases.append(b)
        b = demo(2); b['features'] = b['features'].astype(np.float64); cases.append(b)
        b = demo(2); b['mask'] = b['mask'].astype(np.int32); cases.append(b)
        b = demo(2); b['actions'] = b['actions'].astype(np.int64); cases.append(b)
        b = demo(2); b['actions'][0] = 13; cases.append(b)
        b = demo(2); b['mask'][0] = False; cases.append(b)
        b = demo(2); b['actions'][0] = -1; cases.append(b)
        b = demo(2); b = {k: v[:0] for k, v in b.items()}; cases.append(b)
        for b in cases:
            with self.assertRaises(ValueError): data.validate_demo(b)
        b = {k: v[:256] for k, v in self.batch.items()}; b['valid'] = np.ones(256, bool)
        b['valid'][-1] = False
        with self.assertRaisesRegex(ValueError, 'no padding'):
            learning.update(self.p, self.opt, b, demo())

    def test_game_uniform_sampler_is_deterministic_and_independent(self):
        _, ids = self.dataset.sample_with_indices(seed=22, cohort=11, update=0, rows=10000)
        _, again = self.dataset.sample_with_indices(seed=22, cohort=11, update=0, rows=10000)
        np.testing.assert_array_equal(ids, again)
        self.assertLess(abs(np.mean(ids < 2) - .5), .02)
        self.assertLess(abs(np.mean(ids[ids < 2] == 0) - .5), .03)
        _, different = self.dataset.sample_with_indices(seed=22, cohort=11, update=1, rows=10000)
        self.assertFalse(np.array_equal(ids, different))
        rng = np.random.default_rng(9); first = rng.permutation(1024)
        self.dataset.sample(seed=22, cohort=11, update=0)
        second = rng.permutation(1024)
        reference = np.random.default_rng(9)
        np.testing.assert_array_equal(first, reference.permutation(1024))
        np.testing.assert_array_equal(second, reference.permutation(1024))
        with self.assertRaises(ValueError): self.dataset.arrays['actions'][0] = 7

    def test_ppo_permutations_and_step_count_match_with_assistance(self):
        observed = []
        def capture(p, opt, batch, demonstration):
            observed.append(np.asarray(batch['returns']))
            self.assertEqual(len(demonstration['actions']), 256)
            return p, (opt[0], opt[1], opt[2] + 1), {}
        with patch.object(learning, 'update', side_effect=capture):
            _, opt, records = learning.optimize_batch(self.p, self.opt, self.batch, 391,
                dataset=self.dataset, demo_seed=27, cohort=5)
        rng = np.random.default_rng(391)
        expected = []
        for _ in range(2):
            ids = rng.permutation(1024)
            expected.extend(self.batch['returns'][ids[a:a+256]] for a in range(0,1024,256))
        self.assertEqual(len(records), 8); self.assertEqual(int(opt[2]), 8)
        for a, b in zip(observed, expected): np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            learning.optimize_batch(self.p, self.opt, self.batch, 1,
                dataset=data.Dataset([('probe', demo(1))], split='development'), demo_seed=2)
        with self.assertRaises(ValueError):
            learning.optimize_batch(self.p, self.opt, self.batch, 1, coefficient=.2)

    def test_probe_uses_one_fixed_shape_and_excludes_padding(self):
        corpus=data.Dataset([('one',demo(257)),('two',demo(3,3506))],split='development',terminal_returns=[1.,None])
        changed={**self.p,'vb':self.p['vb']+jnp.float32(.1)}
        shapes=[]
        predict=learning._probe_predictions
        def record(p,features,mask):
            shapes.append(tuple(features.shape))
            return predict(p,features,mask)
        with patch.object(learning,'_probe_predictions',side_effect=record):
            result=learning.probe_metrics(changed,corpus,reference_parameters=self.p)
        self.assertEqual(shapes,[(256,FEATURES)]*4)
        self.assertEqual(result['rows'],260);self.assertEqual(result['terminal_rows'],257)
        logits,values=forward(changed,jnp.asarray(corpus.arrays['features']))
        probabilities,logs=distribution(logits,jnp.asarray(corpus.arrays['mask']))
        expected=-np.asarray(logs)[np.arange(260),corpus.arrays['actions']].mean()
        self.assertAlmostEqual(result['decision_nll'],float(expected),places=5)
        self.assertGreater(result['decision_value_absolute_drift'],0.)

    def test_probe_reports_all_games_and_value_drift(self):
        result = learning.probe_metrics(self.p, self.dataset, reference_parameters=self.p)
        self.assertEqual(result['games'], 2); self.assertEqual(result['rows'], 32)
        self.assertEqual(result['decision_value_absolute_drift'], 0)
        self.assertEqual(sum(result['action_counts']), 32)
        self.assertTrue(np.isfinite(result['game_nll']))
        corpus = data.Dataset([('terminal',demo(2)),('cutoff',demo(5))], split='development',terminal_returns=[1.,None])
        result = learning.probe_metrics(self.p,corpus)
        _, values = forward(self.p,jnp.asarray(corpus.arrays['features'][:2]))
        self.assertEqual(result['terminal_games'],1);self.assertEqual(result['terminal_rows'],2)
        self.assertAlmostEqual(result['decision_terminal_value_mse'],float(np.mean((np.asarray(values)-1.)**2)),places=6)
        self.assertEqual(set(corpus.sample(seed=1,cohort=0,update=0)),set(data.FIELDS))


class DemonstrationAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT/'artifacts/reviews/f17-switch-audit/public-catalog.json').read_text())

    def fixture(self, root):
        folder = root/'game'; folder.mkdir()
        player = player_fixture(); history = empty_history('p1')
        features, mask = encode_packed(player, self.catalog, history)
        provenance = {'split':'train','episode_id':'f2:train:350:0:base',
                      'scenario':{'run_seed':350,'episode_index':0,'swap':False,'partition':'train'},
                      'teams':[{'partition':'train','team_key':'a'*64},
                                               {'partition':'train','team_key':'b'*64}]}
        external = {'side':'p1','request':{'synthetic':True}}
        frame = {'players':{'p1':player,'p2':{'side':'p2','request_id':1,'phase':'wait','mask':[False]*14}},
                 'observable_history':{'p1':history},'split':'train','provenance':provenance,'outcome':None}
        (folder/'transcript.jsonl').write_text(json.dumps({'frame':frame,'external':external})+'\n')
        label = dict(row=0,step_index=0,request_id=1,side='p1',action=0,submission_index=0,
            player_sha256=data.json_hash(player),history_sha256=data.json_hash(history),
            features_sha256=data.array_hash(features),mask_sha256=data.array_hash(mask),
            external_input_sha256=data.json_hash(external))
        write_json(folder/'labels.json', dict(schema_version='f35-teacher-labels-v1',game_id='f2:train:350:0:base',
            split='train',external_side='p1',labels=[label]))
        np.savez(folder/'demonstrations.npz',features=features[None],mask=mask[None],actions=np.array([0],np.int32),
            step_indices=np.array([0],np.int32),request_ids=np.array([1],np.int32))
        outcome={'kind':'terminal','winner':'Full p1','rewards':{'p1':1,'p2':0}}
        write_json(folder/'result.json', dict(status='completed',game_id='f2:train:350:0:base',split='train',external_side='p1',
            run_seed=350,episode_index=0,swap=False,fp_seed=351,
            provenance=provenance,outcome=outcome,accepted_labels=1,foul_play_decisions=1,decisions=1,
            rejected_actions={'p1':0,'p2':0},steps=[dict(index=0,actions={'p1':0},request_ids={'p1':1},
            external={'action':0,'request_id':1},external_phase='move',
            teacher_feedback={'acknowledged':True,'request_id':1,'accepted':True},
            external_input_sha256=data.json_hash(external),submitted=[dict(side='p1',action=0,request_id=1,
                command='move 1',accepted=True,feedback=[])])]))
        with (folder/'transcript.jsonl').open('a') as file:
            file.write(json.dumps({'frame':{**frame,'outcome':outcome},'external':external})+'\n')
        schedule=[dict(game_id='f2:train:350:0:base',split='train',external_side='p1',
            run_seed=350,episode_index=0,swap=False,fp_seed=351)]
        report=dict(schema_version='f35-teacher-data-v1',status='passed',input_hash='c'*64,schedule=schedule,
            games=[dict(**schedule[0],directory='game',provenance=provenance,outcome=outcome,accepted_labels=1,files={})])
        self.rebind(root,report)
        return report,schedule

    def rebind(self,root,report):
        report['games'][0]['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'game').iterdir()}
        write_json(root/'report.json',report)

    def load(self,root,schedule,**kwargs):
        return data.Dataset.load(root,expected_input_hash='c'*64,expected_schedule=schedule,catalog=self.catalog,**kwargs)

    def test_complete_hashed_accepted_teacher_fixture_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,schedule=self.fixture(root)
            corpus=self.load(root,schedule)
            self.assertEqual(corpus.rows,1); self.assertEqual(corpus.game_ids,('f2:train:350:0:base',))
            self.assertTrue(corpus.manifest_sha256)
            self.assertEqual(corpus.terminal_returns, (1.,))
            self.assertEqual(corpus.team_keys, frozenset(['a'*64,'b'*64]))

    def test_digest_duplicate_schedule_and_wrong_input_fail(self):
        for mutation in ('digest','duplicate','input','missing'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);report,schedule=self.fixture(root)
                if mutation=='digest': (root/'game/labels.json').write_text('{}')
                elif mutation=='duplicate': report['games'].append(copy.deepcopy(report['games'][0]));write_json(root/'report.json',report)
                elif mutation=='input': report['input_hash']='d'*64;write_json(root/'report.json',report)
                else: report['games']=[];write_json(root/'report.json',report)
                with self.assertRaises(ValueError): self.load(root,schedule)

    def test_rehashed_wrong_role_rejected_label_foreign_view_or_action_fails(self):
        for mutation in ('role','rejected','request','feature','foreign','source_schedule','ack_request','ack_status','count','omitted'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);report,schedule=self.fixture(root)
                result=json.loads((root/'game/result.json').read_text())
                labels=json.loads((root/'game/labels.json').read_text())
                if mutation=='role': result['provenance']['teams'][0]['partition']='development';report['games'][0]['provenance']=result['provenance']
                elif mutation=='rejected': result['steps'][0]['submitted'][0]['accepted']=False
                elif mutation=='request': result['steps'][0]['submitted'][0]['request_id']=2
                elif mutation=='feature': labels['labels'][0]['features_sha256']='0'*64
                elif mutation=='foreign': labels['labels'][0]['side']='p2'
                elif mutation=='source_schedule': result['run_seed']=351
                elif mutation=='ack_request': result['steps'][0]['teacher_feedback']['request_id']=3
                elif mutation=='ack_status': result['steps'][0]['teacher_feedback']['acknowledged']=False
                elif mutation=='count': result['foul_play_decisions']=0
                else: result['steps'][0]['submitted'].append(copy.deepcopy(result['steps'][0]['submitted'][0]))
                write_json(root/'game/result.json',result);write_json(root/'game/labels.json',labels);self.rebind(root,report)
                with self.assertRaises(ValueError): self.load(root,schedule)

    def test_teacher_rejection_and_wait_are_retained_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,schedule=self.fixture(root)
            result=json.loads((root/'game/result.json').read_text())
            frames=[json.loads(line) for line in (root/'game/transcript.jsonl').read_text().splitlines()]
            initial=frames[0]
            rejected=copy.deepcopy(result['steps'][0])
            rejected['submitted'][0].update(accepted=False,feedback=['[Unavailable choice] trapped'])
            rejected['teacher_feedback']['accepted']=False
            waiting=copy.deepcopy(initial)
            waiting['frame']['players']['p1'].update(phase='wait',mask=[False]*14,request_id=2)
            waiting['frame']['players']['p2'].update(phase='move',mask=[True]+[False]*13,request_id=2)
            wait_step=dict(index=1,actions={'p2':0},request_ids={'p2':2},external={'waiting':True},external_phase='wait',
                external_input_sha256=data.json_hash(waiting['external']),
                submitted=[dict(side='p2',action=0,request_id=2,command='move 1',accepted=True,feedback=[])])
            accepted=copy.deepcopy(result['steps'][0]);accepted['index']=2
            result.update(steps=[rejected,wait_step,accepted],decisions=3,foul_play_decisions=2,
                          rejected_actions={'p1':1,'p2':0})
            labels=json.loads((root/'game/labels.json').read_text());labels['labels'][0]['step_index']=2
            with np.load(root/'game/demonstrations.npz') as archive:
                arrays={k:archive[k] for k in archive.files}
            arrays['step_indices'][:]=2
            np.savez(root/'game/demonstrations.npz',**arrays)
            (root/'game/transcript.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in [initial,waiting,initial,frames[-1]]))
            write_json(root/'game/result.json',result);write_json(root/'game/labels.json',labels);self.rebind(root,report)
            corpus=self.load(root,schedule)
            self.assertEqual(corpus.rows,1)
            result['steps'][1]['teacher_feedback']={'acknowledged':True,'request_id':2,'accepted':True}
            write_json(root/'game/result.json',result);self.rebind(root,report)
            with self.assertRaisesRegex(ValueError,'waiting teacher'):
                self.load(root,schedule)

    def test_zero_label_game_and_quarantined_team_fail(self):
        with self.assertRaises(ValueError):
            data.Dataset([('empty',{k:v[:0] for k,v in demo(1).items()})],split='train')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,schedule=self.fixture(root)
            with self.assertRaises(ValueError):self.load(root,schedule,quarantine=['a'*64])


if __name__ == '__main__':
    unittest.main()
