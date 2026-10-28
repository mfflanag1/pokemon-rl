import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize, forward, distribution
from pokemon_full.learner import parameter_hash
from pokemon_full_train.learning import prepare_batch, optimize
from pokemon_full_train.storage import digest, load_checkpoint
from pokemon_full_learning.collector import Pool as F4Pool
from pokemon_mixture import campaign, collector
from pokemon_mixture.collector import Pool
from pokemon_mixture.learning import select_batch, optimize_batch, training_specs, quotas_for, collect_update
from pokemon_mixture.statistics import contrasts, mixture_gate
from pokemon_mixture.verification import audit

ROOT = Path(__file__).resolve().parents[2]


def small_plan():
    p = json.loads((ROOT/'contracts/opponent-mixture-v1/plan.json').read_text())
    p.update(profile='test', seeds=[9981], updates_per_arm=2, selected_rows_per_update=256,
             evaluation_pairs_per_opponent=2, evaluation_chunk_pairs=2, max_turns=5)
    p['statistics'].update(family_cells=9, resamples=100)
    return p


def synthetic_rows(n=512):
    rows = []
    for i in range(n):
        terminal = i % 256 == 255
        mask = np.zeros(14, bool); mask[:4] = True
        rows.append(dict(features=np.zeros(928, np.float32), mask=mask, action=i % 4,
            old_logp=-float(np.log(4)), value=.5, reward=float(terminal),
            next_index=-1 if terminal else i+1, bootstrap=0., terminal=terminal, trajectory=i//256))
    return rows


def game(i, swap, reward, terminal=True):
    return dict(episode=i, swap=swap, learner_side='p2' if swap else 'p1', opponent='tactical',
        provenance={'pair': i, 'seat': swap}, decisions=1, value_mean=.5,
        terminal_value_mse=(.5-reward)**2 if terminal else None,
        outcome={'kind': 'terminal' if terminal else 'administrative_cutoff', 'rewards': {'p1': reward, 'p2': reward}})


class Budgets(unittest.TestCase):
    def test_full_trajectory_gae_before_selection_and_exact_quota(self):
        rows = synthetic_rows()
        full = prepare_batch(rows)
        batch, ledger = select_batch({'self_play': [(rows, {})]}, {'self_play': 256}, 1, 'self_play', 0)
        ids = ledger['self_play']['selected_indices']
        self.assertEqual(len(set(ids)), 256)
        self.assertEqual(ledger['self_play']['excluded_rows'], 256)
        for key in full:
            np.testing.assert_array_equal(batch[key], full[key][ids])
        self.assertTrue(any(rows[i]['next_index'] not in ids and not rows[i]['terminal'] for i in ids))
        with self.assertRaises(ValueError):
            select_batch({'self_play': [(rows[:5], {})]}, {'self_play': 256}, 1, 'self_play', 0)

    def test_optimizer_matches_frozen_ppo_on_same_prepared_batch(self):
        rows = synthetic_rows(256); p = initialize(71); z = {k: jnp.zeros_like(v) for k, v in p.items()}
        a, ao, am = optimize(p, (z, dict(z), jnp.int32(0)), rows, 99)
        b, bo, bm = optimize_batch(p, (z, dict(z), jnp.int32(0)), prepare_batch(rows), 99)
        self.assertEqual(parameter_hash(a), parameter_hash(b))
        self.assertEqual(campaign.optimizer_hashes(ao), campaign.optimizer_hashes(bo))
        self.assertEqual(am, bm); self.assertEqual(int(bo[2]), 2)

    def test_namespaces_seats_and_registered_exposure(self):
        p = small_plan(); namespaces = set()
        for arm in p['arms']:
            self.assertEqual(sum(quotas_for(p, arm).values()), 256)
            for stratum in quotas_for(p, arm):
                specs = training_specs(p, 9981, arm, stratum, 0)
                namespaces.add(specs[0]['seed'])
                self.assertEqual([s['learner_side'] for s in specs], ['p1', 'p2']*4)
        self.assertEqual(len(namespaces), 4)
        reordered = json.loads(json.dumps(p, sort_keys=True))
        for arm in p['arms']:
            self.assertEqual(list(quotas_for(p, arm)), list(quotas_for(reordered, arm)))
        prod = json.loads((ROOT/'contracts/opponent-mixture-v1/plan.json').read_text())
        campaign.validate_plan(prod)
        self.assertEqual(3*2*prod['updates_per_arm']*prod['selected_rows_per_update'], 786432)
        prod['updates_per_arm'] -= 1
        with self.assertRaisesRegex(ValueError, 'registered'): campaign.validate_plan(prod)

    def test_collection_ceiling_fails_instead_of_expanding_budget(self):
        p = small_plan(); p['max_games_per_stratum_update'] = 8
        with patch('pokemon_mixture.learning.check_training'), self.assertRaisesRegex(RuntimeError, 'ceiling'):
            # Empty rows are a controlled test fixture, never a valid real episode.
            class FixturePool:
                def collect(self, parameters, specs):
                    return [([], {'learner_rows': 0}) for _ in specs], {}
            collect_update(FixturePool(), None, p, 1, 'self_play', 0, {'self_play': 0})

    def test_gate_preserves_every_seed_and_cutoff_uncertainty(self):
        p = small_plan(); initial = {}; final = {}
        for op in p['opponents']:
            initial[op] = [game(i, s, 0) for i in range(16) for s in [False, True]]
            for g in initial[op]: g['opponent'] = op
        final['self_play'] = copy.deepcopy(initial)
        final['mixture'] = copy.deepcopy(initial)
        for panel in final['mixture'].values():
            for g in panel: g['outcome']['rewards'] = {'p1': 1., 'p2': 1.}
        c = contrasts(initial, final, p['statistics'])
        r = {'seed': 1, 'contrasts': c}
        self.assertTrue(mixture_gate([r], p['statistics'])['passed'])
        negative = copy.deepcopy(r); negative['seed'] = 2
        negative['contrasts']['mixture_versus_self_play']['tactical']['approximate_familywise_bootstrap_interval'] = [-.1, .1]
        gate = mixture_gate([r, negative], p['statistics'])
        self.assertFalse(gate['passed']); self.assertEqual(len(gate['per_seed']), 2)
        for panel in final['mixture'].values():
            for g in panel:
                g['outcome'] = {'kind': 'administrative_cutoff', 'rewards': {'p1': 0., 'p2': 0.}}
                g['terminal_value_mse'] = None
        uncertain = contrasts(initial, final, p['statistics'])
        self.assertEqual(uncertain['mixture_versus_self_play']['tactical']['cutoff_aware_point_interval'], [0., 1.])
        self.assertFalse(mixture_gate([{'seed': 1, 'contrasts': uncertain}], p['statistics'])['passed'])


class Integration(unittest.TestCase):
    def test_frozen_opponent_rows_excluded_and_on_policy_log_probabilities(self):
        p = initialize(73); jobs = [dict(seed=73, episode=i, max_turns=8, opponent=o, learner_side=s)
            for i, (o, s) in enumerate((('base_power', 'p1'), ('base_power', 'p2'), ('tactical', 'p1'), ('tactical', 'p2')))]
        original = collector.batched_actor
        def isolated(parameters, players, *args):
            self.assertTrue(all('tactical_actions' not in x and 'provenance' not in x for x in players))
            return original(parameters, players, *args)
        with tempfile.TemporaryDirectory() as d, Pool(d, 2) as pool, patch.object(collector, 'batched_actor', isolated):
            results, _ = pool.collect(p, jobs)
        for (rows, g), spec in zip(results, jobs):
            self.assertGreater(len(rows), 0); self.assertLess(len(rows), g['decisions'])
            self.assertEqual(len(rows), g['learner_decisions'])
            seat = 0 if spec['learner_side'] == 'p1' else 1
            self.assertEqual({r['trajectory'] for r in rows}, {seat})
            for i, r in enumerate(rows):
                if r['next_index'] >= 0:
                    self.assertGreater(r['next_index'], i)
                    self.assertEqual(rows[r['next_index']]['trajectory'], seat)
            x = jnp.asarray(np.stack([r['features'] for r in rows])); masks = jnp.asarray(np.stack([r['mask'] for r in rows]))
            logits, _ = forward(p, x); _, logs = distribution(logits, masks)
            actual = np.asarray(logs)[np.arange(len(rows)), [r['action'] for r in rows]]
            np.testing.assert_allclose(actual, [r['old_logp'] for r in rows], atol=2e-6, rtol=0)
            prepare_batch(rows)

    def test_self_play_and_development_match_f4(self):
        jobs = [dict(seed=74, episode=0, max_turns=4)]
        jobs += [dict(seed=8522026, episode=0, max_turns=4, split='development', opponent=o) for o in ['random', 'base_power', 'tactical']]
        with tempfile.TemporaryDirectory() as d:
            with F4Pool(Path(d)/'old', 2) as old: a, _ = old.collect(initialize(74), jobs)
            with Pool(Path(d)/'new', 2) as new: b, _ = new.collect(initialize(74), jobs)
        for (ar, ag), (br, bg) in zip(a, b):
            for key in ['learner_rows', 'learner_decisions']: bg.pop(key)
            self.assertEqual(ag, bg)
            self.assertEqual(len(ar), len(br))
            for x, y in zip(ar, br):
                for key in x: np.testing.assert_array_equal(x[key], y[key])

    def test_optimized_python_cannot_bypass_audit(self):
        r = subprocess.run([sys.executable, '-O', '-c',
            'from pokemon_mixture.verification import audit; audit("missing", "missing")'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0); self.assertIn('requires enabled assertions', r.stderr)

    def test_recovery_original_update_replay_and_corruption_refusal(self):
        plan = small_plan()
        with tempfile.TemporaryDirectory() as d:
            full = Path(d)/'full'; resumed = Path(d)/'resumed'
            expected = campaign.run_campaign(full, plan)
            original = campaign.save_checkpoint
            def interrupt(directory, p, opt, meta):
                result = original(directory, p, opt, meta)
                if meta['arm'] == 'mixture' and meta['cohorts'] == 1:
                    raise RuntimeError('injected after durable update')
                return result
            with patch.object(campaign, 'save_checkpoint', side_effect=interrupt), self.assertRaisesRegex(RuntimeError, 'injected'):
                campaign.run_campaign(resumed, plan)
            actual = campaign.run_campaign(resumed, plan, resume=True)
            self.assertEqual(actual['seeds'][0]['contrasts'], expected['seeds'][0]['contrasts'])
            for arm in plan['arms']:
                a, ao, _ = load_checkpoint(campaign.checkpoint_dir(full, 9981, arm), digest(full/'inputs/manifest.json'))
                b, bo, _ = load_checkpoint(campaign.checkpoint_dir(resumed, 9981, arm), digest(resumed/'inputs/manifest.json'))
                self.assertEqual(parameter_hash(a), parameter_hash(b))
                self.assertEqual(campaign.optimizer_hashes(ao), campaign.optimizer_hashes(bo))
                self.assertEqual(int(bo[2]), 4)
            self.assertEqual(actual['seeds'][0]['arms']['mixture']['complete_cycle_records'], 1)
            directory = campaign.checkpoint_dir(full, 9981, 'mixture')
            original_file = next(directory.glob('cohort-00001-*.npz'))
            (directory/'cohort-00001-orphan.npz').write_bytes(original_file.read_bytes())
            proof = audit(full, Path(d)/'audit')
            self.assertEqual(proof['totals']['selected_learner_rows'], 1024)
            self.assertEqual(proof['totals']['evaluation_games'], 36)
            self.assertTrue(all(r['original_next_update_exact'] for r in proof['replays'] if 'arm' in r))
            evidence = full/'seed-9981/evaluation/mixture-tactical-0000.json'
            bad = json.loads(evidence.read_text()); bad['games'][0]['trace'][0][2] = 99
            evidence.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError): audit(full, Path(d)/'negative')
            with self.assertRaisesRegex(ValueError, 'Completed'): campaign.run_campaign(resumed, plan, True)


if __name__ == '__main__': unittest.main()
