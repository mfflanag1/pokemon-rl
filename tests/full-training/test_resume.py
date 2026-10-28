"""Real-engine workflow fault injection: committed budget and next update survive resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pokemon_full_train import campaign
from pokemon_full_train.storage import load_checkpoint,digest
from pokemon_full.learner import parameter_hash

ROOT=Path(__file__).resolve().parents[2]


class ResumeIntegration(unittest.TestCase):
    def test_interrupt_after_commit_matches_uninterrupted_final_state_and_games(self):
        plan=json.loads((ROOT/'contracts/full-training-v1/plan.json').read_text())
        plan.update(seeds=[9871],train_games_per_seed=16,evaluation_pairs_per_opponent=2,max_turns=2)
        with tempfile.TemporaryDirectory() as d:
            full=Path(d)/'full'; resumed=Path(d)/'resumed'
            expected=campaign.run_campaign(full,plan)
            original=campaign.save_checkpoint
            def interrupt(directory,parameters,opt,meta):
                result=original(directory,parameters,opt,meta)
                if meta['cohorts']==1:
                    raise RuntimeError('injected interruption after durable commit')
                return result
            with patch.object(campaign,'save_checkpoint',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'injected interruption'):
                    campaign.run_campaign(resumed,plan)
            self.assertEqual(len(list((resumed/'failures').glob('*.json'))),1)
            self.assertFalse((resumed/'report.json').exists())
            actual=campaign.run_campaign(resumed,plan,resume=True)
            self.assertEqual(actual['seeds'][0]['final_parameter_hash'],expected['seeds'][0]['final_parameter_hash'])
            self.assertEqual(actual['seeds'][0]['optimizer_steps'],expected['seeds'][0]['optimizer_steps'])
            self.assertEqual(actual['seeds'][0]['evaluation'],expected['seeds'][0]['evaluation'])
            a,ao,am=load_checkpoint(full/'seed-9871/checkpoints',digest(full/'inputs/manifest.json'))
            b,bo,bm=load_checkpoint(resumed/'seed-9871/checkpoints',digest(resumed/'inputs/manifest.json'))
            for aa,bb in zip(ao[:2],bo[:2]):
                self.assertEqual(parameter_hash(aa),parameter_hash(bb))
            self.assertEqual((bm['cohorts'],bm['next_episode']),(2,16))
            self.assertEqual(actual['seeds'][0]['training']['games'],16)
            self.assertEqual(len(list((resumed/'seed-9871/evaluation').glob('*.json'))),8)
            with self.assertRaisesRegex(ValueError,'Completed run'):
                campaign.run_campaign(resumed,plan,resume=True)


if __name__=='__main__': unittest.main()
