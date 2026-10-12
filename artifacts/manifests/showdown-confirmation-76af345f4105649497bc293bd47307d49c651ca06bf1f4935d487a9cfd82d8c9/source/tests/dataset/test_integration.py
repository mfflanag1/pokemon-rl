"""R7 integration checks use the retained, independently validated small campaign."""
import copy
import gzip
from itertools import islice
from pathlib import Path
import shutil
import tempfile
import unittest

from pokemon_reference.contract import canonical, hash_json
from pokemon_reference.corpus import read_json, sealed, file_digest
from pokemon_reference.dataset_gate import ActorDataset, validate_collection
from pokemon_reference.player_history import load_actor_input

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/runs/r7-dataset-v1'
EXPOSURES = ROOT / 'artifacts/runs/r7-legacy-v1/registry.json'


def write(path, value): path.write_text(canonical(value) + '\n')

def reseal(doc): return sealed({k:v for k,v in doc.items() if k!='content_hash'})


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = read_json(DATA / 'dataset.json')

    def metadata_copy(self, root):
        index = read_json(DATA / 'index.json')
        for name in index['files']:
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DATA/name,dest)
        write(root/'index.json',index)
        return index

    def test_real_dataset_purpose_gate_and_policy_value_inputs(self):
        for purpose, split in [('training','train'),('development','development')]:
            dataset=ActorDataset(DATA, expected_hash=self.document['content_hash'], purpose=purpose)
            examples=list(islice(dataset.actors(split),3));self.assertEqual(len(examples),3)
            for actor in examples:
                self.assertEqual(actor.policy_inputs(),actor.value_inputs())
                self.assertNotIn('spec',actor.to_dict());self.assertNotIn('outcome',actor.to_dict())
            for forbidden in ['final','development' if split=='train' else 'train']:
                with self.assertRaisesRegex(ValueError,'Purpose/split'):next(dataset.actors(forbidden))

    def test_privileged_wire_and_rehashed_extra_field_are_rejected(self):
        row=self.document['actors'][0]
        body=read_json(DATA/'dataset.json')
        with self.assertRaises(Exception):load_actor_input(canonical(body))
        actor=__import__('json').loads(gzip.decompress((DATA/row['file']).read_bytes()).splitlines()[0])
        actor['rng_seed']='sodium,'+'0'*64
        actor['feature_id']=hash_json({k:v for k,v in actor.items() if k!='feature_id'})
        with self.assertRaises(Exception):load_actor_input(canonical(actor))
        record=read_json(DATA/'index.json')['records'][0]
        with self.assertRaises(Exception):load_actor_input(gzip.decompress((DATA/record['file']).read_bytes()))

    def test_resealed_missing_episode_rejects(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.metadata_copy(root);index['records'].pop();write(root/'index.json',reseal(index))
            with self.assertRaisesRegex(ValueError,'Missing/duplicate episodes'):validate_collection(root,EXPOSURES)

    def test_resealed_final_job_rejects(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.metadata_copy(root);jobs=read_json(root/'jobs.json')
            jobs['jobs'][0]['split']='final';write(root/'jobs.json',reseal(jobs))
            index['files']['jobs.json']=file_digest(root/'jobs.json');write(root/'index.json',reseal(index))
            with self.assertRaisesRegex(ValueError,'Pairing/seed/mirror'):validate_collection(root,EXPOSURES)

    def test_resealed_wrong_team_partition_rejects(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.metadata_copy(root);teams=read_json(root/'teams/train.json')
            teams['rows'][0]['split']='development';write(root/'teams/train.json',reseal(teams))
            index['files']['teams/train.json']=file_digest(root/'teams/train.json');write(root/'index.json',reseal(index))
            with self.assertRaisesRegex(ValueError,'Wrong team assignment'):validate_collection(root,EXPOSURES)
