from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
check_boundary=runpy.run_path(str(ROOT/'scripts/seal-switch-soak.py'))['check_boundary']


class Audit(unittest.TestCase):
    def test_final_checkpoint_brackets_missing_final_tick(self):
        item={'elapsed_seconds':16.001,'counters':{'updates':108}}
        rows=[{'elapsed_seconds':16.001,'counters':{'updates':108}}]
        check_boundary(item,rows,{'elapsed_seconds':20.001,'final_counters':{'updates':150}})
        with self.assertRaises(ValueError):check_boundary(item,rows,{'elapsed_seconds':20.001,'final_counters':{'updates':108}})

    def test_same_cycle_counters_must_match(self):
        item={'elapsed_seconds':600.001,'counters':{'updates':109}}
        rows=[{'elapsed_seconds':600.001,'counters':{'updates':108}},
              {'elapsed_seconds':660.001,'counters':{'updates':120}}]
        with self.assertRaises(ValueError):check_boundary(item,rows,{'elapsed_seconds':7200.001,'final_counters':{'updates':1000}})


if __name__=='__main__':unittest.main()
