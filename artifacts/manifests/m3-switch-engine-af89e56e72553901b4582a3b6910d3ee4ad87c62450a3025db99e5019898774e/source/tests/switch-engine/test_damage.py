import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from pokemon_switch_engine.damage import damage, stats, max_pp


class DamageTests(unittest.TestCase):
    def test_sealed_reference_grid(self):
        rows = json.loads((Path(__file__).parent / 'fixtures/damage.json').read_text())
        count = 0
        for row in rows:
            for i, hp in enumerate(row['hp']):
                for move in range(2):
                    for crit in range(2):
                        for roll in range(16):
                            self.assertEqual(damage(row['attacker'], row['defender'], hp, move, bool(crit), roll), row['values'][i][move][crit][roll])
                            count += 1
        self.assertEqual(count, 3456)

    def test_stats_and_pp(self):
        self.assertEqual(stats('bulbasaur', False), [120, 69, 69, 85, 85, 65])
        self.assertEqual(stats('squirtle', True), [119, 68, 85, 70, 84, 95])
        self.assertEqual(stats('fennekin', True), [115, 65, 60, 82, 80, 112])
        self.assertEqual(max_pp('fennekin'), [24, 56])

    def test_rejects_bad_profile_arguments(self):
        for args in [('ditto','squirtle',100,0,False,0), ('bulbasaur','squirtle',0,0,False,0),
                     ('bulbasaur','squirtle',121,0,False,0), ('bulbasaur','squirtle',100,2,False,0),
                     ('bulbasaur','squirtle',100,True,False,0), ('bulbasaur','squirtle',100,0,1,0),
                     ('bulbasaur','squirtle',100,0,False,16)]:
            with self.assertRaises((TypeError, ValueError)): damage(*args)
        with self.assertRaises((TypeError, ValueError)): stats('squirtle', 1)

if __name__ == '__main__': unittest.main()
