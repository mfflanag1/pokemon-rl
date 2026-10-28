"""Eight directly runnable checks for independent exposure accounting."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from pokemon_learning_curve.exposure import Exposure


def labels():
    return np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)


def teacher(offset=0):
    return [offset] * 32 + [2 + offset] * 16 + [4 + offset] * 8 + [6 + offset] * 8


def stats(rows, seen, draws, minimum, maximum):
    return dict(rows=rows, seen_rows=seen, unseen_rows=rows-seen,
                draws=draws, min_draws_per_row=minimum, max_draws_per_row=maximum)


class ExposureChecks(unittest.TestCase):
    def test_two_update_totals(self):
        exposure = Exposure(labels(), 4)
        exposure.add(teacher(), [0] * 48 + [1] * 16, 1)
        exposure.add(teacher(1), [1] * 32 + [2] * 32, 2)
        self.assertEqual(exposure.summary(), {
            "updates": 2, "teacher_draws": 128, "rehearsal_draws": 128,
            "families": {"normal_move": stats(2, 2, 64, 32, 32),
                         "tera_move": stats(2, 2, 32, 16, 16),
                         "voluntary_switch": stats(2, 2, 16, 8, 8),
                         "forced_switch": stats(2, 2, 16, 8, 8)},
            "rehearsal": stats(4, 3, 128, 0, 48)})

    def test_repeated_indices(self):
        exposure = Exposure(labels(), 3)
        self.assertEqual(exposure.summary()["rehearsal"], stats(3, 0, 0, 0, 0))
        exposure.add(np.array(teacher()), np.zeros(64, dtype=np.int32), 1)
        result = exposure.summary()
        for name, draws in [("normal_move", 32), ("tera_move", 16),
                            ("voluntary_switch", 8), ("forced_switch", 8)]:
            self.assertEqual(result["families"][name], stats(2, 1, draws, 0, draws))
        self.assertEqual(result["rehearsal"], stats(3, 1, 64, 0, 64))

    def test_constructor_schema(self):
        bad_labels = [[], [0, 1, 2, 3], np.array([], dtype=np.int32),
                      labels().astype(np.int64), labels().reshape(2, 4),
                      np.array([0, 1, 2], dtype=np.int32),
                      np.array([-1, 0, 1, 2, 3], dtype=np.int32),
                      np.array([0, 1, 2, 3, 4], dtype=np.int32)]
        for value in bad_labels:
            with self.subTest(labels=value), self.assertRaises(ValueError):
                Exposure(value)
        for count in [0, -1, True, 4.0, np.int64(4)]:
            with self.subTest(old_count=count), self.assertRaises(ValueError):
                Exposure(labels(), count)
        self.assertEqual(Exposure(labels()).summary()["rehearsal"]["rows"], 128)

    def test_index_validation(self):
        for side, limit in [("teacher", 8), ("old", 4)]:
            valid = teacher() if side == "teacher" else [0] * 64
            invalid = [valid[:-1], valid + [0], np.array(valid).reshape(8, 8),
                       tuple(valid), np.array(valid, dtype=float)]
            for entry in [-1, limit, True, np.bool_(False), 0.0, "0"]:
                invalid.append([entry] + valid[1:])
            for value in invalid:
                with self.subTest(side=side, value=value):
                    exposure = Exposure(labels(), 4)
                    with self.assertRaises(ValueError):
                        exposure.add(value if side == "teacher" else teacher(),
                                     value if side == "old" else [0] * 64, 1)

    def test_family_quotas(self):
        for value in [[0] * 64, [0] * 31 + [2] * 17 + [4] * 8 + [6] * 8,
                      [0] * 32 + [2] * 16 + [4] * 7 + [6] * 9]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Exposure(labels()).add(value, [0] * 64, 1)

    def test_step_validation(self):
        exposure = Exposure(labels())
        for step in [0, 2, -1, True, 1.0, np.int64(1)]:
            with self.subTest(step=step), self.assertRaises(ValueError):
                exposure.add(teacher(), [0] * 64, step)
        exposure.add(teacher(), [0] * 64, 1)
        for step in [1, 3, False]:
            with self.subTest(step=step), self.assertRaises(ValueError):
                exposure.add(teacher(), [0] * 64, step)
        exposure.add(teacher(), [0] * 64, 2)
        self.assertEqual(exposure.summary()["updates"], 2)

    def test_failed_updates_atomic(self):
        exposure = Exposure(labels(), 4)
        exposure.add(teacher(), [0] * 64, 1)
        before = exposure.summary()
        for teach, old, step in [([-1] * 64, [1] * 64, 2),
                                 ([0] * 64, [1] * 64, 2),
                                 (teacher(1), [4] * 64, 2),
                                 (teacher(1), [1] * 64, 3)]:
            with self.subTest(step=step, old=old[0]):
                with self.assertRaises(ValueError):
                    exposure.add(teach, old, step)
                self.assertEqual(exposure.summary(), before)
        exposure.add(teacher(1), [1] * 64, 2)
        self.assertEqual(exposure.summary()["teacher_draws"], 128)

    def test_input_output_isolation(self):
        source = labels()
        exposure = Exposure(source, 4)
        source[:] = 3
        teach, old = teacher(), np.zeros(64, dtype=np.int32)
        exposure.add(teach, old, 1)
        self.assertEqual(teach, teacher())
        np.testing.assert_array_equal(old, np.zeros(64, dtype=np.int32))
        expected = exposure.summary()
        teach[:] = [7] * 64
        old[:] = 3
        result = exposure.summary()
        result["families"]["normal_move"]["draws"] = -1
        result["rehearsal"].clear()
        result["updates"] = 99
        self.assertEqual(exposure.summary(), expected)
        self.assertEqual(json.loads(json.dumps(expected)), expected)


if __name__ == "__main__":
    unittest.main()
