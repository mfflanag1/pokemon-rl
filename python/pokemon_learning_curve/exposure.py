"""Pure accounting of teacher and rehearsal row exposure."""

from numbers import Integral

import numpy as np


FAMILIES = ("normal_move", "tera_move", "voluntary_switch", "forced_switch")


class Exposure:
    def __init__(self, labels, old_count=128):
        if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int32):
            raise ValueError("labels must be an int32 ndarray")
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be nonempty and one-dimensional")
        if not np.array_equal(np.unique(labels), np.arange(4)):
            raise ValueError("labels must contain exactly families 0, 1, 2, 3")
        if type(old_count) is not int or old_count <= 0:
            raise ValueError("old_count must be a positive Python int")
        self._labels = labels.copy()
        self._teacher = np.zeros(labels.size, dtype=np.int64)
        self._old = np.zeros(old_count, dtype=np.int64)
        self._updates = 0

    @staticmethod
    def _indices(values, size):
        if not isinstance(values, (list, np.ndarray)):
            raise ValueError("indices must be a list or ndarray")
        array = np.asarray(values)
        if array.shape != (64,):
            raise ValueError("indices must have shape (64,)")
        # Check original list entries before NumPy can coerce mixed bool/int data.
        entries = values if isinstance(values, list) else array
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
               for x in entries):
            raise ValueError("indices must be integers, excluding booleans")
        if any(x < 0 or x >= size for x in entries):
            raise ValueError("index out of range")
        return np.array(entries, dtype=np.intp, copy=True)

    def add(self, teacher_indices, old_indices, optimizer_step):
        if type(optimizer_step) is not int or optimizer_step != self._updates + 1:
            raise ValueError("optimizer_step must be the next Python integer")
        teacher = self._indices(teacher_indices, self._labels.size)
        old = self._indices(old_indices, self._old.size)
        if not np.array_equal(np.bincount(self._labels[teacher], minlength=4),
                              [32, 16, 8, 8]):
            raise ValueError("teacher family quotas must be 32/16/8/8")
        np.add.at(self._teacher, teacher, 1)
        np.add.at(self._old, old, 1)
        self._updates += 1

    @staticmethod
    def _stats(counts):
        seen = int(np.count_nonzero(counts))
        return {"rows": int(counts.size), "seen_rows": seen,
                "unseen_rows": int(counts.size) - seen,
                "draws": int(counts.sum()),
                "min_draws_per_row": int(counts.min()),
                "max_draws_per_row": int(counts.max())}

    def summary(self):
        return {"updates": self._updates,
                "teacher_draws": int(self._teacher.sum()),
                "rehearsal_draws": int(self._old.sum()),
                "families": {name: self._stats(self._teacher[self._labels == i])
                             for i, name in enumerate(FAMILIES)},
                "rehearsal": self._stats(self._old)}
