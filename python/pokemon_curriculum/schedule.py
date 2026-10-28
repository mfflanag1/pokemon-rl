"""Deterministic F12 row quotas and train-only sampling (standard library only)."""

import math
import random
from collections.abc import Mapping
from numbers import Real


def _integer(name, value, maximum=None):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def quotas(arm, update):
    """Return the exact selected-row quotas for update 0 through 47."""
    _integer("update", update, 47)
    if arm not in ("official", "curriculum"):
        raise ValueError("unknown arm")
    prefix = 0 if arm == "official" else (102, 205, 256)[update // 16]
    return {"official": 1024 - prefix, "prefix": prefix}


def choose_scenario(bank, weights, seed):
    """Choose a weighted family, then a uniform train-only content hash.

    Bank entries have a nonempty list of distinct families, content_hash, split,
    and a start mapping. Each hash participates once in every listed family.
    Zero-weight families are permitted but never selected. Cryptographic admission
    belongs to the coordinator; this function reads no outcomes or value fields.
    """
    _integer("seed", seed, 2**32 - 1)
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a nonempty family mapping")
    positive = {}
    for family, weight in weights.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("family must be a nonempty string")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("weights must be finite nonnegative numbers")
        try:
            weight = float(weight)
        except (OverflowError, ValueError) as exc:
            raise ValueError("weights must be finite") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
        if weight > 0:
            positive[family] = weight
    if not positive:
        raise ValueError("at least one positive family weight is required")

    families = {}
    seen = set()
    for entry in bank:
        if not isinstance(entry, Mapping):
            raise ValueError("bank entries must be mappings")
        entry_families = entry.get("families")
        content_hash = entry.get("content_hash")
        start = entry.get("start")
        if (not isinstance(entry_families, list) or not entry_families
                or any(not isinstance(family, str) or not family.strip()
                       for family in entry_families)
                or len(set(entry_families)) != len(entry_families)):
            raise ValueError("bank families must be a nonempty list of distinct nonempty strings")
        if not isinstance(content_hash, str) or not content_hash.strip():
            raise ValueError("content_hash must be a nonempty string")
        if content_hash in seen:
            raise ValueError("duplicate content_hash")
        if (entry.get("split") != "train" or not isinstance(start, Mapping)
                or start.get("split") != "train" or start.get("swap") is not False):
            raise ValueError("bank entries must be unswapped train starts")
        seen.add(content_hash)
        for family in entry_families:
            families.setdefault(family, []).append(content_hash)
    if not seen:
        raise ValueError("bank must be nonempty")
    if positive.keys() - families.keys():
        raise ValueError("positive-weight family missing from bank")

    ordered = sorted(positive)
    scale = max(positive.values())
    scaled = [positive[family] / scale for family in ordered]
    total = math.fsum(scaled)
    rng = random.Random(seed)
    threshold = rng.random()
    cumulative = 0.0
    chosen = ordered[-1]
    for family, weight in zip(ordered, scaled):
        cumulative += weight / total
        if threshold < cumulative:
            chosen = family
            break
    return chosen, rng.choice(sorted(families[chosen]))


def select_indices(official_n, prefix_n, prefix_quota, seed):
    """Sample each stratum without replacement, then shuffle all 1024 rows."""
    _integer("seed", seed, 2**32 - 1)
    _integer("official_n", official_n)
    _integer("prefix_n", prefix_n)
    _integer("prefix_quota", prefix_quota, 1024)
    official_quota = 1024 - prefix_quota
    if official_n < official_quota or prefix_n < prefix_quota:
        raise ValueError("insufficient rows for stratum quota")
    rng = random.Random(seed)
    selected = [("official", i) for i in rng.sample(range(official_n), official_quota)]
    selected.extend(("prefix", i) for i in rng.sample(range(prefix_n), prefix_quota))
    rng.shuffle(selected)
    return selected
