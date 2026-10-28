"""Deterministic diagnostic gates for the decision-retention contract."""

import math


_ARMS = {"fresh", "supervised"}
_CELLS = tuple(f"{panel}/{i}" for panel in ("train", "variant") for i in range(16))
_METRICS = (
    "target_mass", "greedy_accuracy", "sampled_accuracy", "mean_reward",
    "local_value_mse",
)
_THRESHOLDS = (
    "minimum_greedy_accuracy", "minimum_context_mass",
    "maximum_context_mass_drop", "minimum_family_improvement",
    "minimum_family_advantage",
)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _unit(value, label):
    if (type(value) not in (int, float) or not 0 <= value <= 1
            or not math.isfinite(value)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")


def compare(initial, final, plan):
    """Compare balanced endpoint dictionaries using exact float thresholds.

    Each endpoint is ``{arm: {"cells": {"train/0": cell, ...}}}``.
    All 32 cells in both arms and endpoints must share one positive integer n.
    Raises ValueError for malformed inputs. Gates are diagnostics, not tests of
    statistical significance. Inputs are never modified.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("thresholds"), dict):
        raise ValueError("plan must contain a thresholds dictionary")
    thresholds = plan["thresholds"]
    for name in _THRESHOLDS:
        if name not in thresholds:
            raise ValueError(f"missing threshold {name}")
        _unit(thresholds[name], name)

    common_n = None
    for label, endpoint in (("initial", initial), ("final", final)):
        _keys(endpoint, _ARMS, label)
        for arm in ("fresh", "supervised"):
            _keys(endpoint[arm], {"cells"}, f"{label}/{arm}")
            cells = endpoint[arm]["cells"]
            _keys(cells, _CELLS, f"{label}/{arm}/cells")
            for key in _CELLS:
                cell = cells[key]
                path = f"{label}/{arm}/{key}"
                _keys(cell, {"n", *_METRICS}, path)
                n = cell["n"]
                if type(n) is not int or n <= 0:
                    raise ValueError(f"{path}/n must be a positive integer")
                if common_n is None:
                    common_n = n
                if n != common_n:
                    raise ValueError("n must be identical across all cells and endpoints")
                for metric in _METRICS:
                    _unit(cell[metric], f"{path}/{metric}")

    cells = {}
    for key in _CELLS:
        si = float(initial["supervised"]["cells"][key]["target_mass"])
        sf = float(final["supervised"]["cells"][key]["target_mass"])
        fi = float(initial["fresh"]["cells"][key]["target_mass"])
        ff = float(final["fresh"]["cells"][key]["target_mass"])
        gain = sf - si
        cells[key] = {
            "supervised_gain": gain,
            "fresh_gain": ff - fi,
            "final_advantage": sf - ff,
            "retained": bool(
                final["supervised"]["cells"][key]["greedy_accuracy"]
                >= thresholds["minimum_greedy_accuracy"]
                and sf >= thresholds["minimum_context_mass"]
                and gain >= -thresholds["maximum_context_mass_drop"]
            ),
        }
    families = {}
    for panel in ("train", "variant"):
        for family, start in (("switch", 0), ("tera", 8)):
            members = [cells[f"{panel}/{i}"] for i in range(start, start + 8)]
            gain = sum(c["supervised_gain"] for c in members) / 8
            advantage = sum(c["final_advantage"] for c in members) / 8
            families[f"{panel}/{family}"] = {
                "supervised_gain": gain,
                "final_advantage": advantage,
                "improved": bool(gain >= thresholds["minimum_family_improvement"]),
                "advantage": bool(advantage >= thresholds["minimum_family_advantage"]),
            }
    return {
        "retained": all(c["retained"] for c in cells.values()),
        "improved": all(f["improved"] for f in families.values()),
        "advantage": all(f["advantage"] for f in families.values()),
        "cells": cells,
        "families": families,
    }
