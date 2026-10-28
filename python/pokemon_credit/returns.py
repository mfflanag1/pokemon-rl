"""Independent forward-view lambda returns for explicitly linked chains."""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def _unit_scalar(value, name, reject_bool=False):
    if (reject_bool and isinstance(value, (bool, np.bool_))) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar in [0, 1]")
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite real scalar in [0, 1]")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def targets(rows, lam):
    """Return lambda targets and chain diagnostics in input order.

    Links, rather than trajectory labels alone, define chains. A cutoff tail
    uses only its recorded bootstrap; terminal outcomes are never inferred.
    Invalid rows or linkage raise ValueError. Inputs are never modified.
    """
    lam = _unit_scalar(lam, "lambda", reject_bool=True)
    rows = list(rows)
    n = len(rows)
    values = np.empty(n, dtype=np.float64)
    rewards = np.empty(n, dtype=np.float64)
    bootstraps = np.empty(n, dtype=np.float64)
    terminal = np.empty(n, dtype=np.bool_)
    links = []
    trajectories = []
    required = {"value", "reward", "next_index", "bootstrap", "terminal", "trajectory"}
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError(f"row {i} is missing required fields")
        values[i] = _unit_scalar(row["value"], "value")
        rewards[i] = _unit_scalar(row["reward"], "reward")
        bootstraps[i] = _unit_scalar(row["bootstrap"], "bootstrap")
        if not isinstance(row["terminal"], (bool, np.bool_)):
            raise ValueError("terminal must be bool")
        terminal[i] = row["terminal"]
        nxt = _integer(row["next_index"], "next_index")
        trajectory = _integer(row["trajectory"], "trajectory")
        if trajectory < 0:
            raise ValueError("trajectory must be nonnegative")
        if nxt != -1 and not i < nxt < n:
            raise ValueError("next_index must be -1 or a strictly later valid row")
        if terminal[i] and (nxt != -1 or bootstraps[i] != 0):
            raise ValueError("terminal tails require next_index=-1 and bootstrap=0")
        if nxt != -1 and bootstraps[i] != 0:
            raise ValueError("linked rows require bootstrap=0")
        if not terminal[i] and rewards[i] != 0:
            raise ValueError("only terminal tails may have nonzero reward")
        links.append(nxt)
        trajectories.append(trajectory)

    predecessors = set()
    for i, nxt in enumerate(links):
        if nxt == -1:
            continue
        if trajectories[i] != trajectories[nxt]:
            raise ValueError("links must stay within a trajectory")
        if nxt in predecessors:
            raise ValueError("merged branches are not permitted")
        predecessors.add(nxt)

    returns = np.empty(n, dtype=np.float64)
    terminal_chain = np.empty(n, dtype=np.bool_)
    distance = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        nxt = links[i]
        if nxt == -1:
            returns[i] = rewards[i] if terminal[i] else bootstraps[i]
            terminal_chain[i] = terminal[i]
            distance[i] = 0
        else:
            returns[i] = (1 - lam) * values[nxt] + lam * returns[nxt]
            terminal_chain[i] = terminal_chain[nxt]
            distance[i] = distance[nxt] + 1
    return {"returns": returns, "advantages": returns - values,
            "terminal_chain": terminal_chain, "distance_to_end": distance}
