"""F37 additive legal-root selector for the Pokemon teacher repair.

This module is a standalone, stdlib-only reimplementation of the pinned Foul
Play root policy aggregation (``fp/search/main.py::select_move_from_mcts_results``)
with an added legality gate. It imports nothing from this project and nothing
from Foul Play / poke-engine, so it can be unit tested without a game, a native
search, a network or a subprocess.

Pinned native aggregation that must be matched bit-for-bit (evidence:
``artifacts/external/foul-play/fp/search/main.py`` lines 16-44)::

    final_policy[choice] = final_policy.get(choice, 0) + (chance * (visits / total_visits))
    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
    final_policy = [i for i in final_policy if i[1] >= final_policy[0][1] * 0.75]
    choice = random.choices(final_policy, weights=[p[1] for p in final_policy])[0]

The only behavioural differences are additive and audited:

* illegal candidates are removed *before* the 75%-of-highest threshold, so the
  threshold is taken over the highest *legal* mass (never over an illegal mass);
* only strictly positive mass is retained;
* if nothing survives, no RNG draw happens at all and a
  :class:`LegalSelectionError` carrying the full audit is raised instead of any
  silent repair, renormalisation or arbitrary fallback.

Per-world masses are never renormalised over a world's legal subset: the
aggregation and the resulting weights are exactly the native ones, which is why
a run where every candidate is legal reproduces the native outcome and the
native RNG consumption for the same seeded RNG.

No singleton optimisation lives here: the caller is responsible for answering a
sole legal action without running a search. This selector always performs
exactly one ``rng.choices`` draw when at least one candidate is retained.
"""

from __future__ import annotations

import math

__all__ = [
    "LegalSelectionError",
    "select_legal_from_mcts_results",
    "MASK_SIZE",
    "THRESHOLD_FRACTION",
    "AUDIT_SCHEMA",
]

#: Number of strict booleans expected in ``legal_mask`` (4 moves, 4 tera moves,
#: 6 switch slots), and therefore the valid half-open action range [0, 14).
MASK_SIZE = 14

#: Pinned native "close to the best move" fraction.
THRESHOLD_FRACTION = 0.75

AUDIT_SCHEMA = "f37-legal-selection-v1"

_MISSING = object()


class LegalSelectionError(ValueError):
    """Raised when no legal, positive-mass root candidate can be selected.

    Also raised for malformed input. The ``audit`` attribute is always a
    JSON-serializable ``dict`` describing the evidence for the rejection.
    """

    def __init__(self, message, audit):
        super().__init__(message)
        self.message = str(message)
        self.audit = audit


def _reject(failure, message, evidence):
    raise LegalSelectionError(
        message,
        {
            "schema": AUDIT_SCHEMA,
            "outcome": "rejected",
            "failure": failure,
            "message": str(message),
            "evidence": evidence,
        },
    )


def _is_real(value):
    """True for a real (non-boolean) Python number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_real(value):
    """Reject magnitudes that Python cannot represent as a finite float."""
    if not _is_real(value):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _finite_sum(values, field):
    total = sum(values)
    if not _finite_real(total):
        _reject("weight_sum_not_finite", "Summed root weight is not finite",
                {"field": field, "total": repr(total)})
    return total


def _describe(value):
    """Short JSON-serializable description of an arbitrary value."""
    try:
        description = repr(value)[:120]
    except (ValueError, OverflowError):
        description = "<magnitude cannot be represented>"
    return {"type": type(value).__name__, "repr": description}


def _validate_mask(legal_mask):
    if not isinstance(legal_mask, (list, tuple)):
        _reject(
            "mask_not_sequence",
            "legal_mask must be a list or tuple of booleans",
            {"legal_mask": _describe(legal_mask)},
        )
    mask = list(legal_mask)
    if len(mask) != MASK_SIZE:
        _reject(
            "mask_wrong_length",
            "legal_mask must contain exactly {} booleans".format(MASK_SIZE),
            {"length": len(mask), "expected_length": MASK_SIZE},
        )
    for position, bit in enumerate(mask):
        if type(bit) is not bool:
            _reject(
                "mask_not_strict_bool",
                "legal_mask[{}] is not a strict bool".format(position),
                {"index": position, "value": _describe(bit)},
            )
    if not any(mask):
        _reject(
            "mask_all_false",
            "legal_mask must contain at least one legal action",
            {"true_count": 0, "length": len(mask)},
        )
    return mask


def _validate_rng(rng):
    if rng is None or not callable(getattr(rng, "choices", None)):
        _reject(
            "rng_without_choices",
            "rng must provide a callable choices(population, weights=...)",
            {"rng": _describe(rng)},
        )


def _world_entry(entry, position):
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        _reject(
            "result_entry_shape",
            "results[{}] must be a (result, chance, index) triple".format(position),
            {"index": position, "entry": _describe(entry)},
        )
    result, chance, index = entry[0], entry[1], entry[2]
    if type(index) is not int:
        _reject(
            "world_index_not_int",
            "results[{}] index is not an int".format(position),
            {"index": position, "world_index": _describe(index)},
        )
    if not _finite_real(chance) or chance < 0:
        _reject(
            "chance_invalid",
            "results[{}] sample chance must be a finite number >= 0".format(position),
            {"index": position, "world_index": index, "chance": _describe(chance)},
        )
    total_visits = getattr(result, "total_visits", _MISSING)
    if total_visits is _MISSING:
        _reject(
            "result_missing_total_visits",
            "results[{}] result has no total_visits".format(position),
            {"index": position, "world_index": index, "result": _describe(result)},
        )
    if not _finite_real(total_visits) or total_visits <= 0:
        _reject(
            "total_visits_invalid",
            "results[{}] total_visits must be a finite number > 0".format(position),
            {
                "index": position,
                "world_index": index,
                "total_visits": _describe(total_visits),
            },
        )
    side_one = getattr(result, "side_one", _MISSING)
    if side_one is _MISSING:
        _reject(
            "result_missing_side_one",
            "results[{}] result has no side_one".format(position),
            {"index": position, "world_index": index, "result": _describe(result)},
        )
    if not isinstance(side_one, (list, tuple)):
        _reject(
            "side_one_not_sequence",
            "results[{}] side_one must be a list or tuple".format(position),
            {"index": position, "world_index": index, "side_one": _describe(side_one)},
        )
    if len(side_one) == 0:
        _reject(
            "side_one_empty",
            "results[{}] side_one is empty".format(position),
            {"index": position, "world_index": index},
        )
    options = []
    for option_position, option in enumerate(side_one):
        move_choice = getattr(option, "move_choice", _MISSING)
        if type(move_choice) is not str or not move_choice:
            _reject(
                "move_choice_invalid",
                "results[{}] side_one[{}] move_choice must be a non-empty str".format(
                    position, option_position
                ),
                {
                    "index": position,
                    "world_index": index,
                    "option_index": option_position,
                    "move_choice": _describe(
                        None if move_choice is _MISSING else move_choice
                    ),
                },
            )
        visits = getattr(option, "visits", _MISSING)
        if not _finite_real(visits) or visits < 0:
            _reject(
                "visits_invalid",
                "results[{}] side_one[{}] visits must be a finite number >= 0".format(
                    position, option_position
                ),
                {
                    "index": position,
                    "world_index": index,
                    "option_index": option_position,
                    "move_choice": move_choice,
                    "visits": _describe(None if visits is _MISSING else visits),
                },
            )
        options.append((move_choice, visits))
    return index, chance, total_visits, options


def _aggregate(results):
    """Replicate the pinned native policy accumulation exactly.

    Returns ``(aggregated, worlds)`` where ``aggregated`` is an insertion
    ordered ``dict`` of choice -> mass.
    """
    if not isinstance(results, (list, tuple)):
        _reject(
            "results_not_sequence",
            "results must be a list of (result, chance, index) triples",
            {"results": _describe(results)},
        )
    if len(results) == 0:
        _reject(
            "results_empty",
            "results must contain at least one searched world",
            {"length": 0},
        )
    aggregated = {}
    worlds = []
    for position, entry in enumerate(results):
        index, chance, total_visits, options = _world_entry(entry, position)
        worlds.append(
            {
                "position": position,
                "index": index,
                "chance": float(chance),
                "total_visits": float(total_visits),
                "option_count": len(options),
            }
        )
        for move_choice, visits in options:
            # Identical expression, order and accumulator seed as the pinned
            # native selector, so float rounding matches bit-for-bit.
            try:
                aggregated[move_choice] = aggregated.get(move_choice, 0) + (
                    chance * (visits / total_visits)
                )
            except (OverflowError, ZeroDivisionError) as exc:
                _reject("mass_arithmetic_overflow", "Root mass arithmetic is not representable",
                        {"choice": move_choice, "error": str(exc)})
    for move_choice, mass in aggregated.items():
        if not math.isfinite(mass):
            _reject(
                "aggregated_mass_not_finite",
                "aggregated mass for {!r} is not finite".format(move_choice),
                {"choice": move_choice, "mass": repr(mass)},
            )
    _finite_sum(aggregated.values(), "aggregated_mass")
    return aggregated, worlds


def _classify(aggregated, legal_mask, action_for_choice):
    """Map every aggregated choice exactly once and record legality."""
    candidates = []
    for insertion_index, (choice, mass) in enumerate(aggregated.items()):
        record = {
            "choice": choice,
            "insertion_index": insertion_index,
            "mass": float(mass),
            "action": None,
            "allowed": True,
            "reason": None,
        }
        try:
            action = action_for_choice(choice)
        except ValueError as exc:
            record["allowed"] = False
            record["reason"] = "unmappable_choice"
            record["detail"] = str(exc)
            candidates.append(record)
            continue
        if isinstance(action, bool):
            record["allowed"] = False
            record["reason"] = "boolean_action"
            record["detail"] = _describe(action)
        elif not isinstance(action, int):
            record["allowed"] = False
            record["reason"] = "non_integer_action"
            record["detail"] = _describe(action)
        elif not 0 <= action < MASK_SIZE:
            record["action"] = action
            record["allowed"] = False
            record["reason"] = "action_out_of_range"
            record["detail"] = {"action": action, "mask_size": MASK_SIZE}
        else:
            record["action"] = action
            if not legal_mask[action]:
                record["allowed"] = False
                record["reason"] = "illegal_action"
        candidates.append(record)
    return candidates


def select_legal_from_mcts_results(results, legal_mask, action_for_choice, *, rng):
    """Select one legal root decision from per-world MCTS results.

    Args:
        results: list of ``(result, chance, index)`` triples. ``result`` exposes
            ``total_visits`` and ``side_one``; each ``side_one`` entry exposes
            ``move_choice`` (str) and ``visits`` (number).
        legal_mask: exactly ``MASK_SIZE`` strict bools, at least one ``True``.
        action_for_choice: callable mapping a choice string to an ``int`` action;
            it may raise ``ValueError`` for a candidate that the current request
            cannot express.
        rng: object providing ``choices(population, weights=...)`` (the
            ``random`` module or a ``random.Random`` instance).

    Returns:
        ``{"decision": str, "audit": dict}``.

    Raises:
        LegalSelectionError: for malformed input, or when no legal candidate
            with positive mass survives. No RNG draw is performed in that case.
    """
    legal_mask = _validate_mask(legal_mask)
    _validate_rng(rng)
    if not callable(action_for_choice):
        _reject(
            "action_for_choice_not_callable",
            "action_for_choice must be callable",
            {"action_for_choice": _describe(action_for_choice)},
        )

    aggregated, worlds = _aggregate(results)
    candidates = _classify(aggregated, legal_mask, action_for_choice)

    # Same stable descending ordering as the pinned native selector: equal
    # masses keep their aggregation insertion order.
    ranked = sorted(candidates, key=lambda record: record["mass"], reverse=True)
    legal = [record for record in ranked if record["allowed"]]

    audit = {
        "schema": AUDIT_SCHEMA,
        "outcome": "rejected",
        "mask_true_count": sum(1 for bit in legal_mask if bit),
        "worlds": worlds,
        "candidates": ranked,
        "aggregated_count": len(ranked),
        "aggregated_mass": float(_finite_sum((record["mass"] for record in ranked), "aggregated_mass")),
        "legal_count": len(legal),
        "highest_legal_mass": None,
        "threshold_fraction": THRESHOLD_FRACTION,
        "threshold": None,
        "retained": [],
        "retained_count": 0,
        "retained_mass": 0.0,
        "excluded_count": len(ranked),
        "excluded_mass": float(_finite_sum((record["mass"] for record in ranked), "aggregated_mass")),
        "decision": None,
    }

    if not legal:
        audit["failure"] = "no_legal_candidate"
        raise LegalSelectionError(
            "no legal root candidate among {} aggregated choices".format(len(ranked)),
            audit,
        )

    # Legality is applied first, so the threshold is a fraction of the highest
    # LEGAL mass, never of an illegal mass.
    highest_legal_mass = legal[0]["mass"]
    threshold = highest_legal_mass * THRESHOLD_FRACTION
    audit["highest_legal_mass"] = float(highest_legal_mass)
    audit["threshold"] = float(threshold)

    retained = []
    for record in legal:
        if record["mass"] <= 0:
            record["allowed"] = False
            record["reason"] = "non_positive_mass"
        elif record["mass"] < threshold:
            record["allowed"] = False
            record["reason"] = "below_threshold"
        else:
            retained.append(record)

    audit["retained"] = [
        {
            "choice": record["choice"],
            "action": record["action"],
            "mass": record["mass"],
        }
        for record in retained
    ]
    audit["retained_count"] = len(retained)
    audit["retained_mass"] = float(_finite_sum((record["mass"] for record in retained), "retained_mass"))
    retained_ids = {id(record) for record in retained}
    excluded = [record for record in ranked if id(record) not in retained_ids]
    audit["excluded_count"] = len(ranked) - len(retained)
    audit["excluded_mass"] = float(_finite_sum((record["mass"] for record in excluded), "excluded_mass"))

    if not retained:
        audit["failure"] = "no_positive_legal_mass"
        raise LegalSelectionError(
            "no legal root candidate with positive mass above threshold",
            audit,
        )

    population = [record["choice"] for record in retained]
    weights = [record["mass"] for record in retained]
    # Exactly one draw, with the native population order and native weights.
    drawn = rng.choices(population, weights=weights)
    try:
        decision = drawn[0]
    except (TypeError, IndexError, KeyError):
        _reject(
            "rng_choices_bad_return",
            "rng.choices did not return an indexable non-empty sample",
            {"drawn": _describe(drawn)},
        )
    if decision not in population:
        _reject(
            "rng_choices_foreign_choice",
            "rng.choices returned a value outside the retained candidates",
            {"drawn": _describe(decision), "retained": population},
        )

    audit["outcome"] = "selected"
    audit["decision"] = decision
    audit["decision_action"] = next(
        record["action"] for record in retained if record["choice"] == decision
    )
    return {"decision": decision, "audit": audit}
