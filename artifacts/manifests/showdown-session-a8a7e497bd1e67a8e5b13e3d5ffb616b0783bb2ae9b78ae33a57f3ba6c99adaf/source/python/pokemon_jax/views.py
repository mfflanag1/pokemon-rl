"""Strict, dependency-free reducer from an R1 PlayerProjection to the fixed-duel view v1.

See ``contracts/jax-slice-v1/view.md``. The reducer has no access to simulator state,
seeds, chance outcomes or the opponent's request: everything is reconstructed from the
public protocol lines, then cross-checked against the carrier's own request.

Every malformed, inconsistent or unsupported input raises ``ValueError``. The input is
never mutated.
"""

import json

__all__ = ["reduce_view"]

# ---------------------------------------------------------------------------
# Public profile constants (contracts/slice-v1/profile.json, gen9-rattata-duel-v1)
# ---------------------------------------------------------------------------
_WIRE_KEYS = frozenset({"side", "request_id", "turn", "request", "protocol"})
_SIDES = ("p1", "p2")
_INITIAL_HP = 105
_INITIAL_PP = (56, 48)
_MAX_TURNS = 4
_MAX_HITS = 8
_HISTORY_ROWS = 8
_HISTORY_COLS = 6
_MOVE_NAMES = {"Tackle": 0, "Quick Attack": 1}
_MOVE_IDS = ("tackle", "quickattack")
_SPECIES = "Rattata"
_SWITCH_DETAILS = "Rattata, L50, M"
_WINNER_NAMES = {"P1": "p1", "P2": "p2"}
# Metadata commands that carry no public game event for this projection.
_IGNORED_COMMANDS = frozenset(
    {"", "t:", "gametype", "player", "gen", "tier", "rule", "raw", "teamsize", "start", "upkeep"}
)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _other(side):
    return "p2" if side == "p1" else "p1"


def _ident_side(ident):
    """Map an exact active-slot ident ('p1a: Rattata') to its side, else raise."""
    for side in _SIDES:
        if ident == side + "a: " + _SPECIES:
            return side
    raise ValueError("unknown or malformed pokemon ident: %r" % (ident,))


class _Reducer:
    """Replays public protocol lines into reconstructed duel state."""

    def __init__(self):
        self.hp = {"p1": _INITIAL_HP, "p2": _INITIAL_HP}
        self.pp = {"p1": list(_INITIAL_PP), "p2": list(_INITIAL_PP)}
        self.switched = {"p1": False, "p2": False}
        self.fainted = {"p1": False, "p2": False}
        self.seen = {"p1": [0, 0], "p2": [0, 0]}
        self.turn = 0
        self.winner = None
        self.request_count = 0
        self.hits = []  # absolute rows: [turn, actor_side, move, crit, hp_lost, hp_after]
        self.pending = None  # [actor_side, target_side, move_index, crit]
        self.embedded_sides = []  # side ids named by embedded |request| payloads

    # -- helpers ------------------------------------------------------------
    @property
    def initialized(self):
        return self.switched["p1"] and self.switched["p2"]

    def _require_live_event(self, what):
        if self.winner is not None:
            raise ValueError("%s after winner declared" % what)
        if not self.initialized:
            raise ValueError("%s before both initial switches" % what)
        if self.turn < 1:
            raise ValueError("%s before turn 1" % what)

    # -- line dispatch ------------------------------------------------------
    def feed(self, line):
        if not isinstance(line, str):
            raise ValueError("protocol line is not a string")
        if line == "" or line == "|":
            return
        if not line.startswith("|"):
            raise ValueError("protocol line does not start with '|': %r" % (line,))
        parts = line.split("|")
        command = parts[1]
        args = parts[2:]
        if command == "request":
            self._on_request(line, args)
        elif command in _IGNORED_COMMANDS:
            if command == "" and args:
                raise ValueError("unknown protocol line: %r" % (line,))
            return
        elif command == "switch":
            self._on_switch(args)
        elif command == "turn":
            self._on_turn(args)
        elif command == "move":
            self._on_move(args)
        elif command == "-crit":
            self._on_crit(args)
        elif command == "-damage":
            self._on_damage(args)
        elif command == "faint":
            self._on_faint(args)
        elif command == "win":
            self._on_win(args)
        else:
            raise ValueError("unknown or unsupported protocol command: %r" % (command,))

    # -- handlers -----------------------------------------------------------
    def _on_request(self, line, args):
        # Transport metadata: count it and check side identity only; never treat
        # its contents as public events.
        self.request_count += 1
        payload = line[len("|request|"):]
        if payload == "":
            raise ValueError("missing embedded request")
        try:
            data = json.loads(payload)
        except ValueError:
            raise ValueError("malformed embedded request payload")
        if not isinstance(data, dict):
            raise ValueError("embedded request payload is not an object")
        side_info = data.get("side")
        if not isinstance(side_info, dict) or side_info.get("id") not in _SIDES:
            raise ValueError("embedded request has malformed side identity")
        self.embedded_sides.append(side_info["id"])

    def _on_switch(self, args):
        if len(args) != 3:
            raise ValueError("malformed switch line")
        if self.winner is not None or self.turn >= 1:
            raise ValueError("switches after initialization are unsupported")
        side = _ident_side(args[0])
        if self.switched[side]:
            raise ValueError("duplicate initial switch for %s" % side)
        if args[1] != _SWITCH_DETAILS:
            raise ValueError("unsupported pokemon details: %r" % (args[1],))
        if args[2] != "%d/%d" % (_INITIAL_HP, _INITIAL_HP):
            raise ValueError("unexpected initial HP: %r" % (args[2],))
        self.switched[side] = True

    def _on_turn(self, args):
        if len(args) != 1 or not args[0].isdigit():
            raise ValueError("malformed turn line")
        if self.winner is not None:
            raise ValueError("turn line after winner declared")
        if not self.initialized:
            raise ValueError("turn line before both initial switches")
        if self.pending is not None:
            raise ValueError("turn line while a move awaits its damage line")
        number = int(args[0])
        if number != self.turn + 1:
            raise ValueError("turn lines must progress 1,2,...; got %d after %d" % (number, self.turn))
        if number > _MAX_TURNS:
            raise ValueError("turn %d exceeds profile maximum %d" % (number, _MAX_TURNS))
        self.turn = number

    def _on_move(self, args):
        if len(args) != 3:
            raise ValueError("malformed or unsupported move line")
        self._require_live_event("move")
        if self.pending is not None:
            raise ValueError("move line while a previous move awaits its damage line")
        actor = _ident_side(args[0])
        target = _ident_side(args[2])
        if target != _other(actor):
            raise ValueError("move must target the opposing active pokemon")
        if args[1] not in _MOVE_NAMES:
            raise ValueError("unsupported move: %r" % (args[1],))
        if self.hp[actor] <= 0 or self.hp[target] <= 0:
            raise ValueError("move involving a fainted pokemon")
        move = _MOVE_NAMES[args[1]]
        if self.pp[actor][move] <= 0:
            raise ValueError("move executed with no PP remaining")
        self.pp[actor][move] -= 1
        self.seen[actor][move] = 1
        self.pending = [actor, target, move, 0]

    def _on_crit(self, args):
        if len(args) != 1:
            raise ValueError("malformed critical-hit line")
        self._require_live_event("critical-hit line")
        if self.pending is None:
            raise ValueError("critical-hit line without an executing move")
        if self.pending[3]:
            raise ValueError("duplicate critical-hit line for one move")
        if _ident_side(args[0]) != self.pending[1]:
            raise ValueError("critical-hit line target does not match executing move")
        self.pending[3] = 1

    def _on_damage(self, args):
        if len(args) != 2:
            raise ValueError("malformed or unsupported damage line")
        self._require_live_event("damage line")
        if self.pending is None:
            raise ValueError("damage line without an executing move")
        actor, target, move, crit = self.pending
        if _ident_side(args[0]) != target:
            raise ValueError("damage line target does not match executing move")
        hp_after = self._parse_hp(args[1])
        hp_before = self.hp[target]
        if hp_after >= hp_before:
            raise ValueError("impossible HP change %d -> %d" % (hp_before, hp_after))
        if len(self.hits) >= _MAX_HITS:
            raise ValueError("more than %d executed hits" % _MAX_HITS)
        hp_lost = hp_before - hp_after  # capped at pre-hit HP by construction
        self.hp[target] = hp_after
        self.hits.append([self.turn, actor, move, crit, hp_lost, hp_after])
        self.pending = None

    @staticmethod
    def _parse_hp(text):
        if text == "0 fnt":
            return 0
        head, sep, tail = text.partition("/")
        if not sep or tail != str(_INITIAL_HP) or not head.isdigit():
            raise ValueError("malformed damage string: %r" % (text,))
        value = int(head)
        if value < 1 or value > _INITIAL_HP:
            raise ValueError("damage string out of range: %r" % (text,))
        return value

    def _on_faint(self, args):
        if len(args) != 1:
            raise ValueError("malformed faint line")
        self._require_live_event("faint line")
        if self.pending is not None:
            raise ValueError("faint line while a move awaits its damage line")
        side = _ident_side(args[0])
        if self.hp[side] != 0:
            raise ValueError("faint line for a pokemon that was not knocked out")
        if self.fainted[side]:
            raise ValueError("duplicate faint line")
        self.fainted[side] = True

    def _on_win(self, args):
        if len(args) != 1 or args[0] not in _WINNER_NAMES:
            raise ValueError("malformed winner line")
        if self.winner is not None:
            raise ValueError("duplicate winner line")
        self._require_live_event("winner line")
        if self.pending is not None:
            raise ValueError("winner line while a move awaits its damage line")
        winner = _WINNER_NAMES[args[0]]
        loser = _other(winner)
        if self.hp[loser] != 0 or not self.fainted[loser]:
            raise ValueError("winner declared without the opponent being knocked out")
        if self.hp[winner] <= 0:
            raise ValueError("winner declared for a fainted pokemon")
        self.winner = winner

    # -- end of protocol ----------------------------------------------------
    def finish(self):
        if not self.initialized:
            raise ValueError("missing initial switches")
        if self.turn < 1:
            raise ValueError("protocol has no turn line")
        if self.pending is not None:
            raise ValueError("protocol ends with a move awaiting its damage line")
        if self.winner is None:
            for side in _SIDES:
                if self.hp[side] <= 0 or self.fainted[side]:
                    raise ValueError("knockout without a winner line")


def _check_request_side(request, side):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    side_info = request.get("side")
    if not isinstance(side_info, dict):
        raise ValueError("request is missing side information")
    if side_info.get("id") != side:
        raise ValueError("request belongs to a different side")
    return side_info


def _validate_live_request(request, side, own_hp, own_pp):
    side_info = _check_request_side(request, side)
    for key in ("teamPreview", "forceSwitch", "wait"):
        if key in request:
            raise ValueError("unsupported request kind: %s" % key)
    pokemon = side_info.get("pokemon")
    if not isinstance(pokemon, list) or len(pokemon) != 1:
        raise ValueError("request must list exactly one own pokemon")
    mon = pokemon[0]
    if not isinstance(mon, dict) or mon.get("active") is not True:
        raise ValueError("own pokemon must be active")
    # Validate reported profile facts; nature/IVs/EVs are not request fields.
    expected = {"ident": side + ": Rattata", "details": _SWITCH_DETAILS,
                "ability": "runaway", "baseAbility": "runaway", "item": "",
                "moves": list(_MOVE_IDS), "teraType": "Normal", "terastallized": "",
                "stats": {"atk":76,"def":55,"spa":45,"spd":55,"spe":92},
                "commanding": False, "reviving": False}
    for key, value in expected.items():
        if key in mon and mon[key] != value:
            raise ValueError("unsupported own profile fact: " + key)
    if mon.get("condition") != "%d/%d" % (own_hp, _INITIAL_HP):
        raise ValueError("request condition disagrees with public own HP")
    active = request.get("active")
    if not isinstance(active, list) or len(active) != 1:
        raise ValueError("request must have exactly one active slot")
    slot = active[0]
    if not isinstance(slot, dict):
        raise ValueError("malformed active slot")
    if "canTerastallize" in slot:
        raise ValueError("terastallization is unsupported")
    moves = slot.get("moves")
    if not isinstance(moves, list) or len(moves) != 2:
        raise ValueError("request must offer exactly two moves")
    for index in range(2):
        move = moves[index]
        if not isinstance(move, dict):
            raise ValueError("malformed move entry")
        if move.get("id") != _MOVE_IDS[index]:
            raise ValueError("unexpected move order or identity")
        if not _is_int(move.get("pp")) or move["pp"] != own_pp[index]:
            raise ValueError("request PP disagrees with reconstructed PP")
        if move.get("maxpp") != _INITIAL_PP[index]:
            raise ValueError("unexpected maxpp")
        if move.get("disabled") is not False:
            raise ValueError("disabled moves are unsupported")


def reduce_view(view):
    """Reduce an R1 PlayerProjection to ``{'header': [...], 'history': [[...]*8]}``.

    Raises ``ValueError`` for malformed input or unsupported mechanics. ``view`` is
    not mutated.
    """
    if not isinstance(view, dict):
        raise ValueError("view must be an object")
    if set(view.keys()) != _WIRE_KEYS:
        raise ValueError("view must have exactly the keys %s" % sorted(_WIRE_KEYS))
    side = view["side"]
    if not isinstance(side, str) or side not in _SIDES:
        raise ValueError("side must be 'p1' or 'p2'")
    request_id = view["request_id"]
    if not _is_int(request_id) or request_id < 0:
        raise ValueError("request_id must be a nonnegative int")
    turn = view["turn"]
    if not _is_int(turn):
        raise ValueError("turn must be an int")
    protocol = view["protocol"]
    if not isinstance(protocol, list):
        raise ValueError("protocol must be a list of strings")

    reducer = _Reducer()
    for line in protocol:
        reducer.feed(line)
    reducer.finish()

    for embedded in reducer.embedded_sides:
        if embedded != side:
            raise ValueError("embedded request belongs to a different side")
    if reducer.request_count != request_id:
        raise ValueError("request_id disagrees with number of request lines")
    if reducer.turn != turn:
        raise ValueError("turn disagrees with last protocol turn")

    opponent = _other(side)
    own_hp = reducer.hp[side]
    own_pp = reducer.pp[side]
    terminal = reducer.winner is not None

    request = view["request"]
    if terminal:
        # A stale previous own request may be retained; ignore its PP/HP but still
        # reject a request for the other side.
        if request is not None:
            _check_request_side(request, side)
        relative_winner = 1 if reducer.winner == side else 2
        legal = [0, 0]
    else:
        _validate_live_request(request, side, own_hp, own_pp)
        relative_winner = 0
        legal = [1, 1]

    header = [
        int(reducer.turn),
        int(terminal),
        int(relative_winner),
        int(own_hp),
        int(reducer.hp[opponent]),
        int(own_pp[0]),
        int(own_pp[1]),
        int(reducer.seen[opponent][0]),
        int(reducer.seen[opponent][1]),
        int(legal[0]),
        int(legal[1]),
    ]

    history = []
    for hit_turn, actor, move, crit, hp_lost, hp_after in reducer.hits:
        history.append(
            [
                int(hit_turn),
                0 if actor == side else 1,
                int(move),
                int(crit),
                int(hp_lost),
                int(hp_after),
            ]
        )
    while len(history) < _HISTORY_ROWS:
        history.append([-1] * _HISTORY_COLS)

    return {"header": header, "history": history}
