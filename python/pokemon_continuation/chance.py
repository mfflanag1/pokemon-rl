"""Same event addresses and word laws; fresh continuation state type."""
from pokemon_switch_engine.chance import sample as original, DOMAIN
from .engine import Chance


def sample(master,game_ids,groups):
    return Chance(*original(master,game_ids,groups))
