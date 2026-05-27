"""
game.py — Core Nim (Misère variant) logic for the Trap Room challenge.

Rules:
  - There are up to 6 piles of coins, each with 0–12 coins.
  - Players alternate taking 1, 2, or 3 coins from a single pile.
  - The player who takes the LAST coin LOSES (misère convention).

This module provides:
  - Move encoding / decoding (pile_index × take_amount → single integer)
  - Legal move enumeration
  - Grundy-number-based optimal play (no search required)
  - Utility helpers shared by all dataset and model modules
"""

import numpy as np
from typing import List, Tuple, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PILES   = 6       # context length analogue — all states padded to this
MAX_COINS   = 12      # maximum coins per pile
TAKE_VALUES = [1, 2, 3]   # legal take amounts
N_MOVES     = MAX_PILES * len(TAKE_VALUES)   # 18 total move logits

# ── Move encoding ─────────────────────────────────────────────────────────────

def encode_move(pile: int, take: int) -> int:
    """Encode (pile_index, take_amount) as a single integer in [0, 17].

    Layout: pile 0 take 1→0, pile 0 take 2→1, pile 0 take 3→2,
            pile 1 take 1→3, ... pile 5 take 3→17.
    """
    assert 0 <= pile < MAX_PILES
    assert take in TAKE_VALUES
    return pile * len(TAKE_VALUES) + (take - 1)


def decode_move(move_id: int) -> Tuple[int, int]:
    """Decode a move integer back to (pile_index, take_amount)."""
    pile = move_id // len(TAKE_VALUES)
    take = (move_id % len(TAKE_VALUES)) + 1
    return pile, take


def legal_moves(piles: List[int]) -> List[int]:
    """Return list of encoded move IDs that are legal in this state.

    A move is legal iff pile[i] >= take_amount (can't take more than exists).
    Empty piles (0 coins) have no legal moves.
    """
    moves = []
    for i, coins in enumerate(piles):
        for t in TAKE_VALUES:
            if coins >= t:
                moves.append(encode_move(i, t))
    return moves


def apply_move(piles: List[int], move_id: int) -> List[int]:
    """Return new pile state after applying move_id. Does not mutate input."""
    pile, take = decode_move(move_id)
    new_piles = list(piles)
    new_piles[pile] -= take
    return new_piles


def is_terminal(piles: List[int]) -> bool:
    """True when all piles are empty (the player who just moved took the last coin)."""
    return all(c == 0 for c in piles)

# ── Grundy / Sprague-Grundy theory ────────────────────────────────────────────

def grundy_nim_value(pile_size: int, max_take: int = 3) -> int:
    """Grundy number for a single Nim pile with max_take per move.

    For takes of {1, 2, ..., k}, the Grundy number of pile n is n % (k+1).
    With max_take=3: G(0)=0, G(1)=1, G(2)=2, G(3)=3, G(4)=0, G(5)=1, ...
    This cycles with period 4.
    """
    return pile_size % (max_take + 1)


def game_grundy(piles: List[int]) -> int:
    """XOR of all pile Grundy values — the overall game Grundy number.

    = 0 → current player is in a P-position (previous player wins, i.e. YOU lose
          with optimal play from the opponent)
    ≠ 0 → current player is in an N-position (Next player wins with optimal play)

    Note: this holds for NORMAL play. Misère Nim tweaks the endgame:
    the strategy is identical to normal Nim EXCEPT when all piles are ≤ 1,
    in which case you want to leave an ODD number of piles of size 1.
    """
    xor = 0
    for c in piles:
        xor ^= grundy_nim_value(c)
    return xor


def is_misere_winning(piles: List[int]) -> bool:
    """True iff the current player wins under misère convention with optimal play."""
    non_zero = [c for c in piles if c > 0]

    # Endgame: all remaining piles have at most 1 coin
    if all(c <= 1 for c in non_zero):
        # Win iff number of size-1 piles is EVEN (so opponent takes the last one)
        return len(non_zero) % 2 == 0

    # General case: same as normal Nim — win iff XOR ≠ 0
    return game_grundy(piles) != 0


def optimal_moves(piles: List[int]) -> List[int]:
    """Return all encoded move IDs that are optimal under misère Nim.

    Uses Grundy theory — O(piles) per call, no tree search needed.
    """
    non_zero = [c for c in piles if c > 0]
    winning_moves = []

    for move_id in legal_moves(piles):
        next_piles = apply_move(piles, move_id)
        # A move is optimal if it puts the OPPONENT in a losing position
        if not is_misere_winning(next_piles):
            winning_moves.append(move_id)

    # If no winning move exists (we're in a P-position), any legal move is
    # equally bad — return all legal moves so the agent plays something
    if not winning_moves:
        return legal_moves(piles)

    return winning_moves


# ── Padding helpers ────────────────────────────────────────────────────────────

def pad_piles(piles: List[int]) -> List[int]:
    """Pad pile list to MAX_PILES length with zeros."""
    p = list(piles)
    while len(p) < MAX_PILES:
        p.append(0)
    return p[:MAX_PILES]


def illegal_move_mask(piles: List[int]) -> np.ndarray:
    """Boolean mask of shape (N_MOVES,): True = move is ILLEGAL.

    Used to set illegal logits to -inf before softmax / argmax.
    """
    mask = np.ones(N_MOVES, dtype=bool)
    for m in legal_moves(piles):
        mask[m] = False
    return mask   # True where illegal
