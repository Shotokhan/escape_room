"""
dataset_random.py — Dataset from randomly / sub-optimally played Nim games.

Each game is played by two agents that choose moves uniformly at random from
the set of legal moves.  We record every (state, move) pair in the game and
label it with the outcome for the player who made that move:
  +1  →  that player eventually won
  -1  →  that player eventually lost

Pre-processing for each architecture is handled by the three arch modules.
This module only produces the raw trajectory dataset and saves it to disk.

Usage
-----
    python dataset_random.py                  # generates both splits
    python dataset_random.py --games 50000   # custom game count
"""

import argparse
import random
import numpy as np
import os
from typing import List, Tuple

from game import (
    MAX_PILES, MAX_COINS, N_MOVES,
    pad_piles, legal_moves, apply_move, is_terminal, encode_move
)

# ── Types ─────────────────────────────────────────────────────────────────────
# A single sample: (piles_before_move [6 ints], move_id [int], outcome [+1/-1])
Sample = Tuple[List[int], int, int]


# ── Game simulation ────────────────────────────────────────────────────────────

def random_starting_piles(
    min_piles: int = 2,
    max_piles: int = MAX_PILES,
    min_coins: int = 1,
    max_coins: int = MAX_COINS,
    rng: random.Random = None
) -> List[int]:
    """Sample a random starting configuration and pad to MAX_PILES."""
    if rng is None:
        rng = random
    n_piles = rng.randint(min_piles, max_piles)
    piles   = [rng.randint(min_coins, max_coins) for _ in range(n_piles)]
    return pad_piles(piles)


def play_random_game(
    piles: List[int],
    rng: random.Random = None
) -> List[Sample]:
    """Play one game with both players choosing uniformly random legal moves.

    Returns a list of (state, move_id, outcome) triples — one per turn.
    Outcome is from the perspective of the player who made that move:
      +1 if they won, -1 if they lost.
    """
    if rng is None:
        rng = random

    trajectory: List[Tuple[List[int], int]] = []   # (state, move_id) before outcome known
    current = list(piles)
    player  = 0   # alternates 0 / 1

    while not is_terminal(current):
        moves  = legal_moves(current)
        move   = rng.choice(moves)
        trajectory.append((list(current), move, player))
        current = apply_move(current, move)
        player  = 1 - player

    # The player who faces an empty board (their turn but no coins left) wins —
    # in misère Nim the LAST taker loses, so the player whose turn it is now
    # is the winner (their opponent took the last coin).
    winner = player

    samples: List[Sample] = []
    for state, move_id, acting_player in trajectory:
        outcome = +1 if acting_player == winner else -1
        samples.append((state, move_id, outcome))

    return samples


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    n_games:    int  = 30_000,
    seed:       int  = 42,
    verbose:    bool = True,
) -> List[Sample]:
    """Generate `n_games` randomly played games and return all samples."""
    rng     = random.Random(seed)
    samples: List[Sample] = []

    for i in range(n_games):
        piles   = random_starting_piles(rng=rng)
        samples += play_random_game(piles, rng=rng)

        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_games} games — {len(samples)} samples so far")

    return samples


def samples_to_arrays(samples: List[Sample]):
    """Convert sample list to numpy arrays.

    Returns
    -------
    states   : (N, 6)  int array  — padded pile counts
    moves    : (N,)    int array  — encoded move IDs in [0, 17]
    outcomes : (N,)    int array  — +1 (win) or -1 (loss)
    """
    states   = np.array([s[0] for s in samples], dtype=np.int8)
    moves    = np.array([s[1] for s in samples], dtype=np.int8)
    outcomes = np.array([s[2] for s in samples], dtype=np.int8)
    return states, moves, outcomes


def save_dataset(path: str, states, moves, outcomes) -> None:
    np.savez_compressed(path, states=states, moves=moves, outcomes=outcomes)
    size_kb = os.path.getsize(path + ".npz") // 1024
    print(f"  Saved → {path}.npz  ({states.shape[0]} samples, {size_kb} KB)")


def load_dataset(path: str):
    """Load a saved dataset. Returns (states, moves, outcomes)."""
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    return data["states"], data["moves"], data["outcomes"]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random-play Nim datasets")
    parser.add_argument("--games",      type=int, default=30_000, help="games for train split")
    parser.add_argument("--test-games", type=int, default=5_000,  help="games for test split")
    parser.add_argument("--out-dir",    type=str, default="data",  help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating RANDOM train dataset ({args.games} games)...")
    train_samples = generate_dataset(n_games=args.games, seed=42)
    s, m, o = samples_to_arrays(train_samples)
    print(f"  Win rate: {(o == 1).mean():.3f}")
    save_dataset(os.path.join(args.out_dir, "random_train"), s, m, o)

    print(f"\nGenerating RANDOM test dataset ({args.test_games} games)...")
    test_samples = generate_dataset(n_games=args.test_games, seed=999)
    s, m, o = samples_to_arrays(test_samples)
    print(f"  Win rate: {(o == 1).mean():.3f}")
    save_dataset(os.path.join(args.out_dir, "random_test"), s, m, o)

    print("\nDone.")
