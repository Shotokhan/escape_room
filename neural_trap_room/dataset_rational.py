"""
dataset_rational.py — Dataset from optimally played Nim games.

Both players always choose an optimal move according to Grundy / Sprague-Grundy
theory (see game.py).  This requires NO minimax search — every move decision is
O(P) where P = number of piles, making generation extremely fast even for tens
of thousands of games.

Label convention (same as dataset_random.py):
  +1  →  the acting player eventually won
  -1  →  the acting player eventually lost

Because both players play optimally:
  - The player in an N-position (XOR ≠ 0) always wins → all their moves get +1
  - The player in a P-position (XOR = 0) always loses → all their moves get -1

This produces very clean labels but a biased distribution: from a random starting
state, the first player wins more often (N-positions are more common than
P-positions when pile sizes are uniformly random).

Usage
-----
    python dataset_rational.py
    python dataset_rational.py --games 20000
"""

import argparse
import random
import numpy as np
import os
from typing import List, Tuple

from game import (
    MAX_PILES, MAX_COINS, N_MOVES,
    pad_piles, legal_moves, apply_move, is_terminal,
    optimal_moves, is_misere_winning
)
from dataset_random import Sample, samples_to_arrays, save_dataset


# ── Game simulation ────────────────────────────────────────────────────────────

def play_rational_game(
    piles: List[int],
    rng: random.Random = None,
) -> List[Sample]:
    """Play one game with BOTH players using Grundy-optimal strategy.

    When the current player has multiple optimal moves, one is chosen
    uniformly at random among them — this adds variety to the dataset
    without sacrificing optimality.

    Returns (state, move_id, outcome) triples.
    """
    if rng is None:
        rng = random

    trajectory: List[Tuple[List[int], int, int]] = []
    current = list(piles)
    player  = 0

    while not is_terminal(current):
        best = optimal_moves(current)
        move = rng.choice(best)
        trajectory.append((list(current), move, player))
        current = apply_move(current, move)
        player  = 1 - player

    winner = player   # same convention as dataset_random

    samples: List[Sample] = []
    for state, move_id, acting_player in trajectory:
        outcome = +1 if acting_player == winner else -1
        samples.append((state, move_id, outcome))

    return samples


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(
    n_games:    int  = 20_000,
    seed:       int  = 42,
    verbose:    bool = True,
) -> List[Sample]:
    """Generate `n_games` rationally played games."""
    rng     = random.Random(seed)
    samples: List[Sample] = []

    for i in range(n_games):
        # Same distribution as random dataset so the two are directly comparable
        n_piles = rng.randint(2, MAX_PILES)
        piles   = pad_piles([rng.randint(1, MAX_COINS) for _ in range(n_piles)])
        samples += play_rational_game(piles, rng=rng)

        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_games} games — {len(samples)} samples so far")

    return samples


def generate_starting_position_dataset(
    n_samples:  int  = 100_000,
    seed:       int  = 42,
    verbose:    bool = True,
) -> List[Sample]:
    """Generate samples directly from random STARTING positions.

    Each sample is (random_start_piles, one_optimal_move, +1).
    This avoids the mid-game trajectory skew: training distribution matches
    evaluation distribution (fresh random starts with large pile values).

    All samples have outcome +1 because we always record an optimal move.
    The model learns: given this start state, this is a winning move.
    """
    rng     = random.Random(seed)
    samples: List[Sample] = []

    for i in range(n_samples):
        # Include single-pile states explicitly — they appear at the end of every
        # real game but were absent from the training set (min was 2 piles).
        # Without them, models never learn the misère flip: in single-pile misère
        # Nim the correct move is always "take pile-1, leave 1" — the opposite of
        # the normal-Nim instinct of "clear a pile with Grundy=pile".
        # Weight: ~20% single-pile, 80% multi-pile (matching game frequency).
        if rng.random() < 0.20:
            n_piles = 1
        else:
            n_piles = rng.randint(2, MAX_PILES)
        piles   = pad_piles([rng.randint(1, MAX_COINS) for _ in range(n_piles)])
        opts    = optimal_moves(piles)
        if not opts:
            continue
        # Sample one optimal move (variety prevents mode collapse)
        move = rng.choice(opts)
        samples.append((list(piles), move, +1))

        if verbose and (i + 1) % 20000 == 0:
            print(f"  {i+1}/{n_samples} samples")

    return samples


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rational-play Nim datasets")
    parser.add_argument("--games",      type=int, default=20_000)
    parser.add_argument("--test-games", type=int, default=5_000)
    parser.add_argument("--out-dir",    type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating RATIONAL train dataset ({args.games * 5} starting positions)...")
    train_samples = generate_starting_position_dataset(n_samples=args.games * 5, seed=42)
    s, m, o = samples_to_arrays(train_samples)
    print(f"  All outcomes +1 (optimal moves): {(o == 1).mean():.3f}")
    save_dataset(os.path.join(args.out_dir, "rational_train"), s, m, o)

    print(f"\nGenerating RATIONAL test dataset ({args.test_games} starting positions)...")
    test_samples = generate_starting_position_dataset(n_samples=args.test_games * 5, seed=1337)
    s, m, o = samples_to_arrays(test_samples)
    print(f"  Samples: {len(s)}")
    save_dataset(os.path.join(args.out_dir, "rational_test"), s, m, o)

    print("\nDone.")
