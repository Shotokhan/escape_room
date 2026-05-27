"""
run_experiments.py — Generate datasets and run all three architectures.

This script:
  1. Generates random-play and rational-play datasets (if not already present)
  2. Trains each architecture on a chosen training set
  3. Evaluates on a chosen test set
  4. Saves plots and model weights
  5. Produces a comparison table across all three architectures

The key experiment:
  --train-set random --test-set rational
      → trains on noisy random-play data, evaluates on clean rational data
      → measures how much "real structure" was learned vs. dataset artefacts

  --train-set rational --test-set rational  (default)
      → clean training and evaluation

Usage
-----
    python run_experiments.py
    python run_experiments.py --train-set random --test-set rational
    python run_experiments.py --arch 1 --epochs 30
    python run_experiments.py --generate-only
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

from game import MAX_PILES, MAX_COINS, optimal_moves, illegal_move_mask, pad_piles
from dataset_random   import generate_dataset as gen_random,   samples_to_arrays, save_dataset
from dataset_rational import generate_dataset as gen_rational, generate_starting_position_dataset as _gen_rational_start


# ── Dataset paths ──────────────────────────────────────────────────────────────

DATASETS = {
    "random":   {"train": "data/random_train",   "test": "data/random_test"},
    "rational": {"train": "data/rational_train",  "test": "data/rational_test"},
}


def ensure_datasets(data_dir: str = "data", force: bool = False) -> None:
    os.makedirs(data_dir, exist_ok=True)

    files = {
        "random_train":   (gen_random,           dict(n_games=30_000,     seed=42)),
        "random_test":    (gen_random,            dict(n_games=5_000,      seed=999)),
        # Rational datasets use starting positions (not trajectories) so the
        # training distribution matches evaluation (fresh random game starts).
        "rational_train": (_gen_rational_start,   dict(n_samples=100_000,  seed=42)),
        "rational_test":  (_gen_rational_start,   dict(n_samples=25_000,   seed=1337)),
    }

    for name, (gen_fn, kwargs) in files.items():
        path = os.path.join(data_dir, name + ".npz")
        if os.path.exists(path) and not force:
            print(f"  [skip] {path} already exists")
            continue
        print(f"  Generating {name} ...")
        t0      = time.time()
        result  = gen_fn(**kwargs, verbose=False)
        # generators return either List[Sample] or (s, m, o) arrays directly
        if isinstance(result, tuple) and len(result) == 3 and hasattr(result[0], 'shape'):
            s, m, o = result
        else:
            s, m, o = samples_to_arrays(result)
        save_dataset(os.path.join(data_dir, name), s, m, o)
        print(f"         → {len(s)} samples in {time.time()-t0:.1f}s")


# ── Per-architecture evaluation ────────────────────────────────────────────────

def evaluate_agent(agent_cls, model_dir: str, n_trials: int = 1000) -> dict:
    """Pit the trained agent against the Grundy-optimal opponent.

    Returns win rate, and accuracy (chose an optimal move) on N/P positions.
    """
    import random, game as gm
    rng    = random.Random(42)
    wins   = 0
    n_acc  = {"N": [0, 0], "P": [0, 0]}   # [correct, total]

    try:
        agent = agent_cls(model_dir)
    except Exception as e:
        return {"error": str(e)}

    for _ in range(n_trials):
        n_piles = rng.randint(2, MAX_PILES)
        piles   = gm.pad_piles([rng.randint(1, MAX_COINS) for _ in range(n_piles)])

        # Accuracy check (single-step, no opponent needed)
        label  = "N" if gm.is_misere_winning(piles) else "P"
        try:
            p_idx, t_amt = agent.next_move(piles)
            from game import encode_move
            move_id = encode_move(p_idx, t_amt)
            correct = move_id in gm.optimal_moves(piles)
        except Exception:
            correct = False
        n_acc[label][0] += int(correct)
        n_acc[label][1] += 1

        # Win-rate: agent (player 0) vs optimal opponent (player 1)
        # Agent is called with the CURRENT pile state at each turn — stateless.
        cur    = list(piles)
        player = 0   # 0 = our agent, 1 = optimal
        for _ in range(50):
            if gm.is_terminal(cur):
                if player == 0:
                    wins += 1   # opponent took the last coin → we win
                break
            if player == 0:
                try:
                    pi, ta = agent.next_move(cur)
                    from game import encode_move
                    mid = encode_move(pi, ta)
                    if mid not in gm.legal_moves(cur):
                        mid = rng.choice(gm.legal_moves(cur))
                except Exception:
                    mid = rng.choice(gm.legal_moves(cur))
            else:
                opts = gm.optimal_moves(cur)
                mid  = rng.choice(opts)
            cur    = gm.apply_move(cur, mid)
            player = 1 - player

    return {
        "win_rate":    wins / n_trials,
        "acc_N":       n_acc["N"][0] / max(n_acc["N"][1], 1),
        "acc_P":       n_acc["P"][0] / max(n_acc["P"][1], 1),
        "n_trials":    n_trials,
    }


# ── Comparison plot ────────────────────────────────────────────────────────────

def plot_comparison(results: dict, out_dir: str = "plots") -> None:
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    archs   = [k for k in results if "error" not in results[k]]
    metrics = ["win_rate", "acc_N", "acc_P"]
    labels  = ["Win rate\nvs optimal", "Accuracy\nN-positions", "Accuracy\nP-positions"]
    colors  = ["#89b4fa", "#a6e3a1", "#f38ba8"]

    x     = np.arange(len(archs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[a][metric] for a in archs]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(archs)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Architecture comparison — win rate and accuracy")
    ax.legend(); ax.grid(axis="y")

    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all Trap Room ML experiments")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate datasets, do not train")
    parser.add_argument("--force-regen",   action="store_true",
                        help="Regenerate datasets even if files exist")
    parser.add_argument("--train-set",     choices=["random", "rational"], default="rational",
                        help="Which dataset to TRAIN on")
    parser.add_argument("--test-set",      choices=["random", "rational"], default="rational",
                        help="Which dataset to TEST on")
    parser.add_argument("--arch",          type=int, choices=[1, 2, 3], default=None,
                        help="Only run this architecture (default: all)")
    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--data-dir",      default="data")
    parser.add_argument("--model-dir",     default="models")
    parser.add_argument("--plot-dir",      default="plots")
    args = parser.parse_args()

    # ── Step 1: datasets ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Generating datasets")
    print("=" * 60)
    ensure_datasets(args.data_dir, force=args.force_regen)

    if args.generate_only:
        print("\nDatasets ready. Exiting (--generate-only).")
        return

    train_path = DATASETS[args.train_set]["train"]
    test_path  = DATASETS[args.test_set]["test"]
    print(f"\nTraining on: {train_path}")
    print(f"Testing on:  {test_path}")

    # ── Step 2: train architectures ───────────────────────────────────────────
    arch_ids = [args.arch] if args.arch else [1, 2, 3]
    results  = {}

    if 1 in arch_ids:
        print("\n" + "=" * 60)
        print("STEP 2a — Architecture 1: One-hot (78-dim)")
        print("=" * 60)
        from arch1_onehot import train as train1, visualise as vis1, OneHotAgent
        h1, m1 = train1(train_path, test_path, args.model_dir, args.epochs)
        vis1(m1, h1, args.plot_dir)
        results["Arch1\n(one-hot)"] = evaluate_agent(OneHotAgent, args.model_dir)

    if 2 in arch_ids:
        print("\n" + "=" * 60)
        print("STEP 2b — Architecture 2: Scalar (6-dim)")
        print("=" * 60)
        from arch2_scalar import train as train2, visualise as vis2, ScalarAgent
        h2, m2 = train2(train_path, test_path, args.model_dir, args.epochs)
        vis2(m2, h2, args.plot_dir)
        results["Arch2\n(scalar)"] = evaluate_agent(ScalarAgent, args.model_dir)

    if 3 in arch_ids:
        print("\n" + "=" * 60)
        print("STEP 2c — Architecture 3: Transformer")
        print("=" * 60)
        from arch3_transformer import train as train3, visualise as vis3, TransformerAgent
        h3, m3 = train3(train_path, test_path, args.model_dir, args.epochs)
        vis3(m3, h3, args.plot_dir)
        results["Arch3\n(transformer)"] = evaluate_agent(TransformerAgent, args.model_dir)

    # ── Step 3: comparison ────────────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("STEP 3 — Comparison")
        print("=" * 60)
        print(f"\n{'Architecture':<22} {'Win rate':>10} {'Acc N':>8} {'Acc P':>8}")
        print("-" * 52)
        for arch, r in results.items():
            name = arch.replace("\n", " ")
            if "error" in r:
                print(f"  {name:<20} ERROR: {r['error']}")
            else:
                print(f"  {name:<20} {r['win_rate']:>10.3f} {r['acc_N']:>8.3f} {r['acc_P']:>8.3f}")
        plot_comparison(results, args.plot_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
