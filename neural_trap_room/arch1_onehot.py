"""
arch1_onehot.py — Architecture 1: One-hot encoded state → move logits.

Input representation
--------------------
Each pile (0–12 coins) is encoded as a length-13 one-hot vector.
Six piles concatenated → 78-dimensional binary input vector.

This representation gives the network EXPLICIT GEOMETRY: two states that
differ only in pile 3 having 5 vs 6 coins will have inputs that differ in
exactly 2 positions (one bit goes from 1→0, another from 0→1).  The network
does not need to learn that "5 and 6 are adjacent" — the encoding says so.

In effect, the network is learning a LOOKUP TABLE from game states to optimal
moves, with some generalisation forced by the shared weights across piles.

Most input dimensions are 0; exactly one dimension per pile-slice is 1.

Output
------
18 logits — one per (pile_index × take_amount) pair.
Illegal moves are masked to -∞ before softmax / argmax at inference time.

Architecture
------------
  Linear(78 → 128)   — expand: map each one-hot state to a dense representation
  ReLU               — non-linearity: allow the network to model XOR-like patterns
  Linear(128 → 64)   — compress: force the network to find a compact representation
  ReLU
  Linear(64 → 18)    — LM head: project to move logits (one per legal action)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from game import (
    MAX_PILES, MAX_COINS, N_MOVES, TAKE_VALUES,
    decode_move, illegal_move_mask, optimal_moves, pad_piles, legal_moves
)
from dataset_random import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

INPUT_DIM   = MAX_PILES * (MAX_COINS + 1)   # 6 × 13 = 78
HIDDEN1     = 128
HIDDEN2     = 64


# ── Pre-processing ─────────────────────────────────────────────────────────────

def encode_state(piles) -> np.ndarray:
    """Encode a pile list as a 78-dimensional one-hot vector.

    Piles are sorted in DECREASING order before encoding so that all
    permutations of the same multiset map to the same input vector.
    This collapses up to 6! = 720 equivalent states into one, removing
    positional noise and halving the effective state space.

    Pile i contributes a 13-dim one-hot at positions [i*13 : (i+1)*13].
    piles[i] = k  →  position i*13 + k is set to 1.
    """
    piles = sorted(pad_piles(list(piles)), reverse=True)
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    for i, c in enumerate(piles):
        vec[i * (MAX_COINS + 1) + int(c)] = 1.0
    return vec


def preprocess_dataset(states: np.ndarray, moves: np.ndarray, outcomes: np.ndarray):
    """Convert raw dataset to model-ready tensors with SOFT targets.

    Hard target: one-hot over the single move the winning player chose.
    Problem: there are often multiple optimal moves for a given state.
             Training with a one-hot punishes the model for choosing a
             *different* optimal move — it gets penalised for correct answers.

    Soft target: distribute probability mass equally over ALL optimal moves
                 for each state (computed via Grundy theorem, O(P) per state).
                 This is equivalent to training with the Grundy policy directly.

    Only states where outcome == +1 (winning player's moves) are included.
    """
    from game import optimal_moves as get_optimal, N_MOVES
    mask     = outcomes == 1
    states_w = states[mask]

    X_list = []
    y_soft = np.zeros((len(states_w), N_MOVES), dtype=np.float32)

    for i, s in enumerate(states_w):
        # Sort piles to match the sorted encoding — optimal_moves must be
        # computed on the same sorted state that encode_state will see.
        s_sorted = sorted(pad_piles(list(s)), reverse=True)
        opts = get_optimal(s_sorted)
        X_list.append(encode_state(s_sorted))
        if opts:
            for m in opts:
                y_soft[i, m] = 1.0 / len(opts)
        else:
            y_soft[i, :] = 1.0 / N_MOVES

    return torch.tensor(np.stack(X_list)), torch.tensor(y_soft)


# ── Model ──────────────────────────────────────────────────────────────────────

class OneHotNet(nn.Module):
    """Feedforward network: one-hot state (78-dim) → move logits (18-dim)."""

    def __init__(self):
        super().__init__()

        # ── Layer 1: expansion ────────────────────────────────────────────────
        # Maps the 78-dim sparse one-hot input to a 128-dim dense representation.
        # Each neuron learns to recognise a PATTERN of pile values — e.g. "pile 2
        # has exactly 5 coins".  With 128 neurons, the network can memorise many
        # distinct pile configurations simultaneously.
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)

        # ── Layer 2: compression / combination ────────────────────────────────
        # Reduces 128 → 64, forcing the network to combine single-pile patterns
        # into JOINT representations, e.g. "pile 2 is 5 AND pile 4 is 1".
        # This is where XOR-like interactions between piles must be captured.
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)

        # ── LM head: move logits ──────────────────────────────────────────────
        # Projects the 64-dim compressed state to 18 logits — one per move.
        # Higher logit = network believes this move leads to a win.
        # Illegal moves are masked externally (not here), keeping gradients clean.
        self.head = nn.Linear(HIDDEN2, N_MOVES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 78) → logits: (batch, 18)."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_path: str,
    test_path:  str,
    out_dir:    str  = "models",
    epochs:     int  = 25,
    batch_size: int  = 256,
    lr:         float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    """Train the one-hot model and return history dict."""
    os.makedirs(out_dir, exist_ok=True)

    print("Loading datasets...")
    Xtr, ytr = preprocess_dataset(*load_dataset(train_path))
    Xte, yte = preprocess_dataset(*load_dataset(test_path))
    print(f"  Train: {Xtr.shape[0]} samples | Test: {Xte.shape[0]} samples")

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)  # ytr is now soft (N, 18)

    model     = OneHotNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history   = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0

        for Xb, yb_soft in loader:
            logits = model(Xb)
            # KL divergence loss against soft target (sum of optimal moves).
            # F.kl_div expects log-probabilities as input and probabilities as target.
            log_probs = F.log_softmax(logits, dim=-1)
            loss      = F.kl_div(log_probs, yb_soft, reduction='batchmean')
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(yb_soft)
            # Accuracy: did the argmax land on ANY optimal move? (not just the one
            # that happened to be in the training data)
            preds   = logits.argmax(1)
            correct += (yb_soft[torch.arange(len(preds)), preds] > 0).sum().item()

        train_loss = total_loss / len(ytr)
        train_acc  = correct    / len(ytr)

        model.eval()
        with torch.no_grad():
            te_logits  = model(Xte)
            te_preds   = te_logits.argmax(1)
            test_acc   = (yte[torch.arange(len(te_preds)), te_preds] > 0).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    # Save weights as numpy arrays for Pyodide / numpy export
    model_path = os.path.join(out_dir, "arch1_onehot.pt")
    torch.save(model.state_dict(), model_path)
    _export_numpy(model, out_dir)
    print(f"\nModel saved → {model_path}")

    return history, model


def _export_numpy(model: OneHotNet, out_dir: str) -> None:
    """Export weights as .npy files for numpy-only inference (e.g. in Pyodide)."""
    sd = model.state_dict()
    np.save(os.path.join(out_dir, "arch1_W1.npy"), sd["fc1.weight"].numpy())
    np.save(os.path.join(out_dir, "arch1_b1.npy"), sd["fc1.bias"].numpy())
    np.save(os.path.join(out_dir, "arch1_W2.npy"), sd["fc2.weight"].numpy())
    np.save(os.path.join(out_dir, "arch1_b2.npy"), sd["fc2.bias"].numpy())
    np.save(os.path.join(out_dir, "arch1_Wh.npy"), sd["head.weight"].numpy())
    np.save(os.path.join(out_dir, "arch1_bh.npy"), sd["head.bias"].numpy())
    print("  Numpy weights exported (arch1_*.npy)")


# ── Inference ──────────────────────────────────────────────────────────────────

class OneHotAgent:
    """Wraps the trained model for use as a Trap Room agent (numpy-only)."""

    def __init__(self, model_dir: str = "models"):
        self.W1 = np.load(os.path.join(model_dir, "arch1_W1.npy"))
        self.b1 = np.load(os.path.join(model_dir, "arch1_b1.npy"))
        self.W2 = np.load(os.path.join(model_dir, "arch1_W2.npy"))
        self.b2 = np.load(os.path.join(model_dir, "arch1_b2.npy"))
        self.Wh = np.load(os.path.join(model_dir, "arch1_Wh.npy"))
        self.bh = np.load(os.path.join(model_dir, "arch1_bh.npy"))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Pure numpy forward pass — 5 lines, no framework."""
        x = np.maximum(0, self.W1 @ x + self.b1)   # ReLU layer 1
        x = np.maximum(0, self.W2 @ x + self.b2)   # ReLU layer 2
        return self.Wh @ x + self.bh                # logits

    def next_move(self, piles) -> tuple:
        """Return (pile_index, take_amount) for the best legal move.

        Piles are sorted descending before encoding (matching training),
        then the chosen pile index is translated back to the original
        position so the caller can apply the move to the unsorted state.
        """
        orig  = pad_piles(list(piles))
        order = sorted(range(len(orig)), key=lambda i: orig[i], reverse=True)
        sorted_piles = [orig[i] for i in order]

        x      = encode_state(sorted_piles)
        logits = self._forward(x)

        mask         = illegal_move_mask(sorted_piles)
        logits[mask] = -np.inf

        sorted_pile_idx, take = decode_move(int(np.argmax(logits)))
        original_pile_idx     = order[sorted_pile_idx]   # map back to caller's index
        return (original_pile_idx, take)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualise(model: OneHotNet, history: dict, out_dir: str = "plots") -> None:
    """Produce four diagnostic plots for Architecture 1."""
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Architecture 1 — One-hot Encoding", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Training curves ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"], label="Train loss")
    ax.set_title("Training loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["train_acc"], label="Train acc")
    ax.plot(history["test_acc"],  label="Test acc")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # ── 2. W1 heatmap — one-hot embedding patterns ───────────────────────────
    # Reshape W1 into 6 pile-blocks × 13 one-hot positions × HIDDEN1 neurons.
    # We show the first pile's embedding: row i = neuron i's response to pile=k coins.
    ax    = fig.add_subplot(gs[0, 2])
    W1    = model.fc1.weight.detach().numpy()   # (128, 78)
    # Take the first pile's 13 columns and show the top-32 neurons
    block = W1[:32, :13]   # (32, 13) — 32 neurons × 13 possible pile values
    im    = ax.imshow(block, aspect="auto", cmap="RdBu_r")
    ax.set_title("W1 — pile 0 embedding\n(first 32 neurons)")
    ax.set_xlabel("Pile value (0–12)"); ax.set_ylabel("Neuron index")
    plt.colorbar(im, ax=ax)

    # ── 3. Move frequency — what moves does the model prefer? ─────────────────
    ax     = fig.add_subplot(gs[1, 0])
    sample_logits = []
    # Probe the model on a grid of states: each pile has a single pile of size k
    for k in range(1, 13):
        piles  = pad_piles([k])
        x      = torch.tensor(encode_state(piles)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0].numpy()
        mask         = illegal_move_mask(piles)
        logits[mask] = -np.inf
        sample_logits.append(logits)
    mat = np.array(sample_logits)   # (12, 18)
    im  = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title("Logits: single-pile game\n(row=pile size, col=move id)")
    ax.set_xlabel("Move ID"); ax.set_ylabel("Pile size"); plt.colorbar(im, ax=ax)

    # ── 4. Grundy XOR accuracy — does the model understand P vs N positions? ──
    ax = fig.add_subplot(gs[1, 1:])
    _plot_grundy_accuracy(model, ax)

    path = os.path.join(out_dir, "arch1_onehot.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


def _plot_grundy_accuracy(model: nn.Module, ax) -> None:
    """Bar chart: model accuracy split by N-position vs P-position."""
    from game import is_misere_winning
    import random

    rng      = random.Random(7)
    n_pos    = {"N": {"correct": 0, "total": 0}, "P": {"correct": 0, "total": 0}}

    for _ in range(2000):
        n_piles = rng.randint(2, MAX_PILES)
        piles   = pad_piles([rng.randint(1, MAX_COINS) for _ in range(n_piles)])
        label   = "N" if is_misere_winning(piles) else "P"

        x      = torch.tensor(encode_state(piles)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0].numpy()
        mask         = illegal_move_mask(piles)
        logits[mask] = -np.inf
        pred_move    = int(np.argmax(logits))

        # A correct prediction means the chosen move is actually in optimal_moves
        correct = pred_move in optimal_moves(piles)
        n_pos[label]["total"]   += 1
        n_pos[label]["correct"] += int(correct)

    labels = ["N-position\n(should win)", "P-position\n(should lose gracefully)"]
    accs   = [
        n_pos["N"]["correct"] / max(n_pos["N"]["total"], 1),
        n_pos["P"]["correct"] / max(n_pos["P"]["total"], 1),
    ]
    bars = ax.bar(labels, accs, color=["#a6e3a1", "#f38ba8"])
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy (optimal move chosen)")
    ax.set_title("Grundy accuracy: N vs P positions")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture 1: one-hot Nim model")
    parser.add_argument("--train",   default="data/random_train",  help="training dataset path")
    parser.add_argument("--test",    default="data/random_test",   help="test dataset path")
    parser.add_argument("--epochs",  type=int, default=25)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--plots",   default="plots")
    args = parser.parse_args()

    history, model = train(
        train_path=args.train,
        test_path=args.test,
        out_dir=args.out_dir,
        epochs=args.epochs,
    )
    visualise(model, history, out_dir=args.plots)
