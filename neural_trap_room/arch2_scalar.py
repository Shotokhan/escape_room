"""
arch2_scalar.py — Architecture 2: Scalar pile counts (6-dim) → move logits.

Input representation
--------------------
The state is a length-6 vector of raw pile counts (integers 0–12, normalised
to [0, 1] by dividing by MAX_COINS).  Nothing else.

Compared to Architecture 1 (one-hot), this is COMPACT: 6 numbers instead of 78.
The trade-off is that the network must LEARN that pile values are ordered and
periodic — it cannot read that off the input geometry.  If it succeeds, it
has effectively discovered Sprague-Grundy theory from game outcomes alone.

Key inductive bias difference from Architecture 1:
  - Arch 1 knows "pile=5" and "pile=6" are different things.
  - Arch 2 knows "pile=5" and "pile=6" are CLOSE (they differ by 1/12).
  - Arch 2 can therefore generalise to pile sizes it never saw during training
    (e.g. trained on piles ≤ 8, tested on piles ≤ 12) — Arch 1 cannot.

Output
------
Same 18-logit LM head as Architecture 1.

Architecture
------------
  Linear(12 → 128)  — expand: input = 6 normalised pile values + 6 mod-4 Grundy
                       values; neurons learn combinations of these features
  ReLU
  Linear(128 → 64)  — combine: with mod-4 values available as input, this
                       layer can compute XOR-like combinations directly
  ReLU
  Linear(64 → 18)   — LM head: map combined game value to per-move logits
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
    decode_move, illegal_move_mask, optimal_moves, pad_piles,
    grundy_nim_value, is_misere_winning
)
from dataset_random import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

# 6 raw pile values (0–12) + 6 pile%4 values = 12 input features.
# The mod-4 channel hands the network the Grundy periodicity explicitly,
# so layer 1 can focus on learning XOR-like combinations rather than
# rediscovering the periodic structure from scratch.
INPUT_DIM = MAX_PILES * 2   # 12
HIDDEN1   = 128
HIDDEN2   = 64


# ── Pre-processing ─────────────────────────────────────────────────────────────

def encode_state(piles) -> np.ndarray:
    """Encode pile counts as raw values + mod-4 Grundy features.  Shape: (12,).

    Piles are sorted in DECREASING order before encoding so that all
    permutations of the same multiset map to the same input vector.

    Channel 0–5: pile[i] / MAX_COINS  (normalised magnitude)
    Channel 6–11: pile[i] % 4 / 3     (Grundy value for this pile, normalised)

    The mod-4 channel is the key inductive bias: it tells the network directly
    which equivalence class each pile belongs to under Sprague-Grundy theory.
    Layer 1 can then learn XOR-like combinations of these values.
    """
    p = np.array(sorted(pad_piles(list(piles)), reverse=True), dtype=np.float32)
    return np.concatenate([p / MAX_COINS, (p % 4) / 3])


def preprocess_dataset(states: np.ndarray, moves: np.ndarray, outcomes: np.ndarray):
    """Convert raw dataset to tensors with SOFT targets over all optimal moves.

    See arch1_onehot.py for a full explanation of why soft targets are necessary.
    """
    from game import optimal_moves as get_optimal, N_MOVES
    mask     = outcomes == 1
    states_w = states[mask]

    X_list = []
    y_soft = np.zeros((len(states_w), N_MOVES), dtype=np.float32)
    for i, s in enumerate(states_w):
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

class ScalarNet(nn.Module):
    """Feedforward network: scalar piles (6-dim) → move logits (18-dim)."""

    def __init__(self):
        super().__init__()

        # ── Layer 1: feature extraction ───────────────────────────────────────
        # Takes 6 normalised pile values and projects to 64 hidden units.
        # Each neuron can compute a LINEAR combination of pile values — with
        # ReLU, piecewise-linear functions approximate the mod-4 periodicity
        # that underlies Grundy theory (pile value mod 4 determines Grundy number).
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)   # INPUT_DIM=12 now

        # ── Layer 2: interaction / XOR approximation ──────────────────────────
        # With mod-4 Grundy features explicitly available, this layer learns
        # XOR-like combinations across the 6 piles.  Width 128→64 gives enough
        # capacity to represent the full XOR table for Grundy values 0–3.
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)

        # ── LM head ───────────────────────────────────────────────────────────
        self.head = nn.Linear(HIDDEN2, N_MOVES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 6) → logits: (batch, 18)."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_path:   str,
    test_path:    str,
    out_dir:      str   = "models",
    epochs:       int   = 25,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading datasets...")
    Xtr, ytr = preprocess_dataset(*load_dataset(train_path))
    Xte, yte = preprocess_dataset(*load_dataset(test_path))
    print(f"  Train: {Xtr.shape[0]} samples | Test: {Xte.shape[0]} samples")

    loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    model     = ScalarNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history   = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0; correct = 0

        for Xb, yb_soft in loader:
            logits    = model(Xb)
            log_probs = F.log_softmax(logits, dim=-1)
            loss      = F.kl_div(log_probs, yb_soft, reduction='batchmean')
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * len(yb_soft)
            preds   = logits.argmax(1)
            correct += (yb_soft[torch.arange(len(preds)), preds] > 0).sum().item()

        train_loss = total_loss / len(ytr)
        train_acc  = correct    / len(ytr)
        model.eval()
        with torch.no_grad():
            te_preds = model(Xte).argmax(1)
            test_acc = (yte[torch.arange(len(te_preds)), te_preds] > 0).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    model_path = os.path.join(out_dir, "arch2_scalar.pt")
    torch.save(model.state_dict(), model_path)
    _export_numpy(model, out_dir)
    print(f"\nModel saved → {model_path}")
    return history, model


def _export_numpy(model: ScalarNet, out_dir: str) -> None:
    sd = model.state_dict()
    np.save(os.path.join(out_dir, "arch2_W1.npy"), sd["fc1.weight"].numpy())
    np.save(os.path.join(out_dir, "arch2_b1.npy"), sd["fc1.bias"].numpy())
    np.save(os.path.join(out_dir, "arch2_W2.npy"), sd["fc2.weight"].numpy())
    np.save(os.path.join(out_dir, "arch2_b2.npy"), sd["fc2.bias"].numpy())
    np.save(os.path.join(out_dir, "arch2_Wh.npy"), sd["head.weight"].numpy())
    np.save(os.path.join(out_dir, "arch2_bh.npy"), sd["head.bias"].numpy())
    print("  Numpy weights exported (arch2_*.npy)")


# ── Inference ──────────────────────────────────────────────────────────────────

class ScalarAgent:
    """Numpy-only inference agent for Architecture 2."""

    def __init__(self, model_dir: str = "models"):
        self.W1 = np.load(os.path.join(model_dir, "arch2_W1.npy"))
        self.b1 = np.load(os.path.join(model_dir, "arch2_b1.npy"))
        self.W2 = np.load(os.path.join(model_dir, "arch2_W2.npy"))
        self.b2 = np.load(os.path.join(model_dir, "arch2_b2.npy"))
        self.Wh = np.load(os.path.join(model_dir, "arch2_Wh.npy"))
        self.bh = np.load(os.path.join(model_dir, "arch2_bh.npy"))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(0, self.W1 @ x + self.b1)
        x = np.maximum(0, self.W2 @ x + self.b2)
        return self.Wh @ x + self.bh

    def next_move(self, piles) -> tuple:
        orig  = pad_piles(list(piles))
        order = sorted(range(len(orig)), key=lambda i: orig[i], reverse=True)
        sorted_piles = [orig[i] for i in order]

        x            = encode_state(sorted_piles)
        logits       = self._forward(x)
        mask         = illegal_move_mask(sorted_piles)
        logits[mask] = -np.inf

        sorted_pile_idx, take = decode_move(int(np.argmax(logits)))
        return (order[sorted_pile_idx], take)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualise(model: ScalarNet, history: dict, out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Architecture 2 — Scalar Encoding", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Training curves ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"]); ax.set_title("Training loss")
    ax.set_xlabel("Epoch"); ax.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["train_acc"], label="Train"); ax.plot(history["test_acc"], label="Test")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # ── 2. W1 response to pile values — does it learn periodicity? ────────────
    # Feed single-pile states (pile_0 = k, all others = 0) through layer 1.
    # If W1 learns Grundy values, activations should repeat with period 4.
    ax       = fig.add_subplot(gs[0, 2])
    W1       = model.fc1.weight.detach().numpy()   # (64, 6)
    # Compute post-ReLU activations for pile 0 = 0..12 (all other piles = 0).
    # We expect to see PERIODIC patterns (period 4) in the mod-4 channel neurons
    # if the model has learned Grundy structure.
    acts = []
    for k in range(13):
        x_in  = np.zeros(INPUT_DIM, dtype=np.float32)
        x_in[0] = k / MAX_COINS       # normalised pile value
        x_in[6] = (k % 4) / 3        # mod-4 Grundy channel for pile 0
        h     = np.maximum(0, W1 @ x_in + model.fc1.bias.detach().numpy())
        acts.append(h[:16])
    acts = np.array(acts)   # (13, 16)
    im   = ax.imshow(acts.T, aspect="auto", cmap="viridis")
    ax.set_title("Layer 1 activations vs pile 0\n(expect period-4 pattern in some rows)")
    ax.set_xlabel("Pile value (0–12)"); ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax)

    # ── 3. Generalisation: accuracy by pile range ─────────────────────────────
    # Train was on MAX_COINS=12; test accuracy broken down by pile-size range.
    ax     = fig.add_subplot(gs[1, 0])
    import random
    rng    = random.Random(42)
    ranges = [(1, 4), (5, 8), (9, 12)]
    accs   = []
    for lo, hi in ranges:
        correct = total = 0
        for _ in range(500):
            piles  = pad_piles([rng.randint(lo, hi) for _ in range(rng.randint(2, MAX_PILES))])
            x      = torch.tensor(encode_state(piles)).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0].numpy()
            mask         = illegal_move_mask(piles)
            logits[mask] = -np.inf
            pred         = int(np.argmax(logits))
            correct     += int(pred in optimal_moves(piles))
            total       += 1
        accs.append(correct / total)
    bars = ax.bar([f"{lo}–{hi}" for lo, hi in ranges], accs, color="#89b4fa")
    ax.set_ylim(0, 1); ax.set_title("Accuracy by pile-size range")
    ax.set_xlabel("Pile value range"); ax.set_ylabel("Accuracy"); ax.grid(axis="y")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", fontsize=9)

    # ── 4. Grundy XOR vs model logit sum — scatter plot ──────────────────────
    ax  = fig.add_subplot(gs[1, 1])
    xor_vals, max_logits = [], []
    for _ in range(300):
        piles  = pad_piles([rng.randint(1, MAX_COINS) for _ in range(rng.randint(2, MAX_PILES))])
        xor    = 0
        for c in piles:
            xor ^= grundy_nim_value(c)
        x      = torch.tensor(encode_state(piles)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0].numpy()
        mask         = illegal_move_mask(piles)
        logits[mask] = -np.inf
        xor_vals.append(xor); max_logits.append(float(np.nanmax(logits)))
    ax.scatter(xor_vals, max_logits, alpha=0.3, s=8)
    ax.set_title("Grundy XOR vs max legal logit")
    ax.set_xlabel("XOR of pile Grundy values"); ax.set_ylabel("Max logit")
    ax.grid(True)

    # ── 5. N vs P accuracy ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _plot_grundy_accuracy(model, ax)

    path = os.path.join(out_dir, "arch2_scalar.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


def _plot_grundy_accuracy(model: nn.Module, ax) -> None:
    import random
    rng   = random.Random(7)
    stats = {"N": [0, 0], "P": [0, 0]}   # [correct, total]
    for _ in range(2000):
        piles  = pad_piles([rng.randint(1, MAX_COINS) for _ in range(rng.randint(2, MAX_PILES))])
        label  = "N" if is_misere_winning(piles) else "P"
        x      = torch.tensor(encode_state(piles)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0].numpy()
        mask         = illegal_move_mask(piles)
        logits[mask] = -np.inf
        pred         = int(np.argmax(logits))
        stats[label][0] += int(pred in optimal_moves(piles))
        stats[label][1] += 1
    labels = ["N-position\n(should win)", "P-position\n(graceful loss)"]
    accs   = [stats["N"][0] / stats["N"][1], stats["P"][0] / stats["P"][1]]
    bars   = ax.bar(labels, accs, color=["#a6e3a1", "#f38ba8"])
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Grundy accuracy: N vs P")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture 2: scalar Nim model")
    parser.add_argument("--train",   default="data/random_train")
    parser.add_argument("--test",    default="data/random_test")
    parser.add_argument("--epochs",  type=int, default=25)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--plots",   default="plots")
    args = parser.parse_args()

    history, model = train(
        train_path=args.train, test_path=args.test,
        out_dir=args.out_dir, epochs=args.epochs,
    )
    visualise(model, history, out_dir=args.plots)
