"""
arch3_transformer.py — Architecture 3: Causal self-attention sequence model.

This architecture is the closest to a small language model.

Sequence format
---------------
Each game is represented as a TOKEN SEQUENCE:

  [v0] [v1] [v2] [v3] [v4] [v5] [SEP] [m0] [m1] ... [m_T]
   ↑── initial pile values (6 tokens) ──↑  ↑── moves ──↑

  - Tokens 0–12  : pile-value tokens (the "prompt" / context)
  - Token 13     : SEP — separates pile values from moves
  - Tokens 14–31 : move tokens (18 possible moves, encoded as in arch1/2)

Vocabulary size = 32 tokens total.

The model predicts the NEXT token at every position (teacher forcing during
training), using CAUSAL attention so position t can only attend to positions
0..t-1.  At inference, we feed the 7-token prefix (piles + SEP) and
autoregressively generate the next move.

Why this is like a language model
----------------------------------
  - The pile values are the "system prompt" — they condition every prediction.
  - The move tokens are the "completion" — each is predicted given all previous.
  - Causal self-attention is exactly the mechanism in GPT / decoder-only LLMs.
  - The attention patterns can be visualised to see WHICH pile tokens the model
    attends to when choosing each move — interpretability for free.

Architecture
------------
  Embedding(32, d_model=32)   — token → dense vector; pile tokens and move
                                 tokens live in the SAME embedding space, so
                                 the model can relate pile values to moves

  [× 2 Transformer blocks]
    LayerNorm                 — stabilise activations before attention
    CausalSelfAttention       — each position attends to all previous positions;
      (4 heads, head_dim=8)     captures DEPENDENCIES: "given pile 3 is large AND
                                 pile 1 is small, take 3 from pile 3"
    Residual connection       — preserve gradient flow; also means the model can
                                 "look up" the raw token embedding at any depth
    LayerNorm
    FFN: d→4d→d, GELU         — per-position transformation; learns non-linear
                                 functions of the attended context (XOR-like logic
                                 can be approximated here with enough width)
    Residual connection

  Linear(d_model → vocab_size=32)   — LM head: project to token logits
                                       Only the 18 move-token logits matter at
                                       inference; pile-value positions are ignored
"""

import argparse
import os
import math
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
    is_misere_winning, apply_move, legal_moves
)
from dataset_random import load_dataset

# ── Vocabulary ─────────────────────────────────────────────────────────────────

N_PILE_TOKENS = MAX_COINS + 1    # 13  (values 0–12)
SEP_TOKEN     = N_PILE_TOKENS    # 13
MOVE_OFFSET   = SEP_TOKEN + 1    # 14  (move i → token MOVE_OFFSET + i)
VOCAB_SIZE    = MOVE_OFFSET + N_MOVES   # 32

D_MODEL       = 64   # increased from 32: more capacity to represent XOR structure
N_HEADS       = 4
N_LAYERS      = 2
FFN_MULT      = 4   # FFN hidden dim = D_MODEL × FFN_MULT

MAX_SEQ_LEN   = MAX_PILES + 1 + N_MOVES   # 6 + 1(SEP) + 18 = 25 (upper bound)


def pile_token(value: int) -> int:
    """Encode a pile value as a vocabulary token."""
    return int(value)   # 0–12


def move_token(move_id: int) -> int:
    """Encode a move ID as a vocabulary token."""
    return MOVE_OFFSET + move_id   # 14–31


def token_to_move(token: int) -> int:
    """Decode a token back to a move ID, or -1 if not a move token."""
    if token < MOVE_OFFSET:
        return -1
    return token - MOVE_OFFSET


# ── Pre-processing ─────────────────────────────────────────────────────────────

def game_to_sequences(piles, move_ids, only_winner_moves: bool = True):
    """Convert a (piles, moves) trajectory to input/target token sequences.

    The prefix is always:
        [pile_0_token, ..., pile_5_token, SEP_TOKEN]

    For each move in the trajectory, we produce ONE training sample:
        input  = prefix + moves so far (length 7 + t)
        target = next move token

    If only_winner_moves=True, only moves that led to the win are included.
    """
    prefix = [pile_token(c) for c in piles] + [SEP_TOKEN]
    samples = []
    for t, move_id in enumerate(move_ids):
        input_seq  = prefix + [move_token(m) for m in move_ids[:t]]
        target_tok = move_token(move_id)
        samples.append((input_seq, target_tok))
    return samples


def preprocess_dataset(
    states:   np.ndarray,
    moves:    np.ndarray,
    outcomes: np.ndarray,
    max_seq:  int = MAX_SEQ_LEN,
):
    """Group raw (state, move, outcome) triples back into per-game trajectories,
    then build token sequences with causal targets.

    Strategy: a 'game' is a contiguous run of rows with alternating outcomes
    (+1, -1, +1, ...).  We reconstruct the trajectory by noting that
    the state after move t equals the state before move t+1.

    Only sequences where the ACTING player eventually won are used as
    positive training examples (outcome == +1).
    """
    # ── Reconstruct game boundaries ───────────────────────────────────────────
    # A new game starts when the pile state at row t is NOT the result of
    # applying moves[t-1] to state[t-1].  We detect this by checking whether
    # any pile count INCREASED (coins can only decrease within a game).
    boundaries = [0]
    for i in range(1, len(states)):
        prev_after = apply_move(list(states[i - 1]), int(moves[i - 1]))
        if list(states[i]) != prev_after:
            boundaries.append(i)
    boundaries.append(len(states))

    # ── Build token sequences ─────────────────────────────────────────────────
    all_inputs  = []   # list of int lists (variable length)
    all_targets = []   # list of int (next move token)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        game_states  = states[start:end]
        game_moves   = moves[start:end]
        game_outcomes = outcomes[start:end]

        piles = sorted(pad_piles(list(game_states[0])), reverse=True)   # canonical form

        # Build list of (move_id, outcome) for each step
        step_moves = []
        for i in range(len(game_moves)):
            if game_outcomes[i] == 1:   # winning move
                step_moves.append(int(game_moves[i]))

        if not step_moves:
            continue

        # Only emit the FIRST winning move per game with NO move history.
        # Input is always exactly [p0, p1, p2, p3, p4, p5, SEP] — the pile prefix.
        # This forces the model to predict entirely from the game state, preventing
        # it from learning sequential move patterns instead of Grundy structure.
        # (With full history, the transformer found it easier to attend to recent
        # moves than to read the pile-value prefix — see plot interpretation.)
        prefix = [pile_token(c) for c in piles] + [SEP_TOKEN]
        winning_positions = [i for i, o in enumerate(game_outcomes) if o == 1]
        if not winning_positions:
            continue
        # Use only the FIRST winning move in the game for each training example
        pos        = winning_positions[0]
        input_seq  = prefix   # no move history — pure game-state conditioning
        target_tok = move_token(int(game_moves[pos]))
        all_inputs.append(input_seq)
        all_targets.append(target_tok)

    # Pad sequences to the same length
    max_len = max(len(s) for s in all_inputs) if all_inputs else 1
    padded  = np.zeros((len(all_inputs), max_len), dtype=np.int64)
    lengths = np.zeros(len(all_inputs), dtype=np.int64)
    for i, seq in enumerate(all_inputs):
        padded[i, :len(seq)] = seq
        lengths[i] = len(seq)

    # Build soft targets: distribute mass over all optimal moves for each state
    from game import optimal_moves as get_optimal
    y_soft = np.zeros((len(all_targets), N_MOVES), dtype=np.float32)
    for i, (seq, tok) in enumerate(zip(all_inputs, all_targets)):
        # Reconstruct pile state from sequence prefix (first MAX_PILES tokens)
        pile_state = [int(seq[j]) for j in range(MAX_PILES)]
        opts = get_optimal(pile_state)
        if opts:
            for m in opts:
                y_soft[i, m] = 1.0 / len(opts)
        else:
            y_soft[i, :] = 1.0 / N_MOVES

    X       = torch.tensor(padded)
    y_soft  = torch.tensor(y_soft)
    lengths = torch.tensor(lengths)

    return X, y_soft, lengths


# ── Model ──────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    'Causal' means: position t can only attend to positions 0..t.
    This is the core mechanism of decoder-only LLMs (GPT family).
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # ── QKV projection ────────────────────────────────────────────────────
        # Each head learns a different "question" (Q), "key" (K), and "value" (V).
        # Q and K determine WHERE to attend; V determines WHAT to read.
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)

        # ── Output projection ─────────────────────────────────────────────────
        # Recombines the per-head outputs back into d_model space.
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask: upper triangle = -inf → future tokens invisible
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) / scale
        attn  = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), attn.detach()   # return attn for visualisation


class TransformerBlock(nn.Module):
    """One transformer decoder block: LayerNorm → Attention → Residual →
                                       LayerNorm → FFN → Residual."""

    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()

        # ── Pre-norm (applied BEFORE attention) ───────────────────────────────
        # Normalising before attention keeps activations in a stable range,
        # making training faster and less sensitive to learning rate.
        self.ln1  = nn.LayerNorm(d_model)

        # ── Causal self-attention ─────────────────────────────────────────────
        # The model reads the sequence and decides, for each position, which
        # earlier positions are relevant.  A move at step t can attend to all
        # pile-value tokens (the "game state") and all previous moves.
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)

        self.ln2  = nn.LayerNorm(d_model)

        # ── Feed-forward network (per position) ───────────────────────────────
        # After attending, each position is transformed independently.
        # The 4× expansion then compression is standard: the wide layer
        # can represent complex functions; the narrow output is forced to
        # summarise them.  GELU is a smooth approximation of ReLU, standard in
        # modern transformers (BERT, GPT).
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, FFN_MULT * d_model),
            nn.GELU(),
            nn.Linear(FFN_MULT * d_model, d_model),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out           # residual: gradient highway
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class NimTransformer(nn.Module):
    """Small decoder-only transformer for Nim move prediction."""

    def __init__(self, max_len: int = MAX_SEQ_LEN, d_model: int = D_MODEL):
        super().__init__()
        # d_model is a parameter so that checkpoints trained with a different
        # size can be loaded without changing the module-level constant.

        # ── Token embedding ───────────────────────────────────────────────────
        # Maps each of the 32 vocabulary tokens to a d_model-dimensional vector.
        # Pile tokens and move tokens share this space — the model learns
        # a geometry where "pile=3" and "move: take-3-from-pile-0" are related.
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)

        # ── Positional embedding ──────────────────────────────────────────────
        # Unlike RNNs, transformers have no inherent sense of order.
        # This learned table adds a position-specific signal so the model
        # knows "this is the 2nd pile value" vs "this is the 5th move".
        self.pos_emb = nn.Embedding(max_len, d_model)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, N_HEADS, max_len)
            for _ in range(N_LAYERS)
        ])

        # ── Final layer norm ──────────────────────────────────────────────────
        # Normalise before the LM head for stable training.
        self.ln_f    = nn.LayerNorm(d_model)

        # ── LM head ───────────────────────────────────────────────────────────
        # Projects the final hidden state to VOCAB_SIZE logits.
        # At move positions, only the 18 move-token logits (14–31) matter.
        # Tie weights with token embedding (standard LLM practice — reduces
        # parameters and encourages consistent representations).
        self.head    = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        self._max_len = max_len

    def forward(self, tokens: torch.Tensor):
        """tokens: (B, T) int64 → logits: (B, T, VOCAB_SIZE), attn_list."""
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device)

        # Sum token and positional embeddings — the input to the transformer
        x    = self.tok_emb(tokens) + self.pos_emb(pos)

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        return self.head(x), attn_maps


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    train_path:   str,
    test_path:    str,
    out_dir:      str   = "models",
    epochs:       int   = 30,
    batch_size:   int   = 256,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading and preprocessing datasets...")
    Xtr, ytr, Ltr = preprocess_dataset(*load_dataset(train_path))
    Xte, yte, Lte = preprocess_dataset(*load_dataset(test_path))
    print(f"  Train: {Xtr.shape[0]} sequences | Test: {Xte.shape[0]} sequences")

    # Determine max sequence length from actual data
    max_len = max(Xtr.shape[1], Xte.shape[1])
    if Xtr.shape[1] < max_len:
        Xtr = F.pad(Xtr, (0, max_len - Xtr.shape[1]))
    if Xte.shape[1] < max_len:
        Xte = F.pad(Xte, (0, max_len - Xte.shape[1]))

    loader    = DataLoader(TensorDataset(Xtr, ytr, Ltr), batch_size=batch_size, shuffle=True)
    model     = NimTransformer(max_len=max(max_len, MAX_SEQ_LEN))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history   = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0; correct = 0

        for Xb, yb_soft, Lb in loader:
            logits, _ = model(Xb)
            # Extract logits at the LAST REAL TOKEN position for each sequence.
            # Sequences are padded to the same length; Lb[i] holds the number of
            # real tokens in sequence i, so position Lb[i]-1 is the last real one.
            # Taking logits[:, -1, :] would read from padding for short sequences.
            idx         = (Lb - 1).clamp(min=0)                    # (B,)
            pred_logits = logits[torch.arange(len(idx)), idx, :]    # (B, VOCAB)
            # Only the move-token slice of the logits matters for our task
            move_logits = pred_logits[:, MOVE_OFFSET:MOVE_OFFSET + N_MOVES]  # (B, 18)
            log_probs   = F.log_softmax(move_logits, dim=-1)
            loss        = F.kl_div(log_probs, yb_soft, reduction='batchmean')
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            total_loss += loss.item() * len(yb_soft)
            preds   = move_logits.argmax(1)
            correct += (yb_soft[torch.arange(len(preds)), preds] > 0).sum().item()

        train_loss = total_loss / len(ytr)
        train_acc  = correct    / len(ytr)
        model.eval()
        with torch.no_grad():
            te_logits, _ = model(Xte)
            te_idx   = (Lte - 1).clamp(min=0)
            te_pred  = te_logits[torch.arange(len(te_idx)), te_idx, MOVE_OFFSET:MOVE_OFFSET + N_MOVES]
            te_preds = te_pred.argmax(1)
            test_acc = (yte[torch.arange(len(te_preds)), te_preds] > 0).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} | "
                  f"train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    model_path = os.path.join(out_dir, "arch3_transformer.pt")
    torch.save({"state_dict": model.state_dict(), "max_len": model._max_len}, model_path)
    print(f"\nModel saved → {model_path}")
    return history, model


# ── Inference ──────────────────────────────────────────────────────────────────

class TransformerAgent:
    """Wraps the trained transformer for stateless inference.

    The model was trained on (current_pile_state → optimal_first_move), so the
    correct way to use it is: at every turn, build a fresh 7-token prompt from
    the CURRENT pile values and do a single forward pass.

    No move history is needed or used.  This is correct because Nim is Markov:
    the optimal move depends only on the current pile state, not on how that
    state was reached.  Appending move history would give the model out-of-
    distribution input (it never saw sequences longer than 7 tokens during
    training) and using only the agent's own moves would give it a wrong picture
    of the board anyway, since the opponent's moves are missing.
    """

    def __init__(self, model_dir: str = "models"):
        checkpoint = torch.load(
            os.path.join(model_dir, "arch3_transformer.pt"), map_location="cpu"
        )
        # Infer d_model from the saved weights — robust to any checkpoint,
        # regardless of whether d_model was explicitly stored.
        saved_d = checkpoint["state_dict"]["tok_emb.weight"].shape[1]
        self.model = NimTransformer(max_len=checkpoint["max_len"], d_model=saved_d)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def next_move(self, piles) -> tuple:
        """Build prompt from current piles (sorted descending), run one forward pass, return move.

        Piles are sorted before encoding to match the training canonical form.
        The returned pile index is translated back to the original (unsorted)
        position so the caller can apply the move to its own pile array.
        """
        orig  = pad_piles(list(piles))
        order = sorted(range(len(orig)), key=lambda i: orig[i], reverse=True)
        sorted_piles = [orig[i] for i in order]

        seq    = [pile_token(c) for c in sorted_piles] + [SEP_TOKEN]
        tokens = torch.tensor([seq], dtype=torch.long)

        with torch.no_grad():
            logits, _ = self.model(tokens)

        move_logits = logits[0, -1, MOVE_OFFSET:MOVE_OFFSET + N_MOVES].numpy()
        mask        = illegal_move_mask(sorted_piles)
        move_logits[mask] = -np.inf

        sorted_pile_idx, take = decode_move(int(np.argmax(move_logits)))
        return (order[sorted_pile_idx], take)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualise(model: NimTransformer, history: dict, out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Architecture 3 — Causal Transformer", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ── 1 & 2. Training curves ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"]); ax.set_title("Training loss")
    ax.set_xlabel("Epoch"); ax.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["train_acc"], label="Train"); ax.plot(history["test_acc"], label="Test")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # ── 3. Attention map for a sample game ────────────────────────────────────
    # Show what the last transformer block attends to when predicting move 1.
    ax = fig.add_subplot(gs[0, 2:])
    _plot_attention(model, ax)

    # ── 4. Token embedding similarity ─────────────────────────────────────────
    # Are pile-value tokens and move tokens geometrically related?
    ax  = fig.add_subplot(gs[1, 0:2])
    emb = model.tok_emb.weight.detach().numpy()   # (32, D_MODEL)
    sim = emb @ emb.T
    # Normalise to cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    sim   = sim / (norms @ norms.T + 1e-8)
    labels = [f"p{i}" for i in range(13)] + ["SEP"] + \
             [f"m{i}" for i in range(N_MOVES)]
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(VOCAB_SIZE)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(VOCAB_SIZE)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Token embedding cosine similarity\n(pile tokens vs move tokens)")
    plt.colorbar(im, ax=ax)

    # ── 5. N vs P accuracy ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _plot_grundy_accuracy(model, ax)

    # ── 6. Sequence length vs accuracy ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    _plot_length_accuracy(model, ax)

    path = os.path.join(out_dir, "arch3_transformer.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}")


def _plot_attention(model: NimTransformer, ax) -> None:
    """Show attention weights for the second block's first head on a sample game."""
    import random
    rng   = random.Random(0)
    piles = pad_piles([rng.randint(2, 8) for _ in range(4)])
    from game import optimal_moves as opt
    import game as gm
    # Build a short trajectory
    seq   = [pile_token(c) for c in piles] + [SEP_TOKEN]
    cur   = list(piles)
    for _ in range(3):
        if gm.is_terminal(cur): break
        m   = rng.choice(opt(cur))
        seq.append(move_token(m))
        cur = apply_move(cur, m)

    tokens = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        _, attn_maps = model(tokens)

    # Last block, first head
    attn = attn_maps[-1][0, 0].numpy()   # (T, T)
    T    = len(seq)
    tok_labels = []
    for t, tok in enumerate(seq):
        if tok < N_PILE_TOKENS:
            tok_labels.append(f"p={tok}")
        elif tok == SEP_TOKEN:
            tok_labels.append("SEP")
        else:
            p, take = decode_move(tok - MOVE_OFFSET)
            tok_labels.append(f"t{take}p{p}")

    im = ax.imshow(attn[:T, :T], cmap="Blues")
    ax.set_xticks(range(T)); ax.set_xticklabels(tok_labels, rotation=45, fontsize=7)
    ax.set_yticks(range(T)); ax.set_yticklabels(tok_labels, fontsize=7)
    ax.set_title("Attention (block 2, head 1)\nrows=query, cols=key")
    plt.colorbar(im, ax=ax)


def _plot_grundy_accuracy(model: NimTransformer, ax) -> None:
    import random, game as gm
    rng   = random.Random(7)
    stats = {"N": [0, 0], "P": [0, 0]}
    for _ in range(500):
        piles  = pad_piles([rng.randint(1, MAX_COINS) for _ in range(rng.randint(2, MAX_PILES))])
        label  = "N" if is_misere_winning(piles) else "P"
        # Single-step inference: just the prefix
        seq    = [pile_token(c) for c in piles] + [SEP_TOKEN]
        tokens = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(tokens)
        ml = logits[0, -1, MOVE_OFFSET:MOVE_OFFSET + N_MOVES].numpy()
        mask = illegal_move_mask(piles); ml[mask] = -np.inf
        pred = int(np.argmax(ml))
        stats[label][0] += int(pred in optimal_moves(piles))
        stats[label][1] += 1
    labels = ["N-position", "P-position"]
    accs   = [stats["N"][0] / max(stats["N"][1], 1),
              stats["P"][0] / max(stats["P"][1], 1)]
    bars   = ax.bar(labels, accs, color=["#a6e3a1", "#f38ba8"])
    ax.set_ylim(0, 1); ax.set_title("Grundy accuracy: N vs P"); ax.grid(axis="y")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", fontsize=10)


def _plot_length_accuracy(model: NimTransformer, ax) -> None:
    """Accuracy by pile-size range — tests generalisation across the value space.

    Comparable to the same plot in arch2_scalar.py.  Since the model is stateless
    (prompt = current piles), this is the natural generalisation axis.
    """
    import random
    rng    = random.Random(99)
    ranges = [(1, 4), (5, 8), (9, 12)]
    accs   = []
    for lo, hi in ranges:
        correct = total = 0
        for _ in range(500):
            piles  = pad_piles([rng.randint(lo, hi)
                                for _ in range(rng.randint(2, MAX_PILES))])
            seq    = [pile_token(c) for c in piles] + [SEP_TOKEN]
            tokens = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(tokens)
            ml   = logits[0, -1, MOVE_OFFSET:MOVE_OFFSET + N_MOVES].numpy()
            mask = illegal_move_mask(piles); ml[mask] = -np.inf
            pred = int(np.argmax(ml))
            correct += int(pred in optimal_moves(piles))
            total   += 1
        accs.append(correct / max(total, 1))
    bars = ax.bar([f"{lo}–{hi}" for lo, hi in ranges], accs, color="#89b4fa")
    ax.set_ylim(0, 1); ax.set_title("Accuracy by pile-size range")
    ax.set_xlabel("Pile value range"); ax.set_ylabel("Accuracy"); ax.grid(axis="y")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}", ha="center", fontsize=9)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Architecture 3: transformer Nim model")
    parser.add_argument("--train",   default="data/random_train")
    parser.add_argument("--test",    default="data/random_test")
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--plots",   default="plots")
    args = parser.parse_args()

    history, model = train(
        train_path=args.train, test_path=args.test,
        out_dir=args.out_dir, epochs=args.epochs,
    )
    visualise(model, history, out_dir=args.plots)
