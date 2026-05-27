# Neural Trap Room — ML Challenge

> *Learning deep learning by watching three networks fail, adapt, and eventually learn misère Nim.*

This project extends the Trap Room Nim challenge into a machine learning study. Three neural network architectures — a one-hot feedforward net, a scalar feedforward net, and a small transformer — are trained to play misère Nim optimally, then pitted against a human player through a local web server. The point is not to arrive at a working model in one step, but to follow the full arc: bad results, diagnosis of why, targeted fixes, and observation of how the diagnostic plots change as understanding improves.

---

## Project structure

```
game.py                  Nim rules, Grundy theorem, move encoding, legal-move masking
dataset_random.py        Random-play game generator (noisy labels)
dataset_rational.py      Optimal-play dataset using Grundy theory (clean labels)
arch1_onehot.py          Architecture 1: 78-dim one-hot input
arch2_scalar.py          Architecture 2: 12-dim scalar + mod-4 Grundy features
arch3_transformer.py     Architecture 3: causal decoder transformer
run_experiments.py       Full training pipeline with cross-dataset experiments
serve_trap_room.py       Local HTTP server — play against trained models in browser
models/                  Trained .pt checkpoints and .npy numpy exports
data/                    Generated datasets (.npz)
plots/                   Diagnostic plots per architecture
```

---

## Quickstart

```bash
# Generate datasets and train all three architectures (50 epochs each)
python run_experiments.py --epochs 50

# Or select a single architecture
python run_experiments.py --arch 2 --epochs 50

# Cross-dataset experiment: train on random play, test on rational
python run_experiments.py --train-set random --test-set rational --epochs 50

# Play against the trained models in the browser
python serve_trap_room.py
# → open http://localhost:8765
```

Dependencies: `torch`, `numpy`, `matplotlib`. No other packages required. The server uses only Python stdlib (`http.server`, `json`).

---

## The game

Misère Nim with a 1–3 coin removal limit. Players alternate; on each turn a player picks one pile and removes 1, 2, or 3 coins from it. **The player who takes the last coin loses.** Up to 6 piles, each with up to 12 coins.

The Sprague-Grundy theorem gives the exact winning strategy in O(P) per move (no search needed): the Grundy value of a pile of size *n* under a take-1-2-3 rule is `n % 4`. The game is a first-player win iff the XOR of all pile Grundy values is non-zero — *except* when all piles are size ≤ 1, where the misère flip applies: you want to leave an odd number of size-1 piles for the opponent.

This matters pedagogically: the three architectures effectively have to rediscover parts of this theorem from game outcome data alone.

---

## Architecture 1 — One-hot encoding (78-dim)

**Input:** each pile (0–12 coins) encoded as a length-13 one-hot vector; six piles concatenated → 78-dimensional sparse binary vector.

**Network:** `Linear(78→128) → ReLU → Linear(128→64) → ReLU → Linear(64→18)`

**Output:** 18 logits, one per (pile_index × take_amount) move. Illegal moves masked to −∞ at inference.

**What it learns:** a lookup table. Two states that differ by a single pile are orthogonal 78-dim inputs — the model has no way to interpolate between them. Width-128 first layer can memorise many pile configurations; width-64 second layer forces compression.

### What the plots reveal

The **W1 heatmap** (32 neurons × 13 pile values for pile 0) shows each neuron responding to specific pile-value identities, not to any periodic structure. This is the lookup-table signature: each neuron fires for one or a few specific values, not for an equivalence class.

The **N vs P accuracy bars** tell the clearest story: this model hits >97% on N-positions (winnable games) but gets there by memorisation, not by understanding. The win-rate against a perfect opponent is much lower (~19–28%), because a single mistake in a full game against a minimax opponent is fatal and the model's errors are not independent.

---

## Architecture 2 — Scalar + mod-4 encoding (12-dim)

**Input:** 6 normalised pile values (pile/12) concatenated with 6 mod-4 Grundy features ((pile%4)/3) → 12-dimensional vector.

**Network:** `Linear(12→128) → ReLU → Linear(128→64) → ReLU → Linear(64→18)`

**What it learns:** the Grundy structure. The mod-4 channel hands the network the periodic equivalence classes directly; layer 1 approximates per-pile Grundy values, layer 2 approximates the XOR across piles.

### What the plots reveal

The **layer 1 activations vs pile 0 value** plot is the most pedagogically important result. After training, neurons 3–4 show bright activations at pile values 3, 7, 11 (mod-4 ≡ 3) and neurons 9–10 fire at 0, 4, 8, 12 (mod-4 ≡ 0). The network spontaneously discovered the Grundy periodicity from game outcome labels alone — it didn't need to be told about Grundy theory.

The **accuracy by pile-size range** bars are flat (88/93/92%), confirming genuine generalisation rather than memorisation. This is the key difference from arch1: because the input is ordered and continuous, the model can interpolate. Arch1 trained on piles ≤ 8 would completely fail on piles 9–12; arch2 would not.

The **Grundy XOR vs max logit** scatter still shows no clear separation between XOR=0 and XOR≠0, which explains why N-accuracy (~91%) is lower than arch1's (~97%): layer 2 has learned partial XOR combinations but not a reliable global game-value classifier.

---

## Architecture 3 — Causal transformer

**Input format:** a 7-token sequence `[p0, p1, p2, p3, p4, p5, SEP]` where each token is a vocabulary entry from a 32-token alphabet (13 pile-value tokens 0–12, 1 SEP token, 18 move tokens).

**Network:** token embedding (32→64) + positional embedding → 2× transformer block (causal self-attention with 4 heads + FFN) → LayerNorm → LM head (64→32 logits, weight-tied to embedding).

**Output:** logits at the SEP position, move-token slice (indices 14–31 = 18 moves).

**What it learns:** a joint function of all 6 pile values simultaneously via attention. Each head learns a different "question" to ask about the pile configuration; the FFN integrates the answers.

### What the plots reveal

The **attention heatmap** (block 2, head 1) shows the SEP token attending strongly to itself and accumulating information from all pile tokens before the LM head produces move logits. This is the transformer using SEP as a summary token — exactly the mechanism that BERT uses for classification tasks.

The **token embedding cosine similarity** matrix shows pile tokens (p0–p12) forming a cluster in the upper-left block, move tokens (m0–m17) forming a separate cluster in the lower-right, with SEP sitting at the boundary. This geometry is interpretable: the model learned that pile values and moves live in different parts of the semantic space.

---

## The pedagogical arc — errors observed and fixes applied

This section documents the full sequence of failures and what each taught. **Reading only the final solution throws away the most valuable part.**

### Round 1 — Hard one-hot targets (all architectures)

**Symptom:** accuracy plateaued at 25–33% even after many epochs. Loss barely moved.

**Diagnosis:** the loss was `cross_entropy(logits, one_hot_of_recorded_move)`. A Nim state with XOR=5 may have 4 equally valid winning moves. The recorded move is whichever one the dataset player happened to choose. The network was penalised for outputting a *different* valid winning move — correct answers were treated as errors.

**Fix:** replaced hard one-hot targets with **soft targets** computed via Grundy theorem. For each training state, `optimal_moves(piles)` returns all winning moves; probability mass is distributed uniformly across them. Loss changed to `kl_div(log_softmax(logits), soft_target)`. Accuracy metric changed to "did argmax land on any optimal move?" rather than "did it match the exact recorded move?".

**Lesson:** your loss function must reflect what you actually want to optimise. If multiple answers are correct, a one-hot target treats all-but-one as wrong, injecting noise that is indistinguishable from genuine errors.

### Round 2 — Train/test distribution mismatch

**Symptom:** 85% test accuracy but only 19–28% win rate against the minimax opponent. Per-move accuracy on fresh random starts: 51%.

**Diagnosis:** the training and test sets both came from mid-game trajectory samples. Mid-game states have heavily depleted piles (20% of pile values were size 1) because games start from random positions and coin counts only decrease. The evaluation harness generates fresh random starting positions with piles uniformly in 1–12. The model learned the mid-game distribution but was tested on a starting-position distribution it had never seen.

The 85% "test accuracy" was not measuring generalisation — it was measuring how well the model memorised the same distribution it trained on, dressed up as a test set.

**Fix:** replaced trajectory-based dataset generation with `generate_starting_position_dataset`: sample 100k random starting configurations, compute optimal moves via Grundy (no games played), save `(state, optimal_move)` pairs directly. Pile value mean shifted from 3.2 (depleted mid-game) to 6.5 (uniform starting positions).

**Lesson:** train/test accuracy tracking each other closely does not mean the model generalises — it means you have not measured generalisation. You need evaluation on a distribution that is structurally different from training. The right question is not "does it score well on held-out data from the same pipeline?" but "does it work on the actual problem?"

### Round 3 — Arch2 not learning periodicity

**Symptom:** layer 1 activations showed monotonically increasing responses to pile value — "bigger pile = more activation" — with no period-4 pattern. N-accuracy 33%.

**Diagnosis:** the raw scalar input `pile / 12` represents values 1, 5, 9 as 0.08, 0.42, 0.75. These look nothing alike numerically, even though their Grundy values are identical (all ≡ 1 mod 4). The model had to rediscover the mod-4 periodicity from scratch with only ReLU units, which is possible but requires significant capacity.

Additionally, the model was too narrow: `6→64→32→18` with 6-dim input could not represent enough independent Grundy detectors in the first layer.

**Fix:** two changes together. (1) Added explicit mod-4 features to the input — `[pile/12 for each pile] + [pile%4/3 for each pile]` → 12-dimensional input. This hands the network the equivalence classes directly. (2) Widened both hidden layers: `12→128→64→18`.

**Observation:** after this fix, the layer 1 activation plot showed the expected period-4 pattern — specific neurons firing at 3,7,11 or at 0,4,8,12 or similar equivalence classes. The model had been given the right inductive bias.

**Lesson:** representation encodes assumptions. A continuous scalar input says "values near each other in magnitude are similar." A mod-4 feature says "values in the same Grundy class are equivalent." Neither is inherently correct — it depends on the problem structure. When you know the structure, encoding it explicitly makes the model's job dramatically easier and its learned representations interpretable.

### Round 4 — Arch3 attending to move history, not pile state

**Symptom:** training accuracy hit 94% but win rate was 13% — paradoxically the worst of all three. The context-length diagnostic plot showed accuracy dropping from 100% at 0 context moves to ~50% at 2 moves and staying flat.

**Diagnosis:** the original arch3 preprocessing generated full game trajectories — sequences like `[p0..p5 SEP m0 m1 m2 ...]` where move tokens were appended as the game progressed. The model discovered that recent moves are strongly correlated with the next move (good moves follow good moves in rational trajectories), and learned to predict from move history rather than from the pile-value prefix. The attention map confirmed this: move tokens had high self-attention weights and the pile-value prefix was largely ignored.

The inference code compounded the problem: the agent maintained a personal move history and appended only its own moves — missing the opponent's moves entirely. The model was predicting from a hallucinated game state that bore no relation to the actual board.

**Fix:** two changes. (1) Changed preprocessing to emit only the first move of each game with **no move history** — every training sequence is exactly `[p0..p5 SEP]`, target is the optimal first move. This forces the model to predict from the pile state alone. (2) Rewrote `TransformerAgent.next_move` to be **stateless**: at every call, build a fresh 7-token prompt from the current pile values and do a single forward pass. No history, no `reset()`, no state between calls.

**Why this is correct:** Nim is Markov. The optimal move depends only on the current pile configuration, not on how that configuration was reached. Conditioning on move history is not just unnecessary — it provides misleading signal when the model can see only its own moves but not the opponent's.

**Lesson:** a model will learn any pattern you show it, including patterns that are correlated with the target but causally irrelevant. Transformer models are particularly good at finding sequential shortcuts. If you want the model to learn a state-based function, make sure the training distribution expresses that function, not a path-dependent proxy for it.

### Round 5 — Single-pile endgame failures

**Symptom:** with a single pile, all three models played incorrectly. Pile=3: models took all 3 coins and self-destructed (triggered the trap). Pile=8: models failed to apply the misère strategy.

**Diagnosis:** the dataset generator used `n_piles = randint(2, MAX_PILES)` — minimum 2 piles. Single-pile states only appear in actual gameplay once all other piles have been depleted, but training data contained only starting positions. Single-pile endgames were 100% out of distribution.

Pile=3 exposed a deeper issue: in *normal* Nim (last taker wins), the correct move from a single pile of 3 is to take all 3 and win. The models had learned an approximation of normal-Nim Grundy logic, and without single-pile examples they defaulted to this — catastrophically wrong for misère Nim where the correct move is "take 2, leave 1 for the opponent."

**Fix:** changed the dataset generator to include 20% single-pile starting positions by sampling `n_piles = 1` with probability 0.2. This matches roughly how often single-pile states appear in real games.

**Lesson:** "test on your actual deployment distribution." The evaluation harness measured accuracy on starting positions, but real games always end in single-pile endgames. A model that has never seen its most common deployment scenario will fail there.

### Round 6 — Permutation noise across pile indices

**Symptom:** the state `[5, 3, 0, 0, 0, 0]` and `[0, 3, 0, 5, 0, 0]` are the same Nim game but produce different model inputs. Arch1 sees orthogonal 78-dim vectors with zero dot product. Arch2 sees completely different 12-dim inputs. Arch3 attends to pile tokens at different positions.

**Fix:** added a single sort step at the start of `encode_state` in all three architectures: `piles = sorted(pad_piles(list(piles)), reverse=True)`. The same sort is applied at inference time in every agent's `next_move`. The sort also happens before calling `optimal_moves` in the preprocess loop, so move IDs reference the sorted pile positions consistently.

**Effect:** all permutations of the same multiset now map to one canonical form. This collapses up to 6! = 720 equivalent training states into one, effectively giving each state 720× more training signal. Verified: all three models give identical answers for all 720 permutations of `[5, 3, 7, 0, 0, 0]`.

**Lesson:** if the task has symmetries, encode them. Treating `[5,3,7]` and `[7,3,5]` as different inputs is like training a function approximator on `f(x,y)=x+y` with examples `(3,5)=8` and `(5,3)=8` treated as different functions — you're wasting capacity and injecting noise. Canonical forms are free, exact, and always beneficial when the symmetry is genuine.

### Round 7 — Arch3 inference with stale move history

**Symptom:** (related to Round 4, but a separate bug in the inference path.) The agent maintained `_initial_piles` and `_move_history`, appending its own move IDs after each call. This caused two distinct bugs: (a) the opponent's moves were never recorded, so after one exchange the model's internal sequence diverged completely from the actual board state; (b) the sequence length grew beyond 7 tokens, which the model had never seen during training (all training sequences were exactly 7 tokens).

**Fix:** `TransformerAgent` was rewritten to be fully stateless. `reset()` was removed. The only state is the model weights. Every `next_move(piles)` call builds `[sorted_p0..sorted_p5 SEP]` from the current pile argument and does one forward pass.

**Lesson:** in sequential games, it is tempting to maintain a history buffer "for context." Before doing so, ask whether the model was trained to use that context, and whether you can actually supply it correctly (you cannot supply opponent moves if you don't observe them internally). For Markov games, stateless inference is not just simpler — it is more correct.

---

## Why win rate is low despite high accuracy

All three models reach per-move accuracy of 90–99%, but win rates against a perfect opponent are in the 19–31% range. This is not a training failure — it is arithmetic.

A game requires roughly 8 correct sequential decisions against a perfect adversary. If per-move accuracy is *p* and errors are independent, the probability of winning is at most *p^8*. At *p*=0.97 that is 78%; at *p*=0.91 it is 47%. The observed win rates are lower than these bounds, which means errors are *not* independent — the model tends to fail on the specific states where a mistake is immediately punished.

The theoretical ceiling is also not 100%: ~25% of random starting positions are P-positions (the moving player loses with optimal play regardless). A perfect neural network would win ~75% of games, the same as the minimax reference agent.

**The deeper point:** accuracy measures "how often does this move match the reference on a random sample?" Win rate measures "can you string 8+ correct moves together against an adversary designed to exploit any error?" These are very different properties, and the gap between them is a direct measure of consistency. Reinforcement learning (training on full game outcomes rather than per-move supervision) is the natural way to close this gap — the model learns to value long-term correctness, not just per-step accuracy.

---

## The cross-dataset experiment

Training on noisy random-play data and testing on clean rational data is the most diagnostic experiment in the suite.

```bash
python run_experiments.py --train-set random --test-set rational --epochs 50
```

Random-play labels are noisy: a random player sometimes wins from a losing position, labelling that move as +1 even though it was sub-optimal. The model trained on this data learns a noisy approximation of the optimal policy. When evaluated on the rational test set — which tests the true optimal policy — accuracy drops. The gap between random-train/rational-test accuracy and rational-train/rational-test accuracy measures exactly how much label noise costs.

---

## Reading the plots

Each architecture produces a 6-panel diagnostic figure.

**Training loss / accuracy curves.** Train and test accuracy tracking each other closely means you are in the underfitting regime with matched distributions — not that the model generalises. To observe a genuine gap, evaluate on a structurally different distribution (different pile ranges, different number of piles, cross-dataset experiment).

**W1 heatmap (arch1).** Rows are neurons, columns are pile values 0–12 for pile 0. A noisy/random-looking pattern indicates lookup-table behaviour — neurons fire for specific pile-value identities. A periodic pattern with period 4 would indicate Grundy learning, which arch1 cannot achieve because its orthogonal inputs give it no basis for generalisation.

**Layer 1 activations vs pile value (arch2).** Rows are neurons, columns are pile values 0–12. The expected signature of Grundy learning: some rows bright at {3,7,11}, others at {0,4,8,12}, others at {1,5,9}, others at {2,6,10}. A monotonically increasing pattern (all rows brighter at larger pile values) indicates the model learned magnitude but not periodicity.

**Attention heatmap (arch3).** Rows=query position, columns=key position. For a pile-state-only input of length 7, each row shows which positions the model attends to when processing that position. The SEP token (row 6) attending broadly across all pile tokens indicates the model is aggregating pile information before making a prediction — the correct behaviour. The SEP token attending only to itself would indicate the model is ignoring the pile state.

**Token embedding cosine similarity (arch3).** A 32×32 matrix. Two visible clusters — pile tokens (top-left) and move tokens (bottom-right) — with low cross-cluster similarity indicates the model has learned a meaningful partition of the embedding space. High cosine similarity between pile token *k* and move token *m* would suggest the model represents "pile has value k" and "take from pile" as related concepts.

**N vs P accuracy bars.** N-positions are winnable (model should find the winning move); P-positions are already lost (any move is equally sub-optimal, but the model should at least not trigger the trap). A perfect model would show ~100% on both. Low N-accuracy with high P-accuracy indicates the model is conservative — it knows what not to do but not what to do. The reverse would be more dangerous in practice.

**Accuracy by pile-size range (arch2, arch3).** Flat bars across 1–4, 5–8, 9–12 indicate true generalisation — the model learned a rule, not a table. Bars that drop sharply for larger pile sizes indicate distribution sensitivity.

---

## The browser server

`serve_trap_room.py` is a self-contained Python HTTP server (no Flask, no external dependencies). It serves a modified Trap Room UI where the code editor is replaced by a model selector, and the agent's `next_move` is computed server-side.

```
GET  /          → serves the full game HTML page (built once, then cached)
GET  /models    → JSON list of available models (only those with .pt files present)
POST /move      → {"model": "arch3_transformer", "piles": [5,3,0,0,0,0]}
                ← {"pile": 0, "amount": 2, "model": "arch3_transformer"}
```

Models are loaded lazily on first request and cached. If a `.pt` file is missing, that model does not appear in the selector. The `minimax` option is always available and uses the Grundy-based `optimal_moves` function — it is the reference baseline that the neural models are trying to match.

The browser makes a `fetch('/move', ...)` call on each monster turn. The round-trip is typically under 5ms for the feedforward models and under 10ms for the transformer. The game loop is otherwise identical to the original Trap Room.

---

## Regenerating everything

```bash
# Delete datasets to force regeneration, then run full pipeline
rm -f data/*.npz
python run_experiments.py --epochs 50

# Or force regeneration without deleting files
python run_experiments.py --force-regen --epochs 50
```

The rational dataset uses `generate_starting_position_dataset` which samples starting positions directly from `optimal_moves` — no games are played, generation takes ~2 seconds for 100k samples. The random dataset plays full games with random-move agents.

Both datasets include ~20% single-pile starting positions to ensure the models learn the misère endgame correctly.

All pile states are sorted in decreasing order before encoding (preprocessing) and before inference (in each agent's `next_move`). This must be consistent — if you change the sort order, retrain from scratch.
