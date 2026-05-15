# ☠ The Trap Room — Minimax Challenge

A self-contained, browser-based coding challenge focused on **game tree search**.  
No server required — open `index.html` and start coding.

---

## Concept

The Trap Room is a 1v1 adversarial game set in the escape room universe. Instead of a random monster that wanders the maze, you now face a **rational opponent** — one whose every move is designed to make you lose. Your task is to implement that opponent.

This challenge introduces **minimax**, the foundational algorithm for two-player, zero-sum, perfect-information games. It is the natural next step after the pathfinding challenges in v2: where Dijkstra and A\* reason about a single agent navigating a static or stochastic environment, minimax reasons about two agents with opposing objectives.

---

## The Game — misère Nim

> You and the Monster stand over piles of cursed gold.  
> On your turn, pick one pile and remove **1, 2, or 3 coins** from it.  
> **Whoever takes the last coin triggers the trap — and dies.**

This is a variant of the classical combinatorial game **Nim**, specifically **misère Nim** with a bounded move set (1–3 coins per turn). The rules are:

- There are N piles of coins (configurable).
- Players alternate turns. You cannot pass.
- On your turn you must pick exactly one pile and remove 1, 2, or 3 coins from it.
- The player who takes the **last coin loses**.

The bounded move set (1–3 instead of unlimited) makes the game less trivially solvable by hand, raises the node count of the search tree, and makes the connection to expectimax more natural when extending to stochastic opponents.

---

## Files

```
trap_room/
└── index.html      # the entire challenge — editor, Python runtime, game UI
```

One file, no dependencies to install, no build step. Requires an internet connection to load Pyodide (Python runtime) and CodeMirror (editor) from CDN.

---

## How to play

1. Open `index.html` in a browser (Chrome recommended; serve over HTTP if using Firefox — see note below).
2. Wait for the **Python ready** badge in the top-right corner.
3. Write your Monster agent in the left-hand editor.
4. Click **▶ Load agent** to validate and instantiate it.
5. Use the config panel to set pile count, max coins per pile, and who moves first.
6. Click **New game** to start, then play by clicking a pile and entering how many coins to take.
7. Click **↺ Same piles** to replay the identical starting position with a different agent.

> **Note on `file://`:** Pyodide fetches its WebAssembly runtime from CDN. Chrome allows this from `file://`; Firefox blocks it. To use Firefox, serve the file locally: `python -m http.server 8080` then open `http://localhost:8080/trap_room/`.

---

## Agent interface

Implement a Python class called `Agent`:

```python
class Agent:
    def __init__(self, piles):
        """
        Called once at the start of each game.

        piles : list of ints — initial pile sizes, e.g. [3, 4, 5]

        Use this to precompute anything that depends only on the
        initial configuration (e.g. transposition table, game value).
        """
        pass

    def next_move(self, piles):
        """
        Called on every Monster turn.

        piles : list of ints — current pile sizes

        Return a tuple (pile_index, amount) where:
          pile_index : int — which pile to take from (0-based)
          amount     : int — how many coins to remove (1, 2, or 3;
                             cannot exceed piles[pile_index])

        Example: (1, 2) means "take 2 coins from pile 1".
        """
        return (0, 1)  # fallback: always take 1 from pile 0
```

`print()` works normally — output appears in the log panel at the bottom.

---

## Starter implementation — Minimax

The **Reset code (Minimax)** button loads a complete working agent that implements the **negamax** formulation of minimax (a cleaner way to write minimax when both players use the same value function):

```python
def _minimax(self, piles):
    if all(p == 0 for p in piles):
        return 1          # current player wins: opponent took the last coin

    best = -1
    for i, p in enumerate(piles):
        for amount in range(1, min(3, p) + 1):
            new_piles = list(piles)
            new_piles[i] -= amount
            val = -self._minimax(tuple(new_piles))   # negate: opponent's gain is our loss
            if val > best:
                best = val
    return best
```

The terminal condition deserves careful reading: when all piles are zero, the player **to move** wins — because it was the **previous** player who took the last coin and thus triggered the trap. Getting this boundary condition right is the most common source of bugs.

The sample also counts and prints visited nodes per turn, so you can directly observe how the search tree grows as pile sizes increase.

---

## Suggested exercises

### 1 — Understand the tree
Run the minimax agent on a single pile of 4 coins. Draw the full game tree by hand (it has fewer than 20 nodes). Verify that the printed node count matches. Who wins with optimal play?

### 2 — Memoization
The same state `(pile_a, pile_b, pile_c)` can be reached by different move sequences. The naive minimax recomputes it every time. Add a `dict` cache (transposition table) keyed on the pile tuple:

```python
def _minimax(self, piles):
    if piles in self.cache:
        return self.cache[piles]
    # ... rest of the search ...
    self.cache[piles] = best
    return best
```

Measure the reduction in visited nodes. At what pile sizes does memoization become essential?

### 3 — Alpha-beta pruning
In the negamax loop, once you find a move with value `+1` you can return immediately — you cannot do better. Generalise this: pass a window `(alpha, beta)` down the tree and prune branches that cannot improve the current best:

```python
def _minimax(self, piles, alpha=-1, beta=1):
    ...
    for move in moves:
        val = -self._minimax(child, -beta, -alpha)
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break          # beta cutoff — prune remaining moves
    return alpha
```

Count visited nodes with and without pruning on the same starting position. Plot the reduction as a function of pile depth.

### 4 — Iterative deepening
Alpha-beta pruning is most effective when better moves are searched first (move ordering). Add a simple heuristic: prefer moves that leave the opponent in a position where the XOR of pile sizes is non-zero (a known losing signal in standard Nim). Does move ordering reduce the node count?

### 5 — From minimax to expectimax
The minimax agent assumes the opponent plays **optimally**. What if the human makes random or suboptimal moves? Replace the MIN node with a **chance node** that averages over the opponent's possible moves (uniform distribution). Does the agent play differently? When does it matter?

This exercise bridges directly to the **expectimax** algorithm used in the escape room v2 challenge for reasoning about randomly moving monsters.

### 6 — Mathematical insight
After playing several games, look at the pile sizes just before you lose. Do you notice a pattern? Research the **Sprague-Grundy theorem** and the role of XOR in Nim. For the bounded (1–3 per move) variant, the winning condition is related to pile sizes modulo 4. Verify this against the minimax values your agent computes.

---

## Connection to the escape room series

| Challenge | Agent model | Opponent model | Algorithm |
|---|---|---|---|
| v1 — Web Security | scripted client | none (timed service) | automation |
| v2 — Pathfinding | rational player | random monsters | Dijkstra / A\* / local search |
| **Trap Room** | **rational player** | **rational monster** | **minimax** |
| *(next)* | rational player | random + rational | expectimax |

The progression models increasing sophistication in the opponent: first no opponent, then a random one, then a fully rational adversary. Expectimax unifies the last two — it handles opponents that are partly random and partly rational, which is the realistic model for the escape room monsters when they are given a strategy.

---

## Technical notes

### Python runtime
Pyodide 0.26 (CPython 3.11 in WebAssembly). The full standard library is available. The agent runs synchronously — the game waits for `next_move` to return before updating the UI, so there is no timeout. For large pile sizes the search may block the browser tab for several seconds; memoization (exercise 2) is the fix.

### JS ↔ Python bridge
- Pile state is passed as a Python list via `pyodide.toPy()`.
- The return value `(pile_index, amount)` is extracted by calling `int(_move[0])` and `int(_move[1])` separately, avoiding tuple proxy issues.
- `print()` is redirected to the log panel via a custom `sys.stdout` writer that calls a JS function exposed through `pyodide.globals.set`.
