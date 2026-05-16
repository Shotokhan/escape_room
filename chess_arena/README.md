# ♞ Chess Arena

> *A multi-agent pathfinding challenge — write a Python knight agent, pit it against rivals and bots, and be the first to escape the maze.*

Chess Arena is a browser-based programming challenge that runs entirely client-side via [Pyodide](https://pyodide.org). No server, no install — open the page, write your agent, and watch the knights battle.

---

## How it works

The arena is a procedurally generated (or hand-drawn) maze. Every cell is either a wall, a free floor tile, or the single exit cell marked **E**. All pieces — players and bots alike — move as **chess knights**: the familiar L-shaped jump of two squares in one direction and one in the perpendicular. The graph of reachable cells connected by valid knight moves is pre-computed and handed to every agent before the game starts.

Turns are **sequential and round-robin**: each agent takes one move per round, in index order. An agent may also choose to **wait** (return its current node). The game ends when every human player has either escaped or been eliminated.

---

## The objective

- **Reach the exit (E)** before the other human players.
- **Avoid the bots** — they will chase or intercept you.
- Optionally, **eliminate rivals** by landing on their cell.

---

## Scoring

| Event | Points |
|---|---|
| 1st human to escape | `N` pts (where N = total human players) |
| 2nd human to escape | `N − 1` pts |
| … and so on | down to a minimum of **1 pt** |
| Killed by a bot / eliminated | **0 pts** |
| Bot kills a human or bot | **+1 pt** to the bot (bots are ranked too) |
| Stalemate (same board state repeats 3 times) | **0 pts** for all non-escaped players |

Escaped players keep their points even if the game ends in stalemate.

---

## Configuring a game

All controls are in the left panel.

| Control | Description |
|---|---|
| **human players** | Number of Python agent slots (0–6) |
| **bots** | Number of built-in monster bots (0–6) |
| **turn timeout** | Seconds before a human agent's turn is force-skipped |
| **anim speed** | Visual animation speed (1 = slow, 10 = instant) |
| **monster AI** | Toggle between `rational` and `greedy` bot behaviour |

Use **New map** to generate a fresh random maze, **↺ Replay** to restart from the same seed, **▶ Start** to run the game, and **■ Stop** to halt at any point.

The **map complexity** slider (in the Map → Generate bar) controls maze density: 0 is an open field, 10 is a full recursive-backtracker maze with minimal extra connections.

You can also hand-draw a map using **Map → Design**: place the exit first, then walls, then starting positions for players and bots.

---

## Writing your agent

Switch to the **Players** tab. Each human player slot has its own independent Python file. The interface exposes a CodeMirror editor with syntax highlighting.

Your code must define a class named `Agent` with exactly two methods:

```python
class Agent:
    def __init__(self, graph, escape_node, player_idx):
        """Called once when the game initialises."""
        ...

    def next_move(self, state) -> int:
        """Called every turn. Must return a valid neighbour node id, or
        the current node id to wait."""
        ...
```

### The graph

`graph` is a list of node dicts, indexed by node id:

```python
graph[id] = {
    'id'   : int,          # same as the list index
    'row'  : int,          # grid row (0-based)
    'col'  : int,          # grid col (0-based)
    'edges': [
        {'to': int, 'weight': int},  # weight is always 1
        ...
    ]
}
```

`escape_node` is the integer id of the exit cell. `player_idx` is your player's index (stable for the whole game).

### The state dict (per turn)

```python
state = {
    'player_node'     : int,   # your current node id
    'escape_node'     : int,   # exit node id (same as __init__)
    'others'          : [      # all other alive, non-escaped agents
        {
            'idx'   : int,
            'node'  : int,
            'is_bot': bool
        },
        ...
    ],
    'dist_from_escape': [int, ...]  # BFS distance from escape for every node id
                                    # -1 means unreachable
}
```

### Rules for `next_move`

- Must return an **integer node id**.
- The returned node must be either your **current node** (wait) or a **direct neighbour** (one edge away). Any other value is treated as a wait.
- Returning a neighbour occupied by another player **eliminates that player** — landing on a bot eliminates you.
- Bots cannot be eliminated by other bots by stepping on them (they treat each other as obstacles), but a human can eliminate a bot by landing on it.
- If your agent raises an exception or times out, the engine skips your turn silently (you wait in place).

---

## Sample agent (A\*)

The built-in sample (accessible via **Reset code (A\* sample)**) runs A\* toward the exit, treating all other players' current cells as blocked. It also opportunistically captures any rival reachable in one knight-move.

```python
import heapq

class Agent:
    def __init__(self, graph, escape_node, player_idx):
        self.graph       = graph
        self.escape_node = escape_node
        self.esc_row     = graph[escape_node]['row']
        self.esc_col     = graph[escape_node]['col']

    def _h(self, nid):
        n = self.graph[nid]
        return abs(n['row'] - self.esc_row) + abs(n['col'] - self.esc_col)

    def _astar(self, start, blocked):
        graph, goal = self.graph, self.escape_node
        n = len(graph)
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start] = 0
        heap = [(self._h(start), 0, start)]
        while heap:
            f, g, u = heapq.heappop(heap)
            if g > dist[u]: continue
            if u == goal: break
            for e in graph[u]['edges']:
                v, w = e['to'], e['weight']
                if v in blocked: continue
                ng = dist[u] + w
                if ng < dist[v]:
                    dist[v] = ng; prev[v] = u
                    heapq.heappush(heap, (ng + self._h(v), ng, v))
        if dist[goal] == float('inf'): return start
        cur = goal
        while prev[cur] != start and prev[cur] != -1: cur = prev[cur]
        return cur

    def next_move(self, state):
        pn = state['player_node']
        other_nodes = {o['node'] for o in state['others']}
        for e in self.graph[pn]['edges']:
            if e['to'] in other_nodes:
                return e['to']          # capture
        return self._astar(pn, other_nodes)
```

This is a reasonable baseline but has clear weaknesses: it ignores bot positions when routing, does not try to predict future moves, and has no threat-avoidance logic.

---

## Bot behaviour modes

The **monster AI** toggle switches all bots between two built-in strategies.

### Rational (default)

The rational bot is a defensive guardian. It:

1. Picks a **guard post** — a cell 1–6 knight-moves from the exit — and tries to hold it, so that no human can reach the exit uncontested.
2. Maintains an **aggro radius** of 4: if a human comes within 4 BFS hops, the bot gives chase.
3. Always takes an **immediate kill** if a human is one move away.
4. Before moving, checks that the destination is **safe** — it will not step onto a cell where a human could capture it next turn unless that human is itself covered by another bot.
5. Never steps onto the exit cell (it has no interest in escaping).
6. Never collides with fellow bots (it treats other bot cells as blocked during BFS).

This makes rational bots methodical and hard to lure out of position.

### Greedy

The greedy bot ignores the exit and guard strategy entirely. It:

1. Always takes an **immediate kill** if a human is adjacent.
2. Otherwise, BFS-searches from its current position and **chases the nearest human** by shortest knight-move distance, stepping one hop per turn.
3. Still treats other bots as obstacles (no bot-on-bot collisions).
4. Falls back to any free neighbouring cell if no humans are reachable.

Greedy bots are more aggressive and predictable — a clever agent can exploit their single-minded pursuit to manoeuvre around them — but in open mazes they converge fast and are hard to outrun.

---

## Timeline and replay

Every move is recorded. After the game ends (or while it's paused) you can:

- Scrub the **history slider** to step through every turn.
- Use **◀ / ▶** to step one frame at a time.
- Press **↺ Replay** to restart the exact same map and initial positions from the beginning.

---

## Ideas for stronger agents

- **Threat-aware routing** — avoid cells reachable by bots in one move, not just cells currently occupied.
- **Bot-mode detection** — use `dist_from_escape` to infer whether nearby bots are guarding (rational) or converging (greedy), and plan accordingly.
- **Opponent modelling** — predict where rivals will be next turn and race or block them.
- **Weighted A\*** — weight the heuristic more aggressively to trade optimality for speed of convergence, useful when the timeout is tight.
- **Multi-target BFS** — expand from the exit backwards, not from your position forwards, to get true shortest distances to all cells in one pass.
- **Decoy manoeuvres** — deliberately approach a bot to pull it off its guard post, then pivot toward the exit.

---

## Technical notes

- The entire Python runtime runs in the browser via **Pyodide** (WebAssembly CPython). No code leaves your machine.
- `print()` calls inside your agent are captured and shown in the log panel (purple entries), which is useful for debugging.
- Standard library modules available to Pyodide (e.g. `heapq`, `collections`, `math`, `random`) are all importable. Third-party packages are not pre-loaded.
- The engine validates every returned node id; an illegal move results in a wait, not a crash.
- If the same complete board state (all alive, non-escaped agents at the same positions) is seen **3 times**, the game is declared a **stalemate** to prevent infinite loops.
