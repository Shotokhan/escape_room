# Escape Room v2 — Algorithmic Challenge

A self-contained, browser-based coding challenge focused on **graph algorithms**.  
No server required — open `index.html` and start coding.

---

## Concept

The original *Escape Room* challenge was a web-security exercise: automate HTTP requests fast enough to navigate a maze before the session expires.

**v2 removes the web layer entirely.** The challenge is now purely algorithmic: implement an agent in Python that navigates a procedurally generated maze from a start cell to the exit (`E`), while monsters move randomly around the map. Your code runs live in the browser via [Pyodide](https://pyodide.org) (CPython compiled to WebAssembly) — no install, no backend.

The intended learning path is:

```
Dijkstra  →  A*  →  Weighted A*  →  monster-aware heuristics
```

---

## Files

```
v2_coding/
└── index.html      # the entire challenge — editor, runtime, maze, game loop
```

That's it. One file, no dependencies to install, no build step.

---

## How to play

1. Open `index.html` in a browser (Chrome or Firefox recommended).
2. Wait for the **Python ready** badge to appear in the top-right corner (Pyodide loads once, ~10 MB).
3. Write your agent in the left-hand editor.
4. Click **▶ Load agent** to compile and validate your code.
5. Click **▶ Run** to start the simulation.
6. Use **↺ Replay map** to re-run a different algorithm on the exact same map.
7. Use **New map** to generate a fresh maze.

---

## Agent interface

You must implement a Python class called `Agent` with exactly two methods:

```python
class Agent:
    def __init__(self, graph, escape_node):
        """
        Called once when a new map is generated.

        graph       : list of dicts, indexed by node id
                      graph[id] = {
                        'id'   : int,
                        'row'  : int,          # row in the grid
                        'col'  : int,          # column in the grid
                        'edges': [             # adjacent walkable cells
                          {'to': int, 'weight': int},
                          ...
                        ]
                      }
        escape_node : int — node id of the exit cell (marked 'E' on the map)

        Walls are not nodes — they simply do not exist in the graph.
        All edge weights are 1 (uniform-cost grid).
        """
        pass

    def next_move(self, player_node, monsters):
        """
        Called every game tick.

        player_node : int       — current node id of the player
        monsters    : list[int] — node ids of all monsters, in fixed order

        Return the node id the player should move to next tick.
        The returned node MUST be a direct neighbour of player_node
        (present in player_node's edge list), or player_node itself to wait.
        """
        return player_node  # WAIT
```

`print()` works normally — output appears in the log panel at the bottom right.

---

## Map & game rules

| Symbol | Meaning |
|--------|---------|
| `P` | Player (you) |
| `M` | Monster |
| `E` | Escape (goal) |
| dark tile | Wall (not a graph node) |
| floor tile | Free cell (graph node) |

- The player moves **first** each tick, then monsters move **randomly**.
- The player **dies** if:
  - they move onto a cell occupied by a monster, or
  - a monster moves onto the player's cell.
- Monsters **cannot** step onto the escape cell.
- Reaching the escape cell wins the round.

---

## Configuration sliders

| Slider | Effect |
|--------|--------|
| **rows / cols** | Map dimensions (odd values only, 5–35 / 5–51) |
| **complexity** | `0` = open arena, `1–3` = sparse random walls, `4–10` = recursive-backtracker maze (10 = densest) |
| **monsters** | Number of monsters (1–24) |
| **speed** | Simulation speed (ticks per second) |

---

## Starter implementation — Dijkstra

The **Reset code (Dijkstra)** button loads a complete working agent that:

- Precomputes nothing in the constructor (all work done per tick).
- Runs Dijkstra from `player_node` to `escape_node` on every call to `next_move`.
- Uses a binary min-heap (`heapq`) with `(priority, node_id)` tuples.
- Prints the number of nodes visited each tick for comparison.

This is your baseline. It reaches the exit on any map (ignoring monsters), and its node-visit count is the number to beat.

---

## Extending to A\*

Change the heap priority from the true path cost alone to `g + h`:

```python
# Dijkstra
heapq.heappush(heap, (dist[u] + w, v))

# A*
heapq.heappush(heap, (dist[u] + w + self.h(v), v))
```

**Important:** store only the true path cost `g` in `dist`, never `g + h`:

```python
g_new = dist[u] + w          # true cost — store this
f_new = g_new + self.h(v)    # estimated total — heap priority only
if g_new < dist[v]:
    dist[v] = g_new           # ← g only
    heapq.heappush(heap, (f_new, v))
```

### Manhattan distance heuristic

```python
def h(self, node):
    return (abs(self.graph[node]['row'] - self.esc_row)
          + abs(self.graph[node]['col'] - self.esc_col))
```

Admissible (never overestimates) on any grid. Exact on `complexity=0` (open arena) — A\* visits almost no extra nodes. Weaker on dense mazes, where walls force detours much longer than the Manhattan distance predicts.

### Weighted A\*

```python
f_new = g_new + epsilon * self.h(v)   # epsilon > 1
```

| `epsilon` | Behaviour |
|-----------|-----------|
| `0` | Pure Dijkstra (heuristic ignored) |
| `1` | Standard A\* (optimal, admissible) |
| `1.5–3` | Fewer nodes visited, path up to `epsilon`× longer than optimal |
| very large | Greedy best-first search |

Use **↺ Replay map** to run the same map with different `epsilon` values and compare node counts directly in the log.

---

## Suggested exercises

1. **Verify the baseline.** Run Dijkstra on `complexity=0`. How many nodes does it visit? Now switch to A\* with Manhattan distance. How does the count change?

2. **Stress the heuristic.** Repeat with `complexity=10`. Why does the gap between Dijkstra and A\* shrink?

3. **Tune weighted A\*.** Find the smallest `epsilon` that halves the node count on a dense maze without making the agent visibly take a longer path.

4. **Add monster awareness.** The `monsters` list in `next_move` gives you their current positions. Design a penalty term that steers the player away from cells a monster is likely to reach next tick. How does this affect node counts and survival rate?

5. **Precompute in the constructor.** The distance from the escape cell to every other node never changes within a map. Compute it once in `__init__` with a reverse BFS/Dijkstra and reuse it each tick instead of re-running from the player.

---

## Technical notes

### Python runtime
Pyodide 0.26 (CPython 3.11 in WASM). The full standard library is available — `heapq`, `math`, `collections`, `itertools`, etc. Packages requiring C extensions that are not bundled with Pyodide (e.g. NumPy, NetworkX) are not available without an explicit `await pyodide.loadPackage(...)` call, which is not wired up in this challenge by design.

### JS ↔ Python bridge
- **JS → Python scalars:** `pyodide.globals.set('name', value)` — plain integers and strings cross directly.
- **JS → Python lists:** `pyodide.toPy(array)` converts a JS array to a native Python list.
- **JS → Python objects:** The graph is serialised with `JSON.stringify` on the JS side and deserialised with `json.loads` in Python, giving the agent a fully native Python data structure.
- **Python → JS:** `pyodide.runPython('expr')` returns the result as a JS value; primitives convert automatically.
- **JS functions in Python:** `pyodide.globals.set('fn', jsFunc)` exposes a JS function as a Python global (used to redirect `print()` to the log panel).

### Maze generation
- **`complexity = 0`:** Completely open grid (border walls only).
- **`complexity 1–3`:** Random scattered internal walls (density 2–12 %).
- **`complexity 4–10`:** Recursive backtracker (depth-first carving). A perfect maze is generated first (one unique path between any two cells), then extra passages are punched through to reduce density. `complexity=10` is the pure perfect maze; `complexity=4` has ~35 % of internal walls removed. The exit is always placed on a border cell adjacent to a carved interior cell, guaranteeing reachability.

### History & replay
Every tick is snapshotted. The timeline scrubber at the bottom of the game panel lets you step through the entire run frame by frame. **↺ Replay map** restores the exact initial state (player position, monster positions, wall layout) without regenerating the maze, so you can compare algorithms under identical conditions.
