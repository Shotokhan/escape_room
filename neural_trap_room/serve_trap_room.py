"""
serve_trap_room.py — Local server for the Trap Room with trained neural agents.

Instead of asking the user to implement an agent in Python in the browser,
this server loads the trained models (arch1, arch2, arch3) and exposes them
via a simple HTTP API.  The browser sends the current pile state and receives
the agent's chosen move.  Inference runs server-side with PyTorch/numpy —
no Pyodide, no weight inlining required.

Usage
-----
    python serve_trap_room.py                    # port 8765, models/ dir
    python serve_trap_room.py --port 9000
    python serve_trap_room.py --models path/to/models

Then open  http://localhost:8765  in the browser.

Endpoints
---------
    GET  /              → serves the Trap Room HTML page
    GET  /models        → JSON list of available model names
    POST /move          → { "model": "arch3", "piles": [3,4,5] }
                          → { "pile": 2, "amount": 3,
                              "logits": [...], "model": "arch3" }
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# ── Make sure the ML modules are importable ────────────────────────────────────
# serve_trap_room.py lives in the same directory as the arch*.py files.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── Agent registry ─────────────────────────────────────────────────────────────

AGENTS = {}        # name → agent instance (loaded lazily)
MODEL_DIR = "models"

ARCH_METADATA = {
    "arch1_onehot": {
        "label":       "Architecture 1 — One-hot (78-dim)",
        "description": "Lookup-table style. Encodes each pile as a 13-dim one-hot vector. "
                       "Learns state→move mapping. Generalises poorly to unseen pile combinations.",
        "file":        "arch1_onehot.pt",
        "cls_module":  "arch1_onehot",
        "cls_name":    "OneHotAgent",
    },
    "arch2_scalar": {
        "label":       "Architecture 2 — Scalar + mod-4 (12-dim)",
        "description": "Compact encoding: raw pile counts + Grundy mod-4 features. "
                       "Layer 1 learns Grundy periodicity; generalises across pile ranges.",
        "file":        "arch2_scalar.pt",
        "cls_module":  "arch2_scalar",
        "cls_name":    "ScalarAgent",
    },
    "arch3_transformer": {
        "label":       "Architecture 3 — Causal Transformer",
        "description": "Decoder-only transformer. Pile values are tokens in a 7-token prompt "
                       "[p0..p5 SEP]. Attention computes joint pile interactions. Best win rate.",
        "file":        "arch3_transformer.pt",
        "cls_module":  "arch3_transformer",
        "cls_name":    "TransformerAgent",
    },
    "minimax": {
        "label":       "Minimax (exact, no model)",
        "description": "Reference solver. Computes the optimal move by exhaustive game-tree search "
                       "using Sprague-Grundy theory. Always plays optimally. Not a neural network.",
        "file":        None,   # no .pt file — always available
        "cls_module":  None,
        "cls_name":    None,
    },
}


def _available_models():
    """Return list of model names whose .pt file exists (plus minimax always)."""
    available = ["minimax"]
    for name, meta in ARCH_METADATA.items():
        if meta["file"] is None:
            continue
        path = os.path.join(MODEL_DIR, meta["file"])
        if os.path.exists(path):
            available.append(name)
    return available


def _load_agent(name: str):
    """Lazily load and cache an agent instance."""
    if name in AGENTS:
        return AGENTS[name]

    if name == "minimax":
        from game import optimal_moves, pad_piles, decode_move
        class MinimaxAgent:
            def next_move(self, piles):
                import random
                p = pad_piles(list(piles))
                opts = optimal_moves(p)
                move_id = random.choice(opts)
                return decode_move(move_id)
        AGENTS[name] = MinimaxAgent()
        return AGENTS[name]

    meta = ARCH_METADATA.get(name)
    if meta is None:
        raise ValueError(f"Unknown model: {name!r}")

    mod = __import__(meta["cls_module"])
    cls = getattr(mod, meta["cls_name"])
    AGENTS[name] = cls(model_dir=MODEL_DIR)
    print(f"  [server] Loaded {name} from {MODEL_DIR}/", flush=True)
    return AGENTS[name]


def _compute_move(name: str, piles: list) -> dict:
    """Run inference for the given model and pile state."""
    agent   = _load_agent(name)
    pile_i, amount = agent.next_move(list(piles))
    return {"pile": int(pile_i), "amount": int(amount), "model": name}


# ── HTML page ──────────────────────────────────────────────────────────────────

def _build_html(available_models: list) -> str:
    """Return the full Trap Room HTML with server-side model selector."""

    # Build the model selector options
    options_html = ""
    for name in available_models:
        meta  = ARCH_METADATA[name]
        label = meta["label"]
        desc  = meta["description"]
        options_html += (
            f'<option value="{name}" title="{desc}">{label}</option>\n'
        )

    # Build ARCH_META JSON before the f-string to avoid {{ }} conflict
    arch_meta_js = json.dumps(
        {name: ARCH_METADATA[name] for name in available_models}, indent=2
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>The Trap Room — Neural Agent</title>
<style>
:root{{
  --bg:#0e0e12;--bg2:#16161c;--bg3:#1e1e28;--border:#2a2a3a;
  --text:#cdd6f4;--muted:#6c7086;--accent:#89b4fa;--green:#a6e3a1;
  --red:#f38ba8;--yellow:#f9e2af;--orange:#fab387;--purple:#cba6f7;
  --font-mono:'Fira Code','Cascadia Code','Consolas',monospace;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden}}
body{{background:var(--bg);color:var(--text);font-family:var(--font-mono);font-size:14px;display:flex;flex-direction:column}}
header{{background:var(--bg2);border-bottom:1px solid var(--border);padding:7px 16px;display:flex;align-items:center;gap:14px;flex-shrink:0;flex-wrap:wrap}}
header h1{{font-size:14px;font-weight:600;color:var(--orange);letter-spacing:.06em}}
header span{{font-size:11px;color:var(--muted)}}
#server-status{{font-size:11px;padding:2px 9px;border-radius:4px;border:1px solid var(--yellow);color:var(--yellow);margin-left:auto;white-space:nowrap}}
#server-status.ready{{color:var(--green);border-color:var(--green)}}
#server-status.err{{color:var(--red);border-color:var(--red)}}
.main{{display:flex;flex:1;min-height:0}}

/* ── Left pane — model selector ── */
.pane-left{{width:320px;flex-shrink:0;display:flex;flex-direction:column;border-right:1px solid var(--border);background:var(--bg2)}}
.section{{padding:14px 16px;border-bottom:1px solid var(--border)}}
.section-title{{font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}}
.cfg{{display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--muted)}}
.cfg label{{color:var(--text);font-size:11px}}
.cfg select,.cfg input[type=range]{{background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:var(--font-mono);font-size:11px;border-radius:4px;padding:4px 8px;width:100%;cursor:pointer;accent-color:var(--orange)}}
.cfg select:focus,.cfg select:hover{{border-color:var(--orange);outline:none}}
.model-desc{{font-size:11px;color:var(--muted);line-height:1.6;margin-top:8px;padding:8px;background:var(--bg);border-radius:5px;border:1px solid var(--border);min-height:60px}}
.range-row{{display:flex;align-items:center;gap:8px}}
.range-row span{{color:var(--text);min-width:20px;text-align:right;font-size:11px}}
button{{background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:5px 13px;cursor:pointer;font-family:var(--font-mono);font-size:11px;transition:background .12s;white-space:nowrap}}
button:hover:not(:disabled){{background:#2a2a3e}}
button:disabled{{opacity:.35;cursor:default;pointer-events:none}}
button.primary{{background:#301e10;border-color:var(--orange);color:var(--orange)}}
button.primary:hover:not(:disabled){{background:#3a2614}}
button.success{{background:#0e2e18;border-color:var(--green);color:var(--green)}}
button.danger{{background:#2e1420;border-color:var(--red);color:var(--red)}}
.btn-row{{padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap}}

/* ── Right pane ── */
.pane-right{{display:flex;flex-direction:column;min-width:260px;overflow:hidden;flex:1}}
.game-area{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:14px 16px;min-height:0;overflow-y:auto;gap:0}}
.narrative{{font-size:12px;color:var(--muted);text-align:center;line-height:1.7;max-width:520px;margin-bottom:14px}}
.narrative b{{color:var(--orange)}}
.status-bar{{display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap;margin-bottom:14px}}
.status-badge{{font-size:12px;padding:3px 12px;border-radius:5px;border:1px solid var(--border)}}
.status-badge.turn-human{{border-color:var(--accent);color:var(--accent);background:#0e1e30}}
.status-badge.turn-agent{{border-color:var(--orange);color:var(--orange);background:#1e0e04}}
.status-badge.info{{color:var(--muted)}}
#outcome-msg{{font-size:16px;font-weight:700;letter-spacing:.04em;min-height:24px;text-align:center;margin-bottom:8px}}
.piles-wrap{{display:flex;gap:24px;align-items:flex-end;justify-content:center;flex-wrap:wrap;margin-bottom:18px;min-height:160px}}
.pile{{display:flex;flex-direction:column;align-items:center;gap:6px;cursor:pointer;border-radius:8px;padding:8px 10px;border:2px solid transparent;transition:border-color .15s,background .15s;user-select:none}}
.pile.selectable:hover{{border-color:var(--accent);background:#0e1e30}}
.pile.selected{{border-color:var(--accent);background:#0e1e30}}
.pile.selected .pile-label{{color:var(--accent)}}
.coins-col{{display:flex;flex-direction:column;gap:3px;align-items:center;min-height:120px;justify-content:flex-end}}
.coin{{width:28px;height:14px;border-radius:50%;background:linear-gradient(135deg,#f9e2af 0%,#e0a020 60%,#f9e2af 100%);border:1px solid #b07010;box-shadow:0 1px 3px rgba(0,0,0,.4);transition:opacity .2s}}
.coin.ghost{{opacity:.18}}
.pile-label{{font-size:13px;font-weight:600;color:var(--text)}}
.pile-count{{font-size:11px;color:var(--muted)}}
.remove-ctrl{{display:flex;flex-direction:column;align-items:center;gap:8px;margin-bottom:14px;min-height:56px}}
.remove-row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center}}
.remove-row label{{font-size:11px;color:var(--muted)}}
#inp-remove{{width:60px;background:var(--bg3);border:1px solid var(--border);border-radius:5px;color:var(--text);font-family:var(--font-mono);font-size:13px;padding:3px 8px;text-align:center}}
#inp-remove:focus{{outline:none;border-color:var(--accent)}}
button.take-btn{{background:#0e1e30;border-color:var(--accent);color:var(--accent);font-size:12px;padding:5px 18px}}
button.take-btn:hover:not(:disabled){{background:#14284a}}
.game-ctrl{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center;padding:8px 10px;background:var(--bg2);border-top:1px solid var(--border);flex-shrink:0}}
.log-panel{{background:#0a0a0f;border-top:1px solid var(--border);height:90px;overflow-y:auto;padding:5px 10px;font-size:11px;line-height:1.75;flex-shrink:0}}
.log-panel .err{{color:var(--red)}}
.log-panel .ok{{color:var(--green)}}
.log-panel .info{{color:var(--muted)}}
.log-panel .warn{{color:var(--yellow)}}
.log-panel .agent{{color:var(--orange)}}
.log-panel .human{{color:var(--accent)}}
</style>
</head>
<body>

<header>
  <h1>☠ THE TRAP ROOM</h1>
  <span>neural agent challenge — whoever takes the <b style="color:var(--orange)">last coin</b> triggers the trap and <b style="color:var(--red)">dies</b></span>
  <div id="server-status">connecting…</div>
</header>

<div class="main">
  <!-- ══ LEFT: model selector ══ -->
  <div class="pane-left">
    <div class="section">
      <div class="section-title">Neural Agent</div>
      <div class="cfg">
        <label>model</label>
        <select id="sel-model" onchange="onModelChange()">
          {options_html}
        </select>
        <div class="model-desc" id="model-desc">Select a model to see its description.</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Game Setup</div>
      <div class="cfg">
        <label>piles</label>
        <div class="range-row">
          <input type="range" id="cfg-piles" min="1" max="6" value="3">
          <span id="v-piles">3</span>
        </div>
        <label style="margin-top:6px">max coins / pile</label>
        <div class="range-row">
          <input type="range" id="cfg-coins" min="2" max="12" value="5">
          <span id="v-coins">5</span>
        </div>
        <label style="margin-top:6px">who goes first</label>
        <select id="cfg-first">
          <option value="human">Human</option>
          <option value="agent">Monster</option>
        </select>
      </div>
    </div>

    <div class="btn-row">
      <button class="primary" id="btn-new">New game</button>
      <button id="btn-replay">↺ Same piles</button>
    </div>
  </div>

  <!-- ══ RIGHT: game ══ -->
  <div class="pane-right">
    <div class="game-area">
      <div class="narrative">
        You and the <b>Monster</b> stand over piles of cursed gold.<br>
        On your turn: pick a pile, take <b>1, 2, or 3 coins</b> from it.<br>
        <b>Whoever takes the last coin triggers the trap — and dies.</b><br>
        The Monster is powered by a <b>neural network</b> trained on Nim games.
      </div>
      <div class="status-bar">
        <div class="status-badge info" id="badge-turn">—</div>
        <div class="status-badge info" id="badge-state">coins left: —</div>
      </div>
      <div id="outcome-msg"></div>
      <div class="piles-wrap" id="piles-wrap"></div>
      <div class="remove-ctrl" id="remove-ctrl">
        <div class="remove-row">
          <label>remove</label>
          <input type="number" id="inp-remove" min="1" value="1">
          <label>coin(s) from selected pile</label>
        </div>
        <button class="take-btn" id="btn-take" disabled>Take coins</button>
      </div>
    </div>

    <div class="game-ctrl">
      <button class="primary" id="btn-new2">New game</button>
      <button id="btn-replay2">↺ Same piles</button>
    </div>
    <div class="log-panel" id="log"></div>
  </div>
</div>

<script>
// ══════════════════════════════════════════════════
// METADATA (mirrored from server for descriptions)
// ══════════════════════════════════════════════════
const ARCH_META = {arch_meta_js};

// ══════════════════════════════════════════════════
// LOGGING
// ══════════════════════════════════════════════════
const logEl = document.getElementById('log');
function lg(msg, cls='info') {{
  const d = document.createElement('div');
  d.className = cls; d.textContent = msg;
  logEl.prepend(d);
  if (logEl.children.length > 300) logEl.lastChild.remove();
}}

// ══════════════════════════════════════════════════
// SERVER HEALTH CHECK
// ══════════════════════════════════════════════════
async function checkServer() {{
  try {{
    const r = await fetch('/models');
    const models = await r.json();
    document.getElementById('server-status').textContent = 'server ready';
    document.getElementById('server-status').className = 'server-status ready';
    lg('Server ready — ' + models.length + ' model(s) available', 'ok');
    onModelChange();
  }} catch(e) {{
    document.getElementById('server-status').textContent = 'server error';
    document.getElementById('server-status').className = 'server-status err';
    lg('Cannot reach server: ' + e.message, 'err');
  }}
}}

// ══════════════════════════════════════════════════
// MODEL SELECTOR
// ══════════════════════════════════════════════════
function onModelChange() {{
  const name = document.getElementById('sel-model').value;
  const meta = ARCH_META[name];
  document.getElementById('model-desc').textContent =
    meta ? meta.description : '';
}}

function selectedModel() {{
  return document.getElementById('sel-model').value;
}}

// ══════════════════════════════════════════════════
// GAME STATE
// ══════════════════════════════════════════════════
let G = null, savedPiles = null;

function getCfg() {{
  return {{
    nPiles:   parseInt(document.getElementById('cfg-piles').value),
    maxCoins: parseInt(document.getElementById('cfg-coins').value),
    first:    document.getElementById('cfg-first').value,
  }};
}}
function totalCoins(piles) {{ return piles.reduce((a,b) => a+b, 0); }}

function newGame(pilesOverride = null) {{
  const cfg = getCfg();
  let piles;
  if (pilesOverride) {{
    piles = [...pilesOverride];
  }} else {{
    piles = [];
    for (let i = 0; i < cfg.nPiles; i++)
      piles.push(1 + Math.floor(Math.random() * cfg.maxCoins));
    savedPiles = [...piles];
  }}
  G = {{ piles, turn: cfg.first, over: false, winner: null, selectedPile: null }};
  render();
  lg('New game — piles: [' + piles.join(', ') + '] — ' +
     (cfg.first === 'human' ? 'You go first' : 'Monster goes first'), 'ok');
  if (G.turn === 'agent') setTimeout(agentTurn, 400);
}}

function applyMove(pileIdx, amount) {{
  if (G.over) return false;
  if (pileIdx < 0 || pileIdx >= G.piles.length) return false;
  if (amount < 1 || amount > 3 || amount > G.piles[pileIdx]) return false;
  G.piles[pileIdx] -= amount;
  if (totalCoins(G.piles) === 0) {{
    G.over   = true;
    G.winner = G.turn === 'human' ? 'agent' : 'human';  // mover loses
  }} else {{
    G.turn = G.turn === 'human' ? 'agent' : 'human';
  }}
  return true;
}}

// ══════════════════════════════════════════════════
// SERVER MOVE REQUEST
// ══════════════════════════════════════════════════
async function agentTurn() {{
  if (!G || G.over || G.turn !== 'agent') return;
  render();  // show "thinking" badge immediately
  try {{
    const resp = await fetch('/move', {{
      method:  'POST',
      headers: {{'Content-Type': 'application/json'}},
      body:    JSON.stringify({{
        model: selectedModel(),
        piles: G.piles,
      }}),
    }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const ok = applyMove(data.pile, data.amount);
    if (!ok) {{
      lg('Agent returned invalid move (' + data.pile + ',' + data.amount + ') — passing', 'err');
      G.turn = 'human';
    }} else {{
      lg('Monster takes ' + data.amount + ' from pile ' + (data.pile+1) +
         ' → [' + G.piles.join(', ') + ']', 'agent');
    }}
  }} catch(e) {{
    lg('Server error: ' + e.message, 'err');
    G.turn = 'human';
  }}
  render();
}}

function humanTake() {{
  if (!G || G.over || G.turn !== 'human') return;
  if (G.selectedPile === null) {{ lg('Select a pile first', 'warn'); return; }}
  const amt = parseInt(document.getElementById('inp-remove').value);
  if (isNaN(amt) || amt < 1 || amt > 3) {{ lg('You can only take 1, 2, or 3 coins', 'warn'); return; }}
  if (amt > G.piles[G.selectedPile]) {{ lg('Not enough coins in that pile', 'warn'); return; }}
  const pi = G.selectedPile;
  applyMove(pi, amt);
  lg('You take ' + amt + ' from pile ' + (pi+1) + ' → [' + G.piles.join(', ') + ']', 'human');
  G.selectedPile = null;
  render();
  if (!G.over && G.turn === 'agent') setTimeout(agentTurn, 400);
}}

// ══════════════════════════════════════════════════
// RENDERING  (identical to original trap room)
// ══════════════════════════════════════════════════
function render() {{
  if (!G) return;
  const omsg = document.getElementById('outcome-msg');
  if (G.over) {{
    if (G.winner === 'human') {{
      omsg.textContent = '★  YOU ESCAPED!'; omsg.style.color = 'var(--green)';
    }} else {{
      omsg.textContent = '☠  THE TRAP SPRINGS — YOU LOSE'; omsg.style.color = 'var(--red)';
    }}
  }} else {{
    omsg.textContent = ''; omsg.style.color = '';
  }}

  const bt = document.getElementById('badge-turn');
  const model = selectedModel();
  if (G.over) {{
    bt.textContent = 'game over'; bt.className = 'status-badge info';
  }} else if (G.turn === 'human') {{
    bt.textContent = '⚔ Your turn'; bt.className = 'status-badge turn-human';
  }} else {{
    bt.textContent = '👾 ' + (ARCH_META[model]?.label.split('—')[0].trim() || model) + ' thinking…';
    bt.className = 'status-badge turn-agent';
  }}
  document.getElementById('badge-state').textContent = 'coins left: ' + totalCoins(G.piles);

  const wrap    = document.getElementById('piles-wrap');
  const maxCoins = getCfg().maxCoins;
  wrap.innerHTML = '';
  const isHumanTurn = !G.over && G.turn === 'human';
  G.piles.forEach((n, i) => {{
    const pile = document.createElement('div');
    pile.className = 'pile' + (isHumanTurn ? ' selectable' : '') +
                     (G.selectedPile === i ? ' selected' : '');
    const col = document.createElement('div'); col.className = 'coins-col';
    for (let k = 0; k < maxCoins; k++) {{
      const coin = document.createElement('div');
      coin.className = 'coin' + (k >= n ? ' ghost' : '');
      col.prepend(coin);
    }}
    const lbl = document.createElement('div'); lbl.className = 'pile-label';
    lbl.textContent = 'Pile ' + (i+1);
    const cnt = document.createElement('div'); cnt.className = 'pile-count';
    cnt.textContent = n + ' coin' + (n !== 1 ? 's' : '');
    pile.appendChild(col); pile.appendChild(lbl); pile.appendChild(cnt);
    if (isHumanTurn) {{
      pile.onclick = () => {{
        if (n === 0) return;
        G.selectedPile = i;
        const inp = document.getElementById('inp-remove');
        inp.max = Math.min(3, n);
        if (parseInt(inp.value) < 1) inp.value = 1;
        if (parseInt(inp.value) > Math.min(3, n)) inp.value = Math.min(3, n);
        render();
      }};
    }}
    wrap.appendChild(pile);
  }});

  const takeBtn = document.getElementById('btn-take');
  const rc      = document.getElementById('remove-ctrl');
  if (isHumanTurn) {{
    rc.style.visibility = 'visible';
    takeBtn.disabled = G.selectedPile === null;
    if (G.selectedPile !== null) {{
      document.getElementById('inp-remove').max = Math.min(3, G.piles[G.selectedPile]);
    }}
  }} else {{
    rc.style.visibility = 'hidden';
    takeBtn.disabled = true;
  }}
}}

// ══════════════════════════════════════════════════
// CONTROLS
// ══════════════════════════════════════════════════
document.getElementById('btn-take').onclick    = humanTake;
document.getElementById('btn-new').onclick     = () => newGame();
document.getElementById('btn-new2').onclick    = () => newGame();
document.getElementById('btn-replay').onclick  = () => {{ if (savedPiles) newGame(savedPiles); else newGame(); }};
document.getElementById('btn-replay2').onclick = () => {{ if (savedPiles) newGame(savedPiles); else newGame(); }};

[['cfg-piles','v-piles'],['cfg-coins','v-coins']].forEach(([id,vid]) => {{
  const sl = document.getElementById(id), sp = document.getElementById(vid);
  sp.textContent = sl.value; sl.oninput = () => sp.textContent = sl.value;
}});

// ══════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════
checkServer().then(() => {{
  newGame();
  lg('Welcome — choose a neural agent on the left and play against it.', 'ok');
}});
</script>
</body>
</html>"""


# ── HTTP request handler ───────────────────────────────────────────────────────

class TrapRoomHandler(BaseHTTPRequestHandler):

    _html_cache: str = None   # built once, reused

    def log_message(self, fmt, *args):
        # Suppress default per-request logging; use our own below
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS preflight (for browsers with strict origin checks)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            if TrapRoomHandler._html_cache is None:
                available = _available_models()
                TrapRoomHandler._html_cache = _build_html(available)
                print(f"  [server] Page built — models: {available}", flush=True)
            self._send_html(TrapRoomHandler._html_cache)

        elif path == "/models":
            available = _available_models()
            result = [
                {"name": n, **{k: v for k, v in ARCH_METADATA[n].items()
                               if k not in ("cls_module", "cls_name")}}
                for n in available
            ]
            self._send_json(result)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/move":
            length  = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            model   = payload.get("model", "minimax")
            piles   = payload.get("piles", [])

            if not piles or not isinstance(piles, list):
                self._send_json({"error": "invalid piles"}, 400)
                return

            try:
                result = _compute_move(model, piles)
                print(f"  [server] {model} | piles={piles} → "
                      f"pile {result['pile']} take {result['amount']}", flush=True)
                self._send_json(result)
            except Exception as e:
                print(f"  [server] ERROR: {e}", flush=True)
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Serve the Trap Room with neural network agents"
    )
    parser.add_argument("--port",   type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--models", type=str, default="models",
                        help="Directory containing *.pt model files (default: models/)")
    args = parser.parse_args()

    global MODEL_DIR
    MODEL_DIR = args.models

    available = _available_models()
    if not available:
        print(f"WARNING: no model files found in {MODEL_DIR}/  "
              f"(only minimax will be available)", flush=True)

    print(f"\n  The Trap Room — Neural Agent Server", flush=True)
    print(f"  ────────────────────────────────────", flush=True)
    print(f"  Models dir : {os.path.abspath(MODEL_DIR)}", flush=True)
    print(f"  Available  : {', '.join(available)}", flush=True)
    print(f"  URL        : http://localhost:{args.port}", flush=True)
    print(f"  Stop       : Ctrl-C\n", flush=True)

    server = HTTPServer(("localhost", args.port), TrapRoomHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.", flush=True)


if __name__ == "__main__":
    main()
