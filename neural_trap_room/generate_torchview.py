"""
generate_torchview.py — Render torchview computation graphs for all three
                         Trap Room neural architectures and save to plots/.

Usage
-----
    python generate_torchview.py
    python generate_torchview.py --models models/ --plots plots/ --format png

Output
------
    plots/torchview_arch1_onehot.png
    plots/torchview_arch2_scalar.png
    plots/torchview_arch3_transformer.png

Each image shows the full computational graph as torchview traces it:
every tensor operation, shape annotation, and module boundary is visible.
This is complementary to the interactive architecture_explorer.html:
  - torchview  = ground truth, every op, dense, technical
  - explorer   = curated, annotated, pedagogically guided

Dependencies
------------
    pip install torchview graphviz
    (graphviz binary also required: apt install graphviz / brew install graphviz)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Make sure the arch modules are importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchview import draw_graph


# ── Architecture imports ───────────────────────────────────────────────────────

def load_arch1(model_dir: str):
    """One-hot feedforward  78 → 128 → 64 → 18."""
    from arch1_onehot import OneHotNet
    model = OneHotNet()
    path  = os.path.join(model_dir, "arch1_onehot.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  Loaded weights from {path}")
    else:
        print(f"  No checkpoint found at {path} — using random weights")
    model.eval()
    return model, (1, 78)


def load_arch2(model_dir: str):
    """Scalar + mod-4  12 → 128 → 64 → 18."""
    from arch2_scalar import ScalarNet
    model = ScalarNet()
    path  = os.path.join(model_dir, "arch2_scalar.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  Loaded weights from {path}")
    else:
        print(f"  No checkpoint found at {path} — using random weights")
    model.eval()
    return model, (1, 12)


def load_arch3(model_dir: str):
    """Causal transformer  7 tokens → 32 logits."""
    from arch3_transformer import NimTransformer, D_MODEL
    checkpoint = torch.load(
        os.path.join(model_dir, "arch3_transformer.pt"), map_location="cpu"
    )
    d = checkpoint["state_dict"]["tok_emb.weight"].shape[1]
    model = NimTransformer(max_len=checkpoint["max_len"], d_model=d)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"  Loaded weights (d_model={d}, max_len={checkpoint['max_len']})")
    model.eval()
    # Input: integer token IDs, shape (batch, seq_len)
    # torchview needs a concrete input tensor
    dummy = torch.zeros(1, 7, dtype=torch.long)
    return model, dummy


# ── Graph rendering ────────────────────────────────────────────────────────────

def render(name: str, model: nn.Module, input_data, plots_dir: str,
           fmt: str = "png", compact: bool = False) -> str:
    """Render a torchview graph and save it to plots_dir.

    compact=False (default): full graph — every tensor op, all depths, most informative.
    compact=True:            module-level overview — hides intermediate tensors, limits
                             depth to 3, rolls repeated blocks. Much smaller image.
    Returns the output file path.
    """
    print(f"  Tracing computation graph…")

    suffix   = "_compact" if compact else ""
    kwargs = dict(
        graph_name         = name,
        expand_nested      = not compact,  # compact: top-level modules only
        show_shapes        = True,         # always annotate tensor shapes
        hide_inner_tensors = compact,      # compact: hide tensors between modules
        depth              = 3 if compact else 6,
        roll               = True,         # roll repeated modules (e.g. both transformer blocks → one)
        directory          = plots_dir,
        filename           = f"torchview_{name}{suffix}",
    )

    if isinstance(input_data, torch.Tensor):
        graph = draw_graph(model, input_data=input_data, **kwargs)
    else:
        graph = draw_graph(model, input_size=input_data, **kwargs)

    vg = graph.visual_graph

    # Set rendering options for legibility
    # `ortho` splines crash dot on large graphs — use `spline` as fallback
    vg.attr(
        rankdir  = "TB",
        bgcolor  = "#0d1117",
        fontname = "JetBrains Mono",
        fontsize = "11",
        pad      = "0.5",
        splines  = "spline",
        nodesep  = "0.4",
        ranksep  = "0.6",
    )
    vg.attr("node",
        fontname  = "JetBrains Mono",
        fontsize  = "10",
        style     = "filled",
        fillcolor = "#161b22",
        color     = "#30363d",
        fontcolor = "#c9d1d9",
        margin    = "0.15,0.08",
    )
    vg.attr("edge",
        color     = "#58a6ff",
        fontcolor = "#6e7681",
        fontname  = "JetBrains Mono",
        fontsize  = "9",
        arrowsize = "0.7",
    )

    out_path = vg.render(format=fmt, cleanup=True)
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate torchview computation graphs for all three Trap Room models"
    )
    parser.add_argument("--models", default="models",
                        help="Directory containing *.pt checkpoint files")
    parser.add_argument("--plots",  default="plots",
                        help="Output directory for rendered images")
    parser.add_argument("--format", default="png",
                        choices=["png", "svg", "pdf"],
                        help="Output image format (default: png)")
    parser.add_argument("--arch",    default=None,
                        choices=["1", "2", "3"],
                        help="Render only this architecture (default: all)")
    parser.add_argument("--compact", action="store_true",
                        help="Hide intermediate tensors and limit depth — "
                             "produces a cleaner module-level view, especially "
                             "useful for arch3 whose full graph is very tall")
    args = parser.parse_args()

    os.makedirs(args.plots, exist_ok=True)

    archs = {
        "1": ("arch1_onehot",      load_arch1),
        "2": ("arch2_scalar",      load_arch2),
        "3": ("arch3_transformer", load_arch3),
    }

    targets = ([args.arch] if args.arch else ["1", "2", "3"])

    for key in targets:
        name, loader = archs[key]
        print(f"\nArchitecture {key} — {name}")
        print("-" * 50)
        model, input_data = loader(args.models)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        render(name, model, input_data, args.plots, fmt=args.format, compact=args.compact)

    print(f"\nDone. Graphs written to: {os.path.abspath(args.plots)}/")


if __name__ == "__main__":
    main()
