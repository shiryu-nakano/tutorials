"""
scripts/analyze_seed_consistency.py

Question: holding everything fixed except the random seed, do the loss
curves (L_data = L_ic + L_bc, and L_pde) follow the same trajectory?

This reuses the existing sweep output (results/sweep_results/) — no new
training needed. For each data-weight setting, it overlays the 5 seeds'
curves in one panel, so seed-to-seed variation is directly visible as
the spread between same-colored lines.

Outputs (to figures/):
  seed_consistency_grid.png   : one panel per weight; L_data (solid) and
                                L_pde (dashed) for every seed overlaid
  seed_consistency_dw<w>.png  : enlarged single-weight version for the
                                weight you care most about (default 10,
                                the original baseline setting)

Run:
    python scripts/analyze_seed_consistency.py
    python scripts/analyze_seed_consistency.py --focus-weight 1
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "sweep_results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_runs():
    runs = []
    for npz_path in sorted(glob.glob(str(RESULTS_DIR / "*.npz"))):
        run_id = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = RESULTS_DIR / f"{run_id}.json"
        if not json_path.exists():
            continue
        data = np.load(npz_path)
        with open(json_path) as f:
            cfg = json.load(f)
        runs.append(dict(run_id=run_id, data=data, cfg=cfg))
    return runs


def plot_panel(ax, runs_for_weight, seed_cmap, show_true_l2=False):
    """Overlay all seeds' L_data (solid) and L_pde (dashed) on one axis."""
    for r in sorted(runs_for_weight, key=lambda r: r["cfg"]["seed"]):
        d = r["data"]
        seed = r["cfg"]["seed"]
        color = seed_cmap(seed)
        L_data = d["L_ic"] + d["L_bc"]
        ax.plot(d["epoch"], L_data, color=color, linewidth=1.2,
                label=f"seed {seed}")
        ax.plot(d["epoch"], d["L_pde"], color=color, linewidth=1.2,
                linestyle="--")
        if show_true_l2:
            ax.plot(d["epoch"], d["true_L2"], color=color, linewidth=0.8,
                    linestyle=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-weight", type=float, default=10.0,
                         help="weight setting to render as an enlarged standalone figure")
    args = parser.parse_args()

    runs = load_all_runs()
    if not runs:
        print(f"No runs found in {RESULTS_DIR}. Run run_sweep.py first.")
        return

    weights = sorted(set(r["cfg"]["w_ic"] for r in runs))
    seeds = sorted(set(r["cfg"]["seed"] for r in runs))
    # fixed color per seed so the same seed is comparable across panels
    cmap = plt.cm.tab10
    seed_cmap = lambda s: cmap(seeds.index(s) % 10)

    # ---- grid figure: one panel per weight ----
    ncols = len(weights)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, w in zip(axes, weights):
        runs_w = [r for r in runs if r["cfg"]["w_ic"] == w]
        plot_panel(ax, runs_w, seed_cmap)
        ax.set_title(f"w_data = {w:g}")
    axes[0].legend(fontsize=7, title="solid: L_data\ndashed: L_pde", title_fontsize=7)
    fig.suptitle("Same setting, different seeds: are the loss dynamics reproducible?")
    fig.tight_layout()
    grid_path = FIGURES_DIR / "seed_consistency_grid.png"
    fig.savefig(grid_path, dpi=130)
    print(f"saved: {grid_path}")

    # ---- enlarged single-weight figure (with true_L2 dotted) ----
    w = args.focus_weight
    runs_w = [r for r in runs if r["cfg"]["w_ic"] == w]
    if runs_w:
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        plot_panel(ax2, runs_w, seed_cmap, show_true_l2=True)
        ax2.set_title(f"w_data = {w:g}: L_data (solid), L_pde (dashed), true_L2 (dotted) per seed")
        ax2.legend(fontsize=8)
        focus_path = FIGURES_DIR / f"seed_consistency_dw{w:g}.png"
        fig2.tight_layout()
        fig2.savefig(focus_path, dpi=130)
        print(f"saved: {focus_path}")
    else:
        print(f"note: no runs found with w_data = {w:g}; skipped the focus figure "
              f"(available: {weights})")


if __name__ == "__main__":
    main()