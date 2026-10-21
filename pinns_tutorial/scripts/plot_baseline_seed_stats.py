"""
scripts/plot_baseline_seed_stats.py

Two figures for the equal-weight baseline (w_ic = w_bc = w_pde = 1,
i.e. the dw1 runs already saved by the sweep):

  1. figures/baseline_seeds_overlay.png
     2x2 panels (L_ic, L_bc, L_pde, true_L2), each overlaying all
     seeds' raw curves — seed-to-seed variation shown directly.

  2. figures/baseline_seed_mean_std.png
     One panel, all four metrics as mean ± std across seeds.
     Statistics are computed in log10 space (so the line is the
     geometric mean and the band is a multiplicative std), because the
     curves span several decades: arithmetic mean/std on a log plot
     would be dominated by the highest seed and the band would dip
     below zero. Pass --linear-stats if you want plain arithmetic
     mean/std instead.

Run:
    python scripts/plot_baseline_seed_stats.py
    python scripts/plot_baseline_seed_stats.py --linear-stats
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

TARGET_WEIGHT = 1.0

METRICS = {
    "L_ic": ("C0", "L_ic (IC)"),
    "L_bc": ("C1", "L_bc (BC)"),
    "L_pde": ("C2", "L_pde (physics)"),
    "true_L2": ("black", "true L2 error (vs exact)"),
}


def load_equal_weight_runs():
    runs = []
    for npz_path in sorted(glob.glob(str(RESULTS_DIR / "*.npz"))):
        run_id = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = RESULTS_DIR / f"{run_id}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            cfg = json.load(f)
        if cfg["w_ic"] != TARGET_WEIGHT or cfg["w_pde"] != 1.0:
            continue
        runs.append(dict(run_id=run_id, data=np.load(npz_path), cfg=cfg))
    return sorted(runs, key=lambda r: r["cfg"]["seed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-stats", action="store_true",
                         help="use arithmetic mean/std instead of log-space stats")
    args = parser.parse_args()

    runs = load_equal_weight_runs()
    if not runs:
        print(f"No equal-weight (w={TARGET_WEIGHT:g}) runs found in {RESULTS_DIR}.")
        return
    print(f"found {len(runs)} runs: {[r['run_id'] for r in runs]}")

    epoch = runs[0]["data"]["epoch"]
    seeds = [r["cfg"]["seed"] for r in runs]
    seed_colors = plt.cm.tab10

    # ---- figure 1: per-seed overlay, one panel per metric ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, (_, label)) in zip(axes.flat, METRICS.items()):
        for i, r in enumerate(runs):
            ax.plot(epoch, r["data"][key], color=seed_colors(i % 10),
                    linewidth=1.1, label=f"seed {r['cfg']['seed']}")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_ylabel("loss")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Equal-weight baseline (1/1/1): per-seed loss curves ({len(runs)} seeds)")
    fig.tight_layout()
    overlay_path = FIGURES_DIR / "baseline_seeds_overlay.png"
    fig.savefig(overlay_path, dpi=130)
    print(f"saved: {overlay_path}")

    # ---- figure 2: mean +- std across seeds ----
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
    for key, (color, label) in METRICS.items():
        stacked = np.stack([r["data"][key] for r in runs])  # (n_seeds, n_points)
        if args.linear_stats:
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            lo = np.clip(mean - std, a_min=np.finfo(float).tiny, a_max=None)
            hi = mean + std
        else:
            logs = np.log10(stacked)
            mu = logs.mean(axis=0)
            sigma = logs.std(axis=0)
            mean = 10 ** mu
            lo = 10 ** (mu - sigma)
            hi = 10 ** (mu + sigma)
        ls = "--" if key == "true_L2" else "-"
        ax2.plot(epoch, mean, color=color, linestyle=ls, linewidth=1.6, label=label)
        ax2.fill_between(epoch, lo, hi, color=color, alpha=0.18, linewidth=0)

    stats_kind = "arithmetic" if args.linear_stats else "log-space (geometric)"
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss (log scale)")
    ax2.set_title(f"Equal-weight baseline (1/1/1): {stats_kind} mean ± std over {len(runs)} seeds")
    ax2.legend()
    fig2.tight_layout()
    stats_path = FIGURES_DIR / "baseline_seed_mean_std.png"
    fig2.savefig(stats_path, dpi=130)
    print(f"saved: {stats_path}")


if __name__ == "__main__":
    main()