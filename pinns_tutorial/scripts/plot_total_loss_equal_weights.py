"""
scripts/plot_total_loss_equal_weights.py

Plots the TOTAL loss for the equal-weight baseline (w_ic=w_bc=w_pde=1)
only. In this condition the total loss the optimizer actually minimizes
is simply L_ic + L_bc + L_pde (all weights = 1).

We recompute the sum from the components rather than trusting the saved
L_total field, so the definition is explicit and unambiguous here (for
dw=1 the two are identical anyway, since the saved L_total already used
weights of 1).

Two figures (to figures/):
  total_loss_eqw_seeds.png     : L_total for each seed, overlaid
  total_loss_eqw_mean_std.png  : geometric mean +- std across seeds
                                 (log-space stats; band is multiplicative)

Run:
    python scripts/plot_total_loss_equal_weights.py
"""

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


def load_equal_weight_runs():
    runs = []
    for npz_path in sorted(glob.glob(str(RESULTS_DIR / "*.npz"))):
        run_id = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = RESULTS_DIR / f"{run_id}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            cfg = json.load(f)
        if cfg["w_ic"] != TARGET_WEIGHT or cfg["w_bc"] != TARGET_WEIGHT or cfg["w_pde"] != 1.0:
            continue
        runs.append(dict(run_id=run_id, data=np.load(npz_path), cfg=cfg))
    return sorted(runs, key=lambda r: r["cfg"]["seed"])


def main():
    runs = load_equal_weight_runs()
    if not runs:
        print(f"No equal-weight (1/1/1) runs found in {RESULTS_DIR}.")
        return
    print(f"found {len(runs)} runs: {[r['run_id'] for r in runs]}")

    epoch = runs[0]["data"]["epoch"]
    # total loss = sum of components (weights are all 1 in this condition)
    totals = np.stack([r["data"]["L_ic"] + r["data"]["L_bc"] + r["data"]["L_pde"]
                       for r in runs])  # (n_seeds, n_points)

    # ---- figure 1: per-seed overlay ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.cm.tab10
    for i, r in enumerate(runs):
        ax.plot(epoch, totals[i], color=cmap(i % 10), linewidth=1.2,
                label=f"seed {r['cfg']['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss  L_ic + L_bc + L_pde  (log scale)")
    ax.set_title("Equal-weight baseline (1/1/1): total loss per seed")
    ax.legend()
    fig.tight_layout()
    p1 = FIGURES_DIR / "total_loss_eqw_seeds.png"
    fig.savefig(p1, dpi=130)
    print(f"saved: {p1}")

    # ---- figure 2: geometric mean +- std ----
    logs = np.log10(totals)
    mu, sigma = logs.mean(axis=0), logs.std(axis=0)
    mean = 10 ** mu
    lo = 10 ** (mu - sigma)
    hi = 10 ** (mu + sigma)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
    ax2.plot(epoch, mean, color="C3", linewidth=1.8, label="geometric mean")
    ax2.fill_between(epoch, lo, hi, color="C3", alpha=0.2, linewidth=0,
                     label="± std (log-space)")
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("total loss  L_ic + L_bc + L_pde  (log scale)")
    ax2.set_title(f"Equal-weight baseline (1/1/1): total loss, mean ± std over {len(runs)} seeds")
    ax2.legend()
    fig2.tight_layout()
    p2 = FIGURES_DIR / "total_loss_eqw_mean_std.png"
    fig2.savefig(p2, dpi=130)
    print(f"saved: {p2}")


if __name__ == "__main__":
    main()