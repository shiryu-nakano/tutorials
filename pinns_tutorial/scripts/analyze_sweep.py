"""
Analyze run_sweep.py output.

Key design point: detecting the "data -> physics" crossover is NOT a
simple first-crossing search. L_pde starts near zero at epoch 0 because
an (almost) all-zero network output trivially satisfies this homogeneous
PDE (u=0 => u_t=0, u_xx=0). So L_pde < L_data at epoch 0 for free, before
any real physics has been "learned". As training pulls the network away
from zero to fit IC/BC, L_pde typically spikes up above L_data, and only
later does L_data crash down and stay below L_pde for good.

The crossover we actually care about is that LAST one: the last epoch at
which L_data transitions from >= L_pde to < L_pde, and then stays below
persistently (not just a noisy blip). We find this by:
  1. Smoothing both curves with a moving average.
  2. Scanning for every index where the smoothed curves cross from
     L_data >= L_pde to L_data < L_pde.
  3. Keeping the LAST such crossing that persists (L_data stays below
     L_pde) for at least `persist_len` consecutive logged points
     afterward (or through the end of the run).

If no persistent crossover is found, we report NaN for that run rather
than guessing.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
    
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "sweep_results"
def moving_average(a, window):
    if window <= 1:
        return a.copy()
    kernel = np.ones(window) / window
    # 'same' mode keeps array length; edges are slightly under-smoothed
    return np.convolve(a, kernel, mode="same")


def find_persistent_crossover(epoch, L_data, L_pde, smooth_window=5, persist_len=10):
    """Return the epoch of the last persistent L_data -> below -> L_pde crossing."""
    Ld = moving_average(L_data, smooth_window)
    Lp = moving_average(L_pde, smooth_window)
    below = Ld < Lp  # boolean array: is data loss below physics loss at each logged point

    n = len(below)
    crossings = []
    for i in range(1, n):
        if not below[i - 1] and below[i]:
            # check persistence: does it stay below for persist_len points (or to the end)?
            end = min(i + persist_len, n)
            if np.all(below[i:end]):
                crossings.append(i)

    if not crossings:
        return np.nan
    last_idx = crossings[-1]
    return epoch[last_idx]


def load_all_runs():
    npz_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.npz")))
    runs = []
    for npz_path in npz_files:
        run_id = os.path.splitext(os.path.basename(npz_path))[0]
        json_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
        data = np.load(npz_path)
        with open(json_path) as f:
            cfg = json.load(f)
        runs.append(dict(run_id=run_id, data=data, cfg=cfg))
    return runs


def main():
    runs = load_all_runs()
    if not runs:
        print(f"No runs found in {RESULTS_DIR}. Run run_sweep.py first.")
        return

    rows = []
    for r in runs:
        d, cfg = r["data"], r["cfg"]
        L_data = d["L_ic"] + d["L_bc"]
        crossover = find_persistent_crossover(d["epoch"], L_data, d["L_pde"])
        rows.append(dict(
            run_id=r["run_id"],
            seed=cfg["seed"],
            data_weight=cfg["w_ic"],  # w_ic == w_bc by construction in the sweep
            crossover_epoch=crossover,
            final_true_L2=d["true_L2"][-1],
        ))

    # ---- summary table ----
    weights = sorted(set(row["data_weight"] for row in rows))
    seeds = sorted(set(row["seed"] for row in rows))

    print(f"{'data_weight':>12} {'seed':>5} {'crossover_epoch':>16} {'final_true_L2':>14}")
    for row in rows:
        ce = "NaN" if np.isnan(row["crossover_epoch"]) else f"{row['crossover_epoch']:.0f}"
        print(f"{row['data_weight']:>12g} {row['seed']:>5d} {ce:>16} {row['final_true_L2']:>14.3e}")

    # save as csv too
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("run_id,seed,data_weight,crossover_epoch,final_true_L2\n")
        for row in rows:
            f.write(f"{row['run_id']},{row['seed']},{row['data_weight']},"
                     f"{row['crossover_epoch']},{row['final_true_L2']}\n")
    print(f"\nsaved summary table to {csv_path}")

    # ---- plot 1: crossover epoch vs data weight (per-seed scatter + mean/std) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(row["data_weight"], row["crossover_epoch"], color="gray", alpha=0.5, s=25)

    means, stds = [], []
    for w in weights:
        vals = np.array([row["crossover_epoch"] for row in rows if row["data_weight"] == w])
        vals = vals[~np.isnan(vals)]
        means.append(np.mean(vals) if len(vals) else np.nan)
        stds.append(np.std(vals) if len(vals) else np.nan)
    ax.errorbar(weights, means, yerr=stds, color="C0", marker="o", capsize=4,
                label="mean ± std across seeds")

    ax.set_xscale("log")
    ax.set_xlabel("data weight  w_ic = w_bc  (w_pde fixed = 1)")
    ax.set_ylabel("crossover epoch (data -> physics)")
    ax.set_title("Does the data->physics crossover shift with loss weight ratio?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "crossover_vs_weight.png"), dpi=130)

    # ---- plot 2: final true_L2 vs data weight (sanity check) ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax2.scatter(row["data_weight"], row["final_true_L2"], color="gray", alpha=0.5, s=25)
    means_l2 = []
    for w in weights:
        vals = np.array([row["final_true_L2"] for row in rows if row["data_weight"] == w])
        means_l2.append(np.mean(vals))
    ax2.plot(weights, means_l2, color="C1", marker="o", label="mean across seeds")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("data weight  w_ic = w_bc  (w_pde fixed = 1)")
    ax2.set_ylabel("final true L2 error")
    ax2.set_title("Final accuracy vs loss weight ratio")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "final_l2_vs_weight.png"), dpi=130)

    # ---- plot 3: overlay of all L_data vs L_pde curves, colored by weight ----
    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis
    for i, w in enumerate(weights):
        color = cmap(i / max(len(weights) - 1, 1))
        for r in runs:
            if r["cfg"]["w_ic"] != w:
                continue
            d = r["data"]
            L_data = d["L_ic"] + d["L_bc"]
            ratio = L_data / d["L_pde"]
            ax3.plot(d["epoch"], ratio, color=color, alpha=0.6, linewidth=1)
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=1, label="L_data = L_pde")
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("L_data / L_pde")
    ax3.set_title("Data/physics loss ratio over training (color = data weight, dark->light = low->high)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, "ratio_curves_overlay.png"), dpi=130)

    print("saved: crossover_vs_weight.png, final_l2_vs_weight.png, ratio_curves_overlay.png")


if __name__ == "__main__":
    main()
