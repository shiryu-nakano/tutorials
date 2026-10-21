"""
scripts/calibrate_llc.py

The hyperparameter-tuning step, isolated. Before any real LLC(t)
measurement, you must find an SGLD step size (lr / epsilon) and
localization (gamma) where the sampler is stable at YOUR checkpoints.

This script loads one trained checkpoint, runs LLC estimation across a
grid of (lr, localization), and plots the per-chain LLC traces for each
config. You then read the plots and pick the config whose traces:
  - do NOT diverge (shoot up without bound)
  - do NOT get stuck flat at ~0 (sampler frozen)
  - settle to a stable plateau, consistent across chains

This mirrors the devinterp calibration notebook, adapted for the PINN
loss. Reading traces by eye is the recommended practice — LLC estimation
is (their words) "more of an art than a science."

Prerequisite: a saved checkpoint. Produce one with, e.g., a training run
that sets save_final_checkpoint=True, or point --ckpt at any *_final.pt.

Run:
    python scripts/calibrate_llc.py --ckpt results/sweep_results/seed0_dw1_final.pt
    python scripts/calibrate_llc.py --ckpt <path> --lrs 1e-5 1e-4 1e-3 --gammas 1 10 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.baseline01 import PINN, DEFAULT_CONFIG
from model.llc_pinn import make_point_sets, estimate_llc

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to a *_final.pt checkpoint")
    parser.add_argument("--lrs", nargs="+", type=float,
                         default=[1e-5, 1e-4, 1e-3], help="SGLD step sizes (epsilon) to try")
    parser.add_argument("--gammas", nargs="+", type=float,
                         default=[1.0, 10.0, 100.0], help="localization strengths to try")
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-draws", type=int, default=200)
    parser.add_argument("--num-burnin", type=int, default=100)
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG["width"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")

    # load checkpoint
    model = PINN(width=args.width, depth=args.depth).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint: {args.ckpt}")

    # fixed point sets for loss evaluation
    points = make_point_sets(device=device)

    n_lr = len(args.lrs)
    n_gamma = len(args.gammas)
    fig, axes = plt.subplots(n_gamma, n_lr, figsize=(4.5 * n_lr, 3.5 * n_gamma),
                             squeeze=False, sharex=True)

    summary = []
    for gi, gamma in enumerate(args.gammas):
        for li, lr in enumerate(args.lrs):
            ax = axes[gi][li]
            try:
                results = estimate_llc(
                    model, points,
                    lr=lr, localization=gamma,
                    num_chains=args.num_chains,
                    num_draws=args.num_draws,
                    num_burnin=args.num_burnin,
                    device=device,
                )
                # result keys differ across devinterp versions — look for
                # both the "llc/mean" and "llc_mean" style, and same for trace
                def pick(d, *names, default=None):
                    for n in names:
                        if n in d:
                            return d[n]
                    return default
                llc_mean = pick(results, "llc/mean", "llc_mean")
                llc_mean = float(llc_mean) if llc_mean is not None else float("nan")
                trace = pick(results, "llc/trace", "llc_trace", "trace")
                if trace is not None:
                    trace = np.asarray(trace)
                    if trace.ndim == 1:
                        trace = trace[None, :]
                    for c in range(trace.shape[0]):
                        ax.plot(trace[c], linewidth=0.8, alpha=0.7)
                    ax.axhline(llc_mean, color="black", linestyle="--", linewidth=1)
                else:
                    ax.text(0.5, 0.5, f"LLC={llc_mean:.2f}\n(no trace key)",
                            ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_title(f"lr={lr:g}, gamma={gamma:g}\nLLC={llc_mean:.2f}", fontsize=9)
                summary.append((lr, gamma, llc_mean))
            except Exception as e:
                ax.text(0.5, 0.5, f"FAILED\n{type(e).__name__}", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, color="red")
                ax.set_title(f"lr={lr:g}, gamma={gamma:g}", fontsize=9)
                print(f"  lr={lr:g} gamma={gamma:g}: FAILED ({e})")
            if li == 0:
                ax.set_ylabel("LLC running est.")
            if gi == n_gamma - 1:
                ax.set_xlabel("draw")

    fig.suptitle(f"SGLD calibration traces — {Path(args.ckpt).name}\n"
                 "pick a cell where chains agree and settle (no divergence, not stuck at 0)")
    fig.tight_layout()
    out_path = FIGURES_DIR / f"llc_calibration_{Path(args.ckpt).stem}.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved calibration figure: {out_path}")

    print("\nLLC estimates by config:")
    print(f"{'lr':>10} {'gamma':>8} {'LLC':>10}")
    for lr, gamma, llc in summary:
        print(f"{lr:>10g} {gamma:>8g} {llc:>10.3f}")
    print("\nNext: pick the (lr, gamma) whose traces are stablest, then use "
          "those in the LLC(t) measurement across checkpoints.")


if __name__ == "__main__":
    main()