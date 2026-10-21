"""
scripts/make_checkpoint.py

Train the equal-weight (1/1/1) heat-equation PINN and save the final
weights, for use as an LLC calibration checkpoint. Single purpose: get a
*_final.pt on disk.

Run:
    python scripts/make_checkpoint.py
    python scripts/make_checkpoint.py --seed 0 --epochs 3000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.baseline01 import train_pinn

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    args = parser.parse_args()

    run_id = f"eqw_seed{args.seed}"
    config = dict(
        seed=args.seed,
        w_ic=1.0, w_bc=1.0, w_pde=1.0,   # equal-weight baseline
        n_epochs=args.epochs,
        log_every=20,
        run_id=run_id,
        out_dir=str(OUT_DIR),
        save_final_checkpoint=True,       # this is the whole point
    )
    history, cfg = train_pinn(config)
    ckpt = OUT_DIR / f"{run_id}_final.pt"
    print(f"\ncheckpoint saved: {ckpt}")
    print(f"final true_L2 = {history['true_L2'][-1]:.3e}")
    print(f"\nnext: python scripts/calibrate_llc.py --ckpt {ckpt}")


if __name__ == "__main__":
    main()