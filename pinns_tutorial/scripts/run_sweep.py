"""
Sweep: seed x data-loss weight ratio.

Purpose: before measuring LLC, pin down whether the "data-phase then
physics-phase" crossover seen in the single baseline run is (a) reproducible
across seeds, and (b) shifts systematically with the data/physics loss
weight ratio (w_ic=w_bc vs w_pde). This is a loss-curve-only precursor to
hypothesis H2.

Architecture, alpha, point counts, and lr are held fixed as controls.
Only w_data (=w_ic=w_bc) and seed vary. w_pde is fixed at 1.0, so
w_data effectively IS the ratio w_data/w_pde.

Edit SEEDS / DATA_WEIGHTS below to change the grid. With n_epochs=3000
and 25 runs, expect this to take a while on CPU — reduce n_epochs or the
grid size for a quick smoke test first if needed.
"""

from pinn_heat_lib import train_pinn

from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]
DATA_WEIGHTS = [1.0, 3.0, 10.0, 30.0, 100.0]
N_EPOCHS = 3000
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "sweep_results"

total = len(SEEDS) * len(DATA_WEIGHTS)
count = 0

for data_weight in DATA_WEIGHTS:
    for seed in SEEDS:
        count += 1
        run_id = f"seed{seed}_dw{data_weight:g}"
        print(f"[{count}/{total}] running {run_id} ...")
        config = dict(
            seed=seed,
            w_ic=data_weight,
            w_bc=data_weight,
            w_pde=1.0,
            n_epochs=N_EPOCHS,
            log_every=20,
            run_id=run_id,
            out_dir=OUT_DIR,
        )
        train_pinn(config)

print("sweep complete. results in", OUT_DIR)
