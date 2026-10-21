"""
scripts/model/baseline01.py

PINN model + training routine for the 1D heat equation.
Imported by scripts/baseline_heat.py, scripts/run_sweep.py,
scripts/make_checkpoint.py, scripts/calibrate_llc.py, etc. as:

    from model.baseline01 import PINN, exact_solution, train_pinn, DEFAULT_CONFIG
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# repo_root/scripts/model/baseline01.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = dict(
    # physics
    alpha=1.0,
    L=1.0,
    T=0.2,
    # architecture
    width=32,
    depth=4,
    # sampling
    n_pde=2000,
    n_ic=200,
    n_bc=200,
    # loss weights  (equal-weight baseline)
    w_ic=1.0,
    w_bc=1.0,
    w_pde=1.0,
    # optimization
    lr=1e-3,
    n_epochs=3000,
    log_every=20,
    # bookkeeping
    seed=0,
    run_id="default",
    out_dir=str(REPO_ROOT / "results" / "sweep_results"),
    save_final_checkpoint=False,
)


class PINN(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def exact_solution(x, t, alpha):
    return torch.sin(np.pi * x) * torch.exp(-alpha * np.pi**2 * t)


def train_pinn(config=None, device=None, verbose=True):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg["out_dir"] = str(cfg["out_dir"])  # Path -> str, for json.dump safety

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    alpha, L, T = cfg["alpha"], cfg["L"], cfg["T"]

    model = PINN(width=cfg["width"], depth=cfg["depth"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    n_pde, n_ic, n_bc = cfg["n_pde"], cfg["n_ic"], cfg["n_bc"]

    def sample_pde():
        x = torch.rand(n_pde, 1, device=device, requires_grad=True) * L
        t = torch.rand(n_pde, 1, device=device, requires_grad=True) * T
        return x, t

    x_ic = torch.rand(n_ic, 1, device=device) * L
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic_target = torch.sin(np.pi * x_ic)

    t_bc = torch.rand(n_bc, 1, device=device) * T
    x_bc0 = torch.zeros(n_bc, 1, device=device)
    x_bc1 = torch.ones(n_bc, 1, device=device)

    xe = torch.linspace(0, L, 100, device=device).reshape(-1, 1)
    te = torch.linspace(0, T, 100, device=device).reshape(-1, 1)
    Xe, Te = torch.meshgrid(xe.squeeze(), te.squeeze(), indexing="ij")
    Xe_flat = Xe.reshape(-1, 1)
    Te_flat = Te.reshape(-1, 1)
    with torch.no_grad():
        U_exact_flat = exact_solution(Xe_flat, Te_flat, alpha)

    w_ic, w_bc, w_pde = cfg["w_ic"], cfg["w_bc"], cfg["w_pde"]
    n_epochs, log_every = cfg["n_epochs"], cfg["log_every"]

    history = {"epoch": [], "L_ic": [], "L_bc": [], "L_pde": [], "L_total": [], "true_L2": []}

    for epoch in range(n_epochs + 1):
        optimizer.zero_grad()

        u_ic_pred = model(x_ic, t_ic)
        L_ic = torch.mean((u_ic_pred - u_ic_target) ** 2)

        u_bc0 = model(x_bc0, t_bc)
        u_bc1 = model(x_bc1, t_bc)
        L_bc = torch.mean(u_bc0**2) + torch.mean(u_bc1**2)

        x_f, t_f = sample_pde()
        u_f = model(x_f, t_f)
        u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                                   create_graph=True)[0]
        u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        residual = u_t - alpha * u_xx
        L_pde = torch.mean(residual**2)

        loss = w_ic * L_ic + w_bc * L_bc + w_pde * L_pde
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            with torch.no_grad():
                u_pred_flat = model(Xe_flat, Te_flat)
                true_l2 = torch.sqrt(torch.mean((u_pred_flat - U_exact_flat) ** 2)).item()
            history["epoch"].append(epoch)
            history["L_ic"].append(L_ic.item())
            history["L_bc"].append(L_bc.item())
            history["L_pde"].append(L_pde.item())
            history["L_total"].append(loss.item())
            history["true_L2"].append(true_l2)

    if verbose:
        print(f"[{cfg['run_id']}] done on {device}. final true_L2={history['true_L2'][-1]:.3e}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{cfg['run_id']}.npz",
        epoch=np.array(history["epoch"]),
        L_ic=np.array(history["L_ic"]),
        L_bc=np.array(history["L_bc"]),
        L_pde=np.array(history["L_pde"]),
        L_total=np.array(history["L_total"]),
        true_L2=np.array(history["true_L2"]),
    )
    with open(out_dir / f"{cfg['run_id']}.json", "w") as f:
        json.dump(cfg, f, indent=2)

    if cfg["save_final_checkpoint"]:
        torch.save(model.state_dict(), out_dir / f"{cfg['run_id']}_final.pt")

    return history, cfg


if __name__ == "__main__":
    model = PINN()
    print(model)