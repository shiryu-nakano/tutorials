"""
1D Heat Equation baseline (no PINN).

    du/dt = alpha * d2u/dx2,   x in [0, 1],  t >= 0
    IC:  u(x, 0) = sin(pi x)
    BC:  u(0, t) = u(1, t) = 0

Exact solution:
    u(x, t) = sin(pi x) * exp(-alpha * pi^2 * t)

This script:
  1. Solves the PDE with an explicit finite-difference (FTCS) scheme.
  2. Compares against the exact solution and prints the max error.
  3. Saves two figures: snapshots over time, and a space-time heatmap.

The point: watch the initial sin wave keep its shape while decaying
exponentially. That "shape is fixed, amplitude decays" behavior is the
thing a PINN also has to discover.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---- parameters ----
alpha = 1.0          # thermal diffusivity
L = 1.0              # domain length
T = 0.2              # final time
nx = 51             # spatial grid points
dx = L / (nx - 1)

# Explicit FTCS is stable only if r = alpha*dt/dx^2 <= 0.5.
# Pick dt to satisfy that with margin.
r = 0.4
dt = r * dx**2 / alpha
nt = int(T / dt)

x = np.linspace(0, L, nx)


def exact(x, t):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


# ---- finite difference solve (FTCS) ----
u = np.sin(np.pi * x)          # initial condition
u[0] = u[-1] = 0.0             # boundary condition

snapshots = {0.0: u.copy()}
save_times = [0.02, 0.05, 0.1, 0.2]
save_idx = {t: int(round(t / dt)) for t in save_times}

U = np.zeros((nt + 1, nx))     # full space-time field for the heatmap
U[0] = u

for n in range(1, nt + 1):
    u_new = u.copy()
    # interior update: u_i += r * (u_{i+1} - 2 u_i + u_{i-1})
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    u_new[0] = u_new[-1] = 0.0
    u = u_new
    U[n] = u
    for t, idx in save_idx.items():
        if n == idx:
            snapshots[t] = u.copy()

# ---- error check against exact solution ----
t_final = nt * dt
max_err = np.max(np.abs(u - exact(x, t_final)))
print(f"grid: nx={nx}, dt={dt:.2e}, r={r}, steps={nt}")
print(f"final time t={t_final:.4f}")
print(f"max |FD - exact| at final time: {max_err:.3e}")

# ---- figure 1: snapshots ----
fig, ax = plt.subplots(figsize=(7, 4.5))
xf = np.linspace(0, L, 400)
for t in sorted(snapshots):
    ax.plot(x, snapshots[t], "o", ms=3, label=f"FD  t={t:.2f}")
    ax.plot(xf, exact(xf, t), "-", lw=1, alpha=0.6)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Heat equation: sin wave decaying (dots=FD, lines=exact)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "heat_snapshots.png", dpi=130)

# ---- figure 2: space-time heatmap ----
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
im = ax2.imshow(
    U.T, origin="lower", aspect="auto",
    extent=[0, t_final, 0, L], cmap="hot",
)
ax2.set_xlabel("t")
ax2.set_ylabel("x")
ax2.set_title("u(x, t): heat spreading and decaying over time")
fig2.colorbar(im, ax=ax2, label="temperature u")
fig2.tight_layout()
fig2.savefig(FIGURES_DIR / "heat_heatmap.png", dpi=130)

print("saved: heat_snapshots.png, heat_heatmap.png")
