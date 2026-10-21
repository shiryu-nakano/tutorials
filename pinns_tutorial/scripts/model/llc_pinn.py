"""
scripts/model/llc_pinn.py

LLC (Local Learning Coefficient) estimation for the heat-equation PINN,
built on devinterp's SGLD sampler.

The core difficulty: devinterp's standard path evaluates a loss over
(input, target) batches from a DataLoader. Our PINN loss is not of that
form — L_pde needs autograd through the network (u_t, u_xx) at
collocation points, and L_ic/L_bc are evaluated at their own point sets.

We handle this by:
  1. Bundling all the fixed point sets (IC/BC/collocation) into a tiny
     "dataset" of a single batch, so devinterp's loader machinery is
     satisfied with one draw per step.
  2. Providing a custom loss function (evaluate) that recomputes the full
     composite PINN loss for the CURRENT model weights, including the
     autograd-based PDE residual. This closure is what SGLD sees as
     "the loss" whose local geometry defines the LLC.

Because L_pde requires create_graph for second derivatives, each SGLD
step is heavier than a normal forward/backward. Keep chain lengths
modest during calibration.

This module exposes:
  - build_pinn_loss_fn(...)   : returns evaluate(model, batch) closure
  - make_single_batch_loader(): a one-batch loader over the point sets
  - estimate_llc(...)         : wraps devinterp sample + LLCEstimator

References for the API shape: devinterp docs (sampler.sample +
LLCEstimator + default_nbeta). LLC estimation is hyperparameter-
sensitive; calibrate lr (epsilon) and localization (gamma) with
scripts/calibrate_llc.py before trusting any number.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_point_sets(alpha=1.0, L=1.0, T=0.2, n_pde=2000, n_ic=200, n_bc=200,
                    seed=0, device="cpu"):
    """Fixed IC/BC/collocation point sets used for LLC loss evaluation.

    Unlike training (which resamples collocation points every step), we
    FIX them here so the loss landscape SGLD explores is a fixed
    function of the weights — resampling would inject extra noise that
    contaminates the LLC estimate.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    x_pde = torch.rand(n_pde, 1, generator=g) * L
    t_pde = torch.rand(n_pde, 1, generator=g) * T

    x_ic = torch.rand(n_ic, 1, generator=g) * L
    t_ic = torch.zeros(n_ic, 1)
    u_ic_target = torch.sin(np.pi * x_ic)

    t_bc = torch.rand(n_bc, 1, generator=g) * T
    x_bc0 = torch.zeros(n_bc, 1)
    x_bc1 = torch.ones(n_bc, 1)

    points = dict(
        x_pde=x_pde.to(device), t_pde=t_pde.to(device),
        x_ic=x_ic.to(device), t_ic=t_ic.to(device),
        u_ic_target=u_ic_target.to(device),
        t_bc=t_bc.to(device), x_bc0=x_bc0.to(device), x_bc1=x_bc1.to(device),
    )
    return points


def build_pinn_loss_fn(points, alpha=1.0, w_ic=1.0, w_bc=1.0, w_pde=1.0):
    """Return evaluate(model, batch) -> (loss, dict) as devinterp expects.

    The `batch` argument is ignored (all point sets live in `points`);
    it exists only to match devinterp's calling convention. The loss is
    the composite PINN loss at the model's CURRENT weights.
    """
    x_pde, t_pde = points["x_pde"], points["t_pde"]
    x_ic, t_ic, u_ic_target = points["x_ic"], points["t_ic"], points["u_ic_target"]
    t_bc, x_bc0, x_bc1 = points["t_bc"], points["x_bc0"], points["x_bc1"]

    def evaluate(model, batch):
        # devinterp may call this under torch.no_grad(); the PINN loss
        # needs autograd (PDE residual = first/second input derivatives),
        # so force grad on for the whole computation.
        with torch.enable_grad():
            # IC
            u_ic = model(x_ic, t_ic)
            L_ic = torch.mean((u_ic - u_ic_target) ** 2)
            # BC
            u_bc0 = model(x_bc0, t_bc)
            u_bc1 = model(x_bc1, t_bc)
            L_bc = torch.mean(u_bc0 ** 2) + torch.mean(u_bc1 ** 2)
            # PDE residual — needs grads w.r.t. inputs, with create_graph
            # so the outer SGLD gradient (w.r.t. weights) can flow through.
            xp = x_pde.clone().requires_grad_(True)
            tp = t_pde.clone().requires_grad_(True)
            u = model(xp, tp)
            u_t = torch.autograd.grad(u, tp, torch.ones_like(u), create_graph=True)[0]
            u_x = torch.autograd.grad(u, xp, torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, xp, torch.ones_like(u_x), create_graph=True)[0]
            residual = u_t - alpha * u_xx
            L_pde = torch.mean(residual ** 2)

            loss = w_ic * L_ic + w_bc * L_bc + w_pde * L_pde
        # devinterp expects (loss, aux_dict); aux is free-form
        return loss, {"L_ic": L_ic.detach(), "L_bc": L_bc.detach(),
                      "L_pde": L_pde.detach()}

    return evaluate


def make_single_batch_loader(n_pde):
    """A one-batch DataLoader whose only job is to make devinterp's loop
    run exactly one 'batch' per step. Contents are dummy — the real
    point sets are captured in the evaluate closure. Batch size = n_pde
    so devinterp's internal n (dataset size) matches the PDE point count,
    which is what enters nbeta scaling."""
    dummy = torch.zeros(n_pde, 1)
    ds = TensorDataset(dummy, dummy)
    return DataLoader(ds, batch_size=n_pde, shuffle=False)


def estimate_llc(model, points, *, lr, localization, nbeta=None,
                 num_chains=4, num_draws=200, num_burnin=100, device="cpu",
                 alpha=1.0, w_ic=1.0, w_bc=1.0, w_pde=1.0, seed=0):
    """Run SGLD-based LLC estimation at the model's current weights.

    lr           : SGLD step size (epsilon) — the primary knob
    localization : SGLD localization strength (gamma)
    nbeta        : inverse temperature; if None, uses default_nbeta(loader)
    Returns the devinterp results dict (includes 'llc/mean', 'llc/trace').
    """
    from devinterp.slt.sampler import sample, LLCEstimator
    from devinterp.optim import SGLD
    from devinterp.utils import default_nbeta

    loader = make_single_batch_loader(points["x_pde"].shape[0])
    evaluate = build_pinn_loss_fn(points, alpha=alpha,
                                  w_ic=w_ic, w_bc=w_bc, w_pde=w_pde)

    if nbeta is None:
        nbeta = default_nbeta(loader)

    # devinterp >=1.x LLCEstimator needs the loss at the starting
    # checkpoint (ell_0): LLC = nbeta * (E[loss] - ell_0).
    # It wants a tensor, so keep it as one (detached).
    with torch.enable_grad():
        init_loss_t, _ = evaluate(model, None)
    init_loss_t = init_loss_t.detach()

    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        init_loss=init_loss_t,
        nbeta=nbeta,
        device=device,
    )

    sample(
        model,
        loader,
        callbacks=[llc_estimator],
        evaluate=evaluate,
        sampling_method=SGLD,
        # SGLD's own args go in optimizer_kwargs (NOT sampling_method_kwargs)
        optimizer_kwargs=dict(lr=lr, localization=localization, nbeta=nbeta),
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin,
        init_loss=init_loss_t,
        device=device,
        seed=seed,
    )

    return llc_estimator.get_results()