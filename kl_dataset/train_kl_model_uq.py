#!/usr/bin/env python3
"""
train_kl_model_uq.py
====================
Uncertainty-aware version of train_kl_model.py: every prediction comes with
an error bar  g ± σ  for g+ and g×.

How the error bar is built
--------------------------
1. Heteroscedastic heads (aleatoric uncertainty, Nix & Weigend 1994;
   Kendall & Gal 2017).  Each head outputs a mean μ(x) and a variance
   σ²(x) and is trained with the Gaussian negative log-likelihood
        NLL = ½ log σ² + (y − μ)² / (2σ²).
   The network learns large σ for galaxies whose shear it cannot pin down
   (e.g. kinematically asymmetric discs) and small σ where it can.  We use
   β-NLL (Seitzer et al. 2022): each term is weighted by stop_grad(σ²)^β,
   which avoids the known failure where plain NLL fits the mean poorly.
   The first --mse_warmup epochs fit the mean alone with MSE.

2. Deep ensemble (epistemic uncertainty, Lakshminarayanan et al. 2017).
   --n_members networks are trained from different seeds on the same
   splits.  For a galaxy, with member means μ_m and variances σ²_m:
        μ        = mean_m μ_m
        σ²_alea  = mean_m σ²_m          (noise the data can't resolve)
        σ²_epi   = var_m  μ_m           (models disagree → lack of data)
        σ²_total = σ²_alea + σ²_epi
   Optional MC dropout (--mc_samples) adds dropout passes per member.

3. Recalibration on the validation set.  NN variances are often over- or
   under-confident, so a single scale s per component is fitted on the
   validation set so that the standardised residuals z = (y − μ)/(s σ)
   have unit variance.  Normalised split-conformal intervals (Lei et al.
   2018) are also reported: their half-width q·σ, with q set on the
   validation set, has a distribution-free marginal coverage guarantee.

Outputs (in --outdir)
---------------------
  member_<k>/best_model.pt, loss_curves.png   one per ensemble member
  split_{train,val,test}.csv                  shared splits
  predictions_test_uq.csv   per row: g1/g2 prediction, σ_total (raw and
                            calibrated), σ_alea, σ_epi, conformal widths
  uq_summary_test.csv       RMSE, NLL, coverage, calibration scale, bias m/c
  uq_pred_vs_true_test.png  predictions with error bars
  uq_calibration_test.png   expected vs observed interval coverage
  uq_zscore_test.png        standardised residuals vs N(0, 1)
  uq_rmse_vs_sigma_test.png does predicted σ match the actual error?

Usage
-----
  # Train all members in one job, then evaluate the ensemble
  python train_kl_model_uq.py --use_original_image \
      --images_root kl_dataset/images \
      --csv kl_dataset/data/dataset_plan_to500_10draws.csv \
      --outdir kl_dataset/model_output_uq --n_members 5 ...

  # Or one member per SLURM array task, then a final aggregation job
  python train_kl_model_uq.py ... --member $SLURM_ARRAY_TASK_ID
  python train_kl_model_uq.py ... --aggregate_only
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm, spearmanr
from torch.utils.data import DataLoader

from train_kl_model import (
    KLShearDataset,
    TwoStreamShearNet,
    build_splits,
    filter_by_vel_fill,
    filter_to_existing,
)

COMPS   = [(0, "g1", "g+", "#2E86AB"), (1, "g2", "g×", "#E84855")]
MIN_VAR = 1e-8                    # variance floor (σ ≥ 1e-4)
PRIOR_SIGMA = 0.05                # initial σ ≈ spread of the shear draws


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class TwoStreamShearNetUQ(TwoStreamShearNet):
    """
    Same backbone as TwoStreamShearNet; each regression head now outputs
    (mean, raw variance).  σ² = softplus(raw) + MIN_VAR keeps it positive.
    """

    def __init__(self, pretrained: bool = True, freeze_stem: bool = False):
        super().__init__(pretrained=pretrained, freeze_stem=freeze_stem)
        feat_dim = 1280

        def head():
            return nn.Sequential(
                nn.Linear(feat_dim, 256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),       nn.SiLU(),
                nn.Linear(64, 2),
            )
        self.head_g1, self.head_g2 = head(), head()
        # Start with σ ≈ PRIOR_SIGMA so early NLL is sensible
        raw0 = float(np.log(np.expm1(PRIOR_SIGMA ** 2)))
        for h in (self.head_g1, self.head_g2):
            with torch.no_grad():
                h[-1].bias[1].fill_(raw0)
                h[-1].weight[1].mul_(0.1)

    def features(self, photo, vel):
        f = torch.cat([self.photo_stem(photo), self.vel_stream(vel)], dim=1)
        f = self.shared_body(self.fusion_proj(f))
        return self.pool(f).flatten(1)

    def forward(self, photo, vel):
        z  = self.features(photo, vel)
        o1, o2 = self.head_g1(z), self.head_g2(z)
        mean = torch.stack([o1[:, 0], o2[:, 0]], dim=1)
        raw  = torch.stack([o1[:, 1], o2[:, 1]], dim=1)
        var  = nn.functional.softplus(raw) + MIN_VAR
        return mean, var


# ═══════════════════════════════════════════════════════════════════════════════
# Losses and epochs
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_nll(mean, var, y):
    """Element-wise Gaussian NLL including the ½ log 2π constant."""
    return 0.5 * (torch.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def run_epoch_uq(model, loader, optimizer, device, training: bool,
                 beta: float = 0.5, mse_only: bool = False):
    model.train() if training else model.eval()
    tot_loss = tot_nll = 0.0
    means, vars_, labels = [], [], []

    with (torch.enable_grad() if training else torch.no_grad()):
        for photo, vel, y in loader:
            photo, vel, y = photo.to(device), vel.to(device), y.to(device)
            mean, var = model(photo, vel)
            nll = gaussian_nll(mean, var, y)

            if mse_only:
                loss = ((mean - y) ** 2).mean()
            elif beta > 0:
                loss = (nll * var.detach() ** beta).mean()     # β-NLL
            else:
                loss = nll.mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            b = y.size(0)
            tot_loss += loss.item() * b
            tot_nll  += nll.mean().item() * b
            means.append(mean.detach().cpu()); vars_.append(var.detach().cpu())
            labels.append(y.detach().cpu())

    means  = torch.cat(means).numpy()
    vars_  = torch.cat(vars_).numpy()
    labels = torch.cat(labels).numpy()
    n = len(labels)
    return {
        "loss": tot_loss / n, "nll": tot_nll / n,
        "rmse_g1": float(np.sqrt(np.mean((means[:, 0] - labels[:, 0]) ** 2))),
        "rmse_g2": float(np.sqrt(np.mean((means[:, 1] - labels[:, 1]) ** 2))),
        "sigma_g1": float(np.sqrt(np.median(vars_[:, 0]))),
        "sigma_g2": float(np.sqrt(np.median(vars_[:, 1]))),
    }


@torch.no_grad()
def predict_member(model, loader, device, mc_samples: int = 0):
    """
    Returns means, vars of shape (S, N, 2) and labels (N, 2).
    S = 1 (deterministic) or mc_samples (dropout kept on, BatchNorm frozen).
    """
    model.eval()
    if mc_samples > 0:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    S = max(1, mc_samples)
    mus, vs, labels = [], [], []
    for photo, vel, y in loader:
        photo, vel = photo.to(device), vel.to(device)
        bm, bv = [], []
        for _ in range(S):
            mean, var = model(photo, vel)
            bm.append(mean.cpu().numpy()); bv.append(var.cpu().numpy())
        mus.append(np.stack(bm)); vs.append(np.stack(bv))
        labels.append(y.numpy())
    model.eval()
    return (np.concatenate(mus, axis=1), np.concatenate(vs, axis=1),
            np.concatenate(labels))


# ═══════════════════════════════════════════════════════════════════════════════
# Uncertainty statistics
# ═══════════════════════════════════════════════════════════════════════════════

def combine(mus, vars_):
    """Gaussian-mixture moments over all members × MC samples, (S,N,2)."""
    mean = mus.mean(0)
    alea = vars_.mean(0)
    epi  = mus.var(0)                     # population variance over samples
    return mean, alea, epi


def coverage(y, mu, sigma, k):
    return np.mean(np.abs(y - mu) <= k * sigma, axis=0)


def gauss_nll_np(y, mu, sigma):
    return np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                   + (y - mu) ** 2 / (2 * sigma ** 2), axis=0)


def variance_scale(y, mu, sigma):
    """s such that z = (y − μ)/(s σ) has unit variance (per component)."""
    return np.sqrt(np.mean(((y - mu) / sigma) ** 2, axis=0))


def conformal_q(y, mu, sigma, alpha):
    """Normalised split-conformal multiplier for 1 − alpha coverage."""
    scores = np.abs(y - mu) / sigma
    n = scores.shape[0]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, axis=0, method="higher")


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_member_curves(hist, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist["train_nll"], label="train"); axes[0].plot(hist["val_nll"], label="val")
    axes[0].set_title("Gaussian NLL"); axes[0].set_ylabel("NLL")
    for (_, tag, sym, color) in COMPS:
        axes[1].plot(hist[f"val_rmse_{tag}"], color=color, label=f"val {sym}")
        axes[2].plot(hist[f"val_sigma_{tag}"], color=color, label=f"median σ {sym}")
        axes[2].plot(hist[f"val_rmse_{tag}"], color=color, ls=":", label=f"RMSE {sym}")
    axes[1].set_title("Validation RMSE"); axes[2].set_title("Predicted σ vs actual RMSE (val)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[1].set_yscale("log"); axes[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_pred_vs_true_err(y, mu, sig, out_path, n_show=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n_show, len(y)), replace=False)
    lim = np.abs(y).max() * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (c, _, sym, color) in zip(axes, COMPS):
        ax.errorbar(y[idx, c], mu[idx, c], yerr=sig[idx, c], fmt="o", ms=3,
                    color=color, ecolor=color, elinewidth=0.7, alpha=0.5,
                    capsize=0)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"{sym} true"); ax.set_ylabel(f"{sym} predicted ± 1σ (calibrated)")
        ax.set_title(f"{sym}  ({len(idx)} of {len(y)} test rows shown)")
        ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_calibration(y, mu, sig_raw, sig_cal, out_path):
    ps = np.linspace(0.05, 0.95, 19)
    ks = norm.ppf(0.5 + ps / 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (c, _, sym, color) in zip(axes, COMPS):
        for sig, ls, lab in ((sig_raw, ":", "raw"), (sig_cal, "-", "calibrated")):
            obs = [np.mean(np.abs(y[:, c] - mu[:, c]) <= k * sig[:, c]) for k in ks]
            ax.plot(ps, obs, ls, marker="o", ms=3, color=color, label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.set_xlabel("expected coverage of central interval")
        ax.set_ylabel("observed coverage (test)")
        ax.set_title(f"{sym}: above diagonal = σ too large, below = too small",
                     fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_zscores(y, mu, sig_cal, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.linspace(-5, 5, 200)
    for ax, (c, _, sym, color) in zip(axes, COMPS):
        z = (y[:, c] - mu[:, c]) / sig_cal[:, c]
        ax.hist(np.clip(z, -5, 5), bins=50, density=True, color=color, alpha=0.7)
        ax.plot(xs, norm.pdf(xs), "k-", lw=1.2, label="N(0, 1)")
        ax.set_title(f"{sym}: (y − μ)/σ  mean={z.mean():+.2f}  std={z.std():.2f}")
        ax.set_xlabel("standardised residual"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_rmse_vs_sigma(y, mu, sig_cal, out_path, nbins=10):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (c, _, sym, color) in zip(axes, COMPS):
        s = sig_cal[:, c]; e = y[:, c] - mu[:, c]
        edges = np.unique(np.quantile(s, np.linspace(0, 1, nbins + 1)))
        idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
        xb = [np.sqrt(np.mean(s[idx == i] ** 2)) for i in range(len(edges) - 1) if (idx == i).any()]
        yb = [np.sqrt(np.mean(e[idx == i] ** 2)) for i in range(len(edges) - 1) if (idx == i).any()]
        ax.plot(xb, yb, "-o", color=color, label="binned by predicted σ")
        lo, hi = min(xb + yb) * 0.8, max(xb + yb) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted σ (RMS in bin)"); ax.set_ylabel("actual RMSE in bin")
        ax.set_title(f"{sym}: does a larger σ mean a larger error?", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Training one member
# ═══════════════════════════════════════════════════════════════════════════════

def make_loader(ds, args, device, shuffle, seed=None):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, generator=gen,
                      pin_memory=(device.type == "cuda"),
                      persistent_workers=(args.num_workers > 0))


def train_member(k, args, train_ds, val_ds, device, out_dir):
    seed = args.seed + k
    torch.manual_seed(seed); np.random.seed(seed)
    mdir = out_dir / f"member_{k}"
    mdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'═'*70}\n  Ensemble member {k}  (seed {seed})\n{'═'*70}")

    train_loader = make_loader(train_ds, args, device, shuffle=True, seed=seed)
    val_loader   = make_loader(val_ds,   args, device, shuffle=False)

    model = TwoStreamShearNetUQ(pretrained=not args.no_pretrain,
                                freeze_stem=args.freeze_stem).to(device)
    stem_params  = list(model.photo_stem.parameters())
    stem_ids     = {id(p) for p in stem_params}
    other_params = [p for p in model.parameters() if id(p) not in stem_ids]
    optimizer = optim.AdamW([
        {"params": stem_params,  "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    hist = {k_: [] for k_ in ["train_nll", "val_nll", "val_rmse_g1", "val_rmse_g2",
                              "val_sigma_g1", "val_sigma_g2"]}
    best, patience, ckpt_path = float("inf"), 0, mdir / "best_model.pt"

    def save(epoch, val_nll):
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "val_nll": val_nll, "member": k, "seed": seed,
                    "args": vars(args)}, ckpt_path)

    for epoch in range(1, args.epochs + 1):
        warm = epoch <= args.mse_warmup
        tr = run_epoch_uq(model, train_loader, optimizer, device, True,
                          args.beta, mse_only=warm)
        vl = run_epoch_uq(model, val_loader, None, device, False,
                          args.beta, mse_only=warm)
        scheduler.step()
        hist["train_nll"].append(tr["nll"]); hist["val_nll"].append(vl["nll"])
        for _, tag, _, _ in COMPS:
            hist[f"val_rmse_{tag}"].append(vl[f"rmse_{tag}"])
            hist[f"val_sigma_{tag}"].append(vl[f"sigma_{tag}"])

        print(f"[m{k}] Epoch {epoch:3d}/{args.epochs}{' (MSE warm-up)' if warm else ''} | "
              f"NLL train={tr['nll']:+.3f} val={vl['nll']:+.3f} | "
              f"val RMSE g+={vl['rmse_g1']:.4f} g×={vl['rmse_g2']:.4f} | "
              f"val median σ g+={vl['sigma_g1']:.4f} g×={vl['sigma_g2']:.4f}")

        if warm:
            continue                      # variance head not trained yet
        if vl["nll"] < best:
            best, patience = vl["nll"], 0
            save(epoch, vl["nll"])
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[m{k}] Early stopping at epoch {epoch}.")
                break

    if not ckpt_path.exists():            # e.g. epochs <= mse_warmup
        save(epoch, hist["val_nll"][-1])
    plot_member_curves(hist, mdir / "loss_curves.png")
    pd.DataFrame(hist).to_csv(mdir / "history.csv", index_label="epoch")
    print(f"[m{k}] Best val NLL {best:+.4f} → {ckpt_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(ckpts, ds, args, device):
    loader = make_loader(ds, args, device, shuffle=False)
    all_mu, all_var, labels = [], [], None
    for c in ckpts:
        model = TwoStreamShearNetUQ(pretrained=False).to(device)
        model.load_state_dict(torch.load(c, map_location=device,
                                         weights_only=False)["model_state_dict"])
        mu, var, labels = predict_member(model, loader, device, args.mc_samples)
        all_mu.append(mu); all_var.append(var)
        del model
    return np.concatenate(all_mu), np.concatenate(all_var), labels


def evaluate_ensemble(args, val_ds, test_ds, df_test, device, out_dir):
    ckpts = sorted(out_dir.glob("member_*/best_model.pt"))
    if not ckpts:
        raise SystemExit(f"No member_*/best_model.pt found in {out_dir}")
    if len(ckpts) < args.n_members:
        print(f"  [WARN] Found {len(ckpts)} of {args.n_members} members; "
              f"using what exists.")
    print(f"\n── Ensemble evaluation: {len(ckpts)} members"
          f"{f' × {args.mc_samples} MC-dropout passes' if args.mc_samples else ''} ──")

    mu_v, var_v, y_v = ensemble_predict(ckpts, val_ds,  args, device)
    mu_t, var_t, y_t = ensemble_predict(ckpts, test_ds, args, device)

    m_v, a_v, e_v = combine(mu_v, var_v)
    m_t, a_t, e_t = combine(mu_t, var_t)
    sig_v = np.sqrt(a_v + e_v)
    sig_t = np.sqrt(a_t + e_t)

    # Calibration fitted on validation, applied to test
    scale  = variance_scale(y_v, m_v, sig_v)
    sig_tc = sig_t * scale
    q68 = conformal_q(y_v, m_v, sig_v, alpha=1 - 0.6827)
    q95 = conformal_q(y_v, m_v, sig_v, alpha=1 - 0.9545)

    summary = {"n_members": len(ckpts), "mc_samples": args.mc_samples,
               "beta": args.beta, "n_test": len(y_t)}
    print(f"\n  {'':28}{'g+':>12}{'g×':>12}")
    def row(label, vals, fmt="{:>12.4f}", key=None):
        print(f"  {label:<28}" + "".join(fmt.format(v) for v in vals))
        if key:
            for (_, tag, _, _), v in zip(COMPS, vals):
                summary[f"{key}_{tag}"] = float(v)

    err = y_t - m_t
    row("RMSE",                         np.sqrt(np.mean(err ** 2, 0)), key="rmse")
    row("median σ_total (raw)",         np.median(sig_t, 0), key="median_sigma_raw")
    row("median σ_aleatoric",           np.median(np.sqrt(a_t), 0), key="median_sigma_alea")
    row("median σ_epistemic",           np.median(np.sqrt(e_t), 0), key="median_sigma_epi")
    row("epistemic share of variance",  np.mean(e_t / (a_t + e_t), 0),
        "{:>11.0%} ", key="epi_var_share")
    row("calibration scale s (val)",    scale, key="calib_scale")
    row("NLL raw",                      gauss_nll_np(y_t, m_t, sig_t), "{:>12.3f}", key="nll_raw")
    row("NLL calibrated",               gauss_nll_np(y_t, m_t, sig_tc), "{:>12.3f}", key="nll_cal")
    row("1σ coverage raw (68.3%)",      coverage(y_t, m_t, sig_t, 1),  "{:>11.1%} ", key="cov1_raw")
    row("1σ coverage calibrated",       coverage(y_t, m_t, sig_tc, 1), "{:>11.1%} ", key="cov1_cal")
    row("2σ coverage raw (95.4%)",      coverage(y_t, m_t, sig_t, 2),  "{:>11.1%} ", key="cov2_raw")
    row("2σ coverage calibrated",       coverage(y_t, m_t, sig_tc, 2), "{:>11.1%} ", key="cov2_cal")
    row("conformal 68% coverage",       coverage(y_t, m_t, sig_t * q68, 1), "{:>11.1%} ", key="cov68_conformal")
    row("conformal 95% coverage",       coverage(y_t, m_t, sig_t * q95, 1), "{:>11.1%} ", key="cov95_conformal")
    rho = [spearmanr(np.abs(err[:, c]), sig_t[:, c])[0] for c, *_ in COMPS]
    row("Spearman(|error|, σ)",         rho, "{:>12.3f}", key="spearman_err_sigma")
    fits_ = [np.polyfit(y_t[:, c], m_t[:, c], 1) for c, *_ in COMPS]
    row("multiplicative bias m",        [f[0] - 1 for f in fits_], "{:>+12.4f}", key="bias_m")
    row("additive bias c",              [f[1] for f in fits_],     "{:>+12.5f}", key="bias_c")

    # Per-row predictions
    pred = df_test.reset_index(drop=True).copy()
    for c, tag, _, _ in COMPS:
        pred[f"{tag}_pred"]         = m_t[:, c]
        pred[f"{tag}_sigma"]        = sig_tc[:, c]      # calibrated total
        pred[f"{tag}_sigma_raw"]    = sig_t[:, c]
        pred[f"{tag}_sigma_alea"]   = np.sqrt(a_t[:, c])
        pred[f"{tag}_sigma_epi"]    = np.sqrt(e_t[:, c])
        pred[f"{tag}_conf68_halfwidth"] = sig_t[:, c] * q68[c]
        pred[f"{tag}_conf95_halfwidth"] = sig_t[:, c] * q95[c]
    pred.to_csv(out_dir / "predictions_test_uq.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "uq_summary_test.csv", index=False)

    plot_pred_vs_true_err(y_t, m_t, sig_tc, out_dir / "uq_pred_vs_true_test.png")
    plot_calibration(y_t, m_t, sig_t, sig_tc, out_dir / "uq_calibration_test.png")
    plot_zscores(y_t, m_t, sig_tc, out_dir / "uq_zscore_test.png")
    plot_rmse_vs_sigma(y_t, m_t, sig_tc, out_dir / "uq_rmse_vs_sigma_test.png")
    print(f"\nSaved predictions, summary and plots to {out_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="KL shear regressor with error bars.")
    # Same data / training options as train_kl_model.py
    p.add_argument("--images_root", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="./kl_model_output_uq")
    p.add_argument("--npix", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--freeze_stem", action="store_true")
    p.add_argument("--seed", type=int, default=42,
                   help="Member k uses seed + k")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for the train/val/test split (same for all members)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--use_original_image", action="store_true")
    p.add_argument("--min_vel_fill", type=float, default=0.5)
    p.add_argument("--crop_frac", type=float, default=0.75)
    p.add_argument("--smooth_sigma", type=float, default=0.0)
    p.add_argument("--smooth_target", choices=["both", "photo", "vel"], default="both")
    # Uncertainty options
    u = p.add_argument_group("uncertainty")
    u.add_argument("--n_members", type=int, default=5,
                   help="Deep-ensemble size (5 is the usual choice)")
    u.add_argument("--member", type=int, default=None,
                   help="Train only this member and exit (for SLURM arrays)")
    u.add_argument("--aggregate_only", action="store_true",
                   help="Skip training; evaluate the members in --outdir")
    u.add_argument("--beta", type=float, default=0.5,
                   help="β of the β-NLL loss (0 = plain NLL, 1 ≈ MSE-like)")
    u.add_argument("--mse_warmup", type=int, default=5,
                   help="Epochs fitting the mean with MSE before NLL")
    u.add_argument("--mc_samples", type=int, default=0,
                   help="MC-dropout passes per member at evaluation (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data and splits (deterministic, shared by all members) ──────────────
    df = pd.read_csv(args.csv)
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)
    images_root = Path(args.images_root)
    df = filter_to_existing(df, images_root, args.use_original_image)
    df = filter_by_vel_fill(df, images_root, args.min_vel_fill)
    print(f"Usable rows: {len(df)}")

    train_ds, val_ds, test_ds, df_train, df_val, df_test = build_splits(
        df, images_root, args.npix, args.use_original_image,
        smooth_sigma=args.smooth_sigma, smooth_target=args.smooth_target,
        crop_frac=args.crop_frac, seed=args.split_seed)
    for name, d in (("train", df_train), ("val", df_val), ("test", df_test)):
        f = out_dir / f"split_{name}.csv"
        if not f.exists():
            d.to_csv(f, index=False)

    # ── Train ───────────────────────────────────────────────────────────────
    if not args.aggregate_only:
        members = [args.member] if args.member is not None else range(args.n_members)
        for k in members:
            train_member(k, args, train_ds, val_ds, device, out_dir)
        if args.member is not None:
            print("\nSingle member trained; run with --aggregate_only once all "
                  "members are done.")
            return

    # ── Evaluate the ensemble ────────────────────────────────────────────────
    evaluate_ensemble(args, val_ds, test_ds, df_test, device, out_dir)


if __name__ == "__main__":
    main()