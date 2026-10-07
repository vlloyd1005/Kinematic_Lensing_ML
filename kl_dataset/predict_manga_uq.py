#!/usr/bin/env python3
"""
predict_manga_uq.py
===================
predict_manga.py for the uncertainty-aware deep ensemble trained by
train_kl_model_uq.py: every MaNGA prediction comes with an error bar.

Inputs are built exactly as in predict_manga.py (its functions are imported,
not copied), so the same --data_dir / --kl_dir folders, masks and PSF
handling apply.  For every galaxy:

  ensemble    each member k outputs a mean mu_k and variance sigma2_k per
              component (optionally x --mc_samples dropout passes), combined
              as in train_kl_model_uq.py:
                  mu        = mean_k mu_k
                  sigma2_al = mean_k sigma2_k       (aleatoric)
                  sigma2_ep = var_k  mu_k           (epistemic)
                  sigma     = s * sqrt(sigma2_al + sigma2_ep)
              with s the per-component calibration scale fitted on the TNG
              validation set (read from <ensemble>/uq_summary_test.csv).
              Normalised conformal 68% / 95% half-widths q*sigma_raw use the
              q recovered from <ensemble>/predictions_test_uq.csv.
  8-fold      the same on the 8 rotations / flips of the input, mapped back
              to the original frame (90 deg rotation flips both signs, a
              mirror flips g2; variances are unchanged).  The spread of the
              ensemble mean over the 8 orientations is a third, frame-
              dependent error term; *_dihedral_sigma includes it.

Calibration caveat: s and q were fitted on TNG.  Whether the error bars hold
for real data is tested here directly: the true cosmic shear on these
galaxies is ~0, so z = g / sigma should be ~N(0, 1).  The summary prints the
z statistics and the inverse-variance-weighted mean shear.

Frame: as predict_manga.py, the grid frame (x = West, y = North).

Outputs (in --outdir)
---------------------
  predictions_manga_uq.csv     one row per galaxy per mask (columns below)
  uq_null_test_<mask>.png      g ± sigma per galaxy and the z distribution
  inputs_real_vs_training_<mask>.png, saliency_manga_<mask>.png
                               as in predict_manga.py; saliency is of the
                               ensemble mean
  inputs_<mask>/               prepared model inputs (training layout)

Columns (per component c = g1, g2): <c>_pred, <c>_sigma (calibrated total),
<c>_sigma_raw, <c>_sigma_alea, <c>_sigma_epi, <c>_conf68_halfwidth,
<c>_conf95_halfwidth, <c>_z (= pred / sigma), <c>_dihedral_mean,
<c>_dihedral_std, <c>_dihedral_sigma, <c>_member<k> (each member's mean).

Usage
-----
  python predict_manga_uq.py --ensemble_dir kl_dataset/model_output_uq \
      --data_dir /content/Kinematic_Lensing_ML/sdss-manga/fits
  # --checkpoint also works: a member checkpoint or the ensemble folder
  # (so shear_test.py --predict_script predict_manga_uq.py runs it)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.table import Table

import predict_manga as pm                       # input building, plots
from train_kl_model import KLShearDataset
from train_kl_model_uq import TwoStreamShearNetUQ

COMPS = [(0, "g1", "g+"), (1, "g2", "g×")]
C1, C2 = "#2a78d6", "#eb6834"


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def find_members(path):
    """Member checkpoints from the ensemble folder, or from any one member's
    checkpoint / folder (then all members next to it are used)."""
    p = Path(path)
    if p.is_file():
        p = p.parent
    if p.name.startswith("member_"):
        p = p.parent
    ckpts = sorted(p.glob("member_*/best_model.pt"),
                   key=lambda c: int(c.parent.name.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"no member_*/best_model.pt under {p}")
    return p, ckpts


def load_calibration(ens_dir, use):
    """Calibration scale s and conformal multipliers q68, q95 per component,
    from train_kl_model_uq.py's outputs; 1 / NaN where unavailable."""
    cal = {"scale": np.ones(2), "q68": np.full(2, np.nan), "q95": np.full(2, np.nan)}
    if not use:
        print("Calibration: off (--no_calibration), sigma = raw ensemble sigma")
        return cal
    f = ens_dir / "uq_summary_test.csv"
    if f.exists():
        s = pd.read_csv(f).iloc[0]
        cal["scale"] = np.array([s.get("calib_scale_g1", 1.0), s.get("calib_scale_g2", 1.0)],
                                float)
    else:
        print(f"[WARN] {f} not found: no calibration scale (s = 1). "
              f"Run train_kl_model_uq.py --aggregate_only to make it.")
    f = ens_dir / "predictions_test_uq.csv"
    if f.exists():
        t = pd.read_csv(f)
        for c, tag, _ in COMPS:
            for q in ("68", "95"):
                col = f"{tag}_conf{q}_halfwidth"
                if col in t and f"{tag}_sigma_raw" in t:
                    cal[f"q{q}"][c] = float(np.median(t[col] / t[f"{tag}_sigma_raw"]))
    print(f"Calibration (fitted on TNG validation): s = {cal['scale'].round(3)}, "
          f"conformal q68 = {cal['q68'].round(3)}, q95 = {cal['q95'].round(3)}")
    return cal


def load_ensemble(ckpts, device):
    models, ta = [], None
    for c in ckpts:
        ck = torch.load(c, map_location=device, weights_only=False)
        m = TwoStreamShearNetUQ(pretrained=False).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
        if ta is None:
            ta = ck.get("args", {})
        elif ck.get("args", {}).get("npix", ta.get("npix")) != ta.get("npix"):
            raise SystemExit(f"{c}: npix differs between members")
    return models, ta


def _set_mc(model, on):
    model.eval()
    if on:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()


@torch.no_grad()
def ensemble_dihedral(models, photo, vel, device, mc_samples=0):
    """
    Means and variances for every orientation, member and MC sample:
    arrays (8, K*S, 2), already mapped back to the original frame.
    Orientation 0 is the untransformed input.
    """
    batch_p, batch_v, signs = [], [], []
    for k in range(4):
        for flip in (False, True):
            p = torch.rot90(photo, k, dims=(-2, -1))
            v = torch.rot90(vel, k, dims=(-2, -1))
            if flip:
                p, v = torch.flip(p, dims=(-1,)), torch.flip(v, dims=(-1,))
            batch_p.append(p); batch_v.append(v)
            s = (-1) ** k
            signs.append([s, s * (-1 if flip else 1)])
    P = torch.stack(batch_p).contiguous().to(device)
    V = torch.stack(batch_v).contiguous().to(device)
    signs = np.array(signs, float)                       # (8, 2)
    S = max(1, mc_samples)
    mus, vars_ = [], []
    for m in models:
        _set_mc(m, mc_samples > 0)
        for _ in range(S):
            mu, var = m(P, V)
            mus.append(mu.cpu().numpy() * signs)
            vars_.append(var.cpu().numpy())
        _set_mc(m, False)
    return np.stack(mus, 1), np.stack(vars_, 1)         # (8, K*S, 2)


class EnsembleMean(nn.Module):
    """Ensemble-mean prediction as (g1, g2): lets predict_manga.smoothgrad
    compute saliency of the ensemble."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, photo, vel):
        mu = torch.stack([m(photo, vel)[0] for m in self.models]).mean(0)
        return mu[:, 0], mu[:, 1]


def summarise_galaxy(mus, vars_, cal, n_members):
    """Columns for one galaxy from ensemble_dihedral output."""
    row = {}
    mu0, var0 = mus[0], vars_[0]                         # (K*S, 2), as given
    mean = mu0.mean(0); alea = var0.mean(0); epi = mu0.var(0)
    raw = np.sqrt(alea + epi)
    ens_by_orient = mus.mean(1)                          # (8, 2)
    d_alea = vars_.mean((0, 1)); d_epi = mus.reshape(-1, 2).var(0)
    S = mus.shape[1] // n_members
    for c, tag, _ in COMPS:
        row[f"{tag}_pred"]       = mean[c]
        row[f"{tag}_sigma"]      = raw[c] * cal["scale"][c]
        row[f"{tag}_sigma_raw"]  = raw[c]
        row[f"{tag}_sigma_alea"] = np.sqrt(alea[c])
        row[f"{tag}_sigma_epi"]  = np.sqrt(epi[c])
        row[f"{tag}_conf68_halfwidth"] = raw[c] * cal["q68"][c]
        row[f"{tag}_conf95_halfwidth"] = raw[c] * cal["q95"][c]
        row[f"{tag}_z"]          = mean[c] / row[f"{tag}_sigma"]
        row[f"{tag}_dihedral_mean"]  = ens_by_orient[:, c].mean()
        row[f"{tag}_dihedral_std"]   = ens_by_orient[:, c].std()
        # all 8 x K x S predictions pooled: orientation spread counts as error
        row[f"{tag}_dihedral_sigma"] = np.sqrt(d_alea[c] + d_epi[c]) * cal["scale"][c]
        for k in range(n_members):
            row[f"{tag}_member{k}"] = mu0[k * S:(k + 1) * S, c].mean()
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Per-mask run
# ═══════════════════════════════════════════════════════════════════════════════

def run_mode_uq(mode, items, drp, cfg, ta, min_fill, models, cal, device, out_dir, a):
    cfg  = {**cfg, "vel_mask": mode}
    root = out_dir / f"inputs_{mode}"
    rows, n_low = [], 0
    for k, it in enumerate(items, 1):
        try:
            if it["source"] == "dap_fits":
                src = pm.load_dap_pair(it["img"], it["maps"], drp, cfg)
            elif it["source"] == "grid":
                src = pm.load_grid(it["img"], it["maps"], cfg)
            else:
                src = pm.load_converted(it["path"], cfg)
            photo, vel, wcs, info = pm.build_galaxy(src, cfg)
        except Exception as exc:
            print(f"  [ERROR] {it['mangaid']} ({it['source']}): {exc}")
            continue
        info.update(mangaid=it["mangaid"], source=it["source"])
        sid = len(rows)
        pm.write_inputs(root, sid, photo, vel, wcs, info)
        n_low += info["vel_fill_frac"] < min_fill
        if not a.verbose and (k % 100 == 0 or k == len(items)):
            print(f"  built {k}/{len(items)}")
        rows.append({"subhalo_id": sid, "snap": 0, "draw_idx": 0,
                     "g1": 0.0, "g2": 0.0, **info})
    if not rows:
        return None, []
    df = pd.DataFrame(rows)
    if n_low:
        print(f"  [WARN] {n_low}/{len(df)} galaxies have velmap fill below the "
              f"training cut ({min_fill}); see column vel_fill_frac.")

    smooth_in_ds = cfg["smooth_sigma"] if a.psf_mode == "train" else 0.0
    ds = KLShearDataset(df, root, cfg["npix"], ta.get("use_original_image", False),
                        smooth_sigma=smooth_in_ds,
                        smooth_target=cfg["smooth_target"], crop_frac=cfg["crop"])

    rng = np.random.default_rng(a.seed)
    n_sal = len(df) if a.n_saliency < 0 else min(a.n_saliency, len(df))
    sal_idx = set(rng.choice(len(df), size=n_sal, replace=False).tolist())
    ens_mean = EnsembleMean(models).eval()

    print(f"\n── Predictions [{mode}], {len(models)}-member ensemble "
          f"(grid frame: x = West, y = North) ──")
    res_rows, results = [], []
    for i in range(len(ds)):
        photo, vel, _ = ds[i]
        mus, vars_ = ensemble_dihedral(models, photo, vel, device, a.mc_samples)
        r = summarise_galaxy(mus, vars_, cal, len(models))
        if i in sal_idx:
            sal = pm.smoothgrad(ens_mean, photo, vel, device,
                                a.smoothgrad_samples, a.smoothgrad_noise)
            results.append({"mangaid": df.loc[i, "mangaid"], "source": df.loc[i, "source"],
                            "g1": r["g1_pred"], "g2": r["g2_pred"],
                            "photo": photo[0].numpy(), "vel": vel.numpy(), "sal": sal})
        res_rows.append(r)
        if a.verbose:
            print(f"  {df.loc[i, 'mangaid']:<10}  "
                  f"g+ = {r['g1_pred']:+.4f} ± {r['g1_sigma']:.4f}   "
                  f"g× = {r['g2_pred']:+.4f} ± {r['g2_sigma']:.4f}   "
                  f"(epistemic share g+ "
                  f"{r['g1_sigma_epi']**2 / max(r['g1_sigma_raw']**2, 1e-30):.0%})")
        elif (i + 1) % 100 == 0 or i + 1 == len(ds):
            print(f"  predicted {i + 1}/{len(ds)}")

    df = pd.concat([df, pd.DataFrame(res_rows)], axis=1)
    df = df.drop(columns=["g1", "g2", "snap", "draw_idx"]).rename(
        columns={"subhalo_id": "index"})
    return df, results


# ═══════════════════════════════════════════════════════════════════════════════
# Null test: real shear ~ 0, so z = g / sigma should be ~N(0, 1)
# ═══════════════════════════════════════════════════════════════════════════════

def null_test(df, mode, min_fill, out_dir, plot=True):
    from scipy.stats import norm
    out = {}
    sub = df[df["vel_fill_frac"] >= min_fill]
    print(f"\n── Null test [{mode}], {len(sub)} galaxies with fill ≥ {min_fill} "
          f"(true shear ≈ 0, so z = g/σ should be ~N(0, 1)) ──")
    if len(sub) < 3:
        print("  too few galaxies")
        return out
    print(f"  {'':34}{'g+':>12}{'g×':>12}")
    def line(label, vals, fmt="{:>12.4f}"):
        print(f"  {label:<34}" + "".join(fmt.format(v) for v in vals))
    g = sub[["g1_pred", "g2_pred"]].to_numpy()
    s = sub[["g1_sigma", "g2_sigma"]].to_numpy()
    z = g / s
    w = 1 / s**2
    wmean = (w * g).sum(0) / w.sum(0); werr = 1 / np.sqrt(w.sum(0))
    line("median predicted σ", np.median(s, 0))
    line("rms of predictions", np.sqrt(np.mean(g**2, 0)))
    line("rms(z)   (1 if σ is right)", np.sqrt(np.mean(z**2, 0)), "{:>12.2f}")
    line("|z| < 1 fraction  (68.3%)", np.mean(np.abs(z) < 1, 0), "{:>11.1%} ")
    line("|z| < 2 fraction  (95.4%)", np.mean(np.abs(z) < 2, 0), "{:>11.1%} ")
    line("χ²/N vs zero", np.mean(z**2, 0), "{:>12.2f}")
    line("weighted mean shear", wmean, "{:>+12.4f}")
    line("  ± error", werr)
    line("  significance (σ)", wmean / werr, "{:>+12.1f}")
    for c, tag, _ in COMPS:
        out.update({f"rms_z_{tag}": float(np.sqrt(np.mean(z[:, c]**2))),
                    f"wmean_{tag}": float(wmean[c]), f"wmean_err_{tag}": float(werr[c])})
    print("  rms(z) > 1: the TNG-calibrated σ is too small for real data "
          "(domain shift); < 1: too large.")
    if not plot:
        return out
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    order = np.argsort(sub["g1_sigma"].to_numpy())
    x = np.arange(len(sub))
    for c, (tag, sym), col in zip(range(2), [("g1", "g+"), ("g2", "g×")], (C1, C2)):
        ax[c].errorbar(x, g[order, c], yerr=s[order, c], fmt="o", ms=3, color=col,
                       elinewidth=0.6, alpha=0.6)
        ax[c].axhline(0, color="k", lw=0.8)
        ax[c].set_xlabel("galaxy (sorted by σ of g+)"); ax[c].set_ylabel(f"{sym} ± σ")
        ax[c].set_title(f"{sym}: weighted mean {wmean[c]:+.4f} ± {werr[c]:.4f}", fontsize=10)
    xs = np.linspace(-5, 5, 200); bins = np.linspace(-5, 5, 41)
    for c, (sym, col) in enumerate([("g+", C1), ("g×", C2)]):
        ax[2].hist(np.clip(z[:, c], -5, 5), bins=bins, density=True, histtype="step",
                   lw=2, color=col, label=f"{sym}: rms {np.sqrt(np.mean(z[:, c]**2)):.2f}")
    ax[2].plot(xs, norm.pdf(xs), "k--", lw=1, label="N(0, 1)")
    ax[2].set_xlabel("z = g / σ"); ax[2].set_ylabel("density")
    ax[2].set_title("standardised predictions (should match N(0, 1))", fontsize=10)
    ax[2].legend(fontsize=8)
    for axx in ax:
        axx.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"uq_null_test_{mode}.png", dpi=140); plt.close(fig)
    print(f"  Saved: {out_dir / f'uq_null_test_{mode}.png'}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    # predict_manga's options, with the ensemble replacing the checkpoint
    argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ensemble_dir", "--checkpoint", dest="ensemble_dir", required=True,
                   help="model_output_uq folder (with member_*/), or any member's "
                        "best_model.pt; all members found are used")
    p.add_argument("--mc_samples", type=int, default=0,
                   help="MC-dropout passes per member (0 = off, as in training's default)")
    p.add_argument("--no_calibration", action="store_true",
                   help="report raw ensemble σ (no TNG calibration scale)")
    p.add_argument("--outdir", default="/content/manga_predictions_uq")
    p.add_argument("--n_saliency", type=int, default=12)
    # everything else as in predict_manga.py
    for flag, kw in [
        ("--data_dir", dict(default=None)), ("--kl_dir", dict(default=None)),
        ("--max_galaxies", dict(type=int, default=0)),
        ("--drpall", dict(default="/content/drpall-v3_1_1.fits")),
        ("--fov_kpc", dict(type=float, default=30.0)),
        ("--line", dict(default="Ha-6564")),
        ("--min_ha_snr", dict(type=float, default=3.0)),
        ("--vel_masks", dict(nargs="+", default=["none", "strict"],
                             choices=["none", "dap", "strict"])),
        ("--ha_anr_cut", dict(type=float, default=5.0)),
        ("--circular_ifu", dict(action="store_true")),
        ("--clean_nsigma", dict(type=float, default=2.0)),
        ("--sdss_fwhm", dict(type=float, default=1.4)),
        ("--manga_fwhm", dict(type=float, default=2.5)),
        ("--psf_mode", dict(choices=["match", "train", "none"], default="match")),
        ("--smoothgrad_samples", dict(type=int, default=16)),
        ("--smoothgrad_noise", dict(type=float, default=0.05)),
        ("--n_train_compare", dict(type=int, default=3)),
        ("--seed", dict(type=int, default=0)),
        ("--verbose", dict(action="store_true"))]:
        p.add_argument(flag, **kw)
    return p.parse_args(argv)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    out_dir = Path(a.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ens_dir, ckpts = find_members(a.ensemble_dir)
    models, ta = load_ensemble(ckpts, device)
    cal = load_calibration(ens_dir, not a.no_calibration)
    cfg = dict(npix=ta.get("npix", 128), crop=ta.get("crop_frac", 1.0),
               smooth_sigma=ta.get("smooth_sigma", 0.0),
               smooth_target=ta.get("smooth_target", "both"),
               fov_kpc=a.fov_kpc, line=a.line, min_ha_snr=a.min_ha_snr,
               ha_anr_cut=a.ha_anr_cut, clean_nsigma=a.clean_nsigma,
               circular_ifu=a.circular_ifu, sdss_fwhm=a.sdss_fwhm,
               manga_fwhm=a.manga_fwhm, psf_mode=a.psf_mode)
    min_fill = ta.get("min_vel_fill", 0.0)
    print(f"Ensemble: {ens_dir}  ({len(models)} members: "
          f"{', '.join(c.parent.name for c in ckpts)}"
          f"{f', × {a.mc_samples} MC-dropout passes' if a.mc_samples else ''})")
    print(f"  npix={cfg['npix']}  crop_frac={cfg['crop']}  "
          f"smooth_sigma={cfg['smooth_sigma']} ({cfg['smooth_target']})  "
          f"min_vel_fill={min_fill}  fov={a.fov_kpc} kpc  psf_mode={a.psf_mode}")

    items = pm.find_inputs(a)
    if not items:
        raise SystemExit("No galaxies found: give --data_dir and/or --kl_dir.")
    drp = {}
    if any(it["source"] == "dap_fits" for it in items):
        t = Table.read(a.drpall, hdu=1)
        drp = {str(p).strip(): r for p, r in zip(t["plateifu"], t)}
    print(f"\n{len(items)} galaxies: "
          + ", ".join(f"{k} {v}" for k, v in
                      pd.Series([it['source'] for it in items]).value_counts().items()))

    train_rows = pm.training_examples(ens_dir, ta, a.n_train_compare, a.seed)
    all_df, null = [], []
    for mode in a.vel_masks:
        print(f"\n══ Velocity mask: {mode} " + "═" * 50)
        df, results = run_mode_uq(mode, items, drp, cfg, ta, min_fill, models, cal,
                                  device, out_dir, a)
        if df is None:
            continue
        all_df.append(df)
        null.append({"vel_mask": mode, **null_test(df, mode, min_fill, out_dir)})
        if results:
            real = [(f"{r['mangaid']} ({r['source']}, {mode})", r["photo"], r["vel"])
                    for r in results]
            pm.plot_inputs(real + train_rows,
                           out_dir / f"inputs_real_vs_training_{mode}.png")
            pm.plot_saliency(results, out_dir / f"saliency_manga_{mode}.png")

    if not all_df:
        raise SystemExit("No galaxies could be built.")
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(out_dir / "predictions_manga_uq.csv", index=False)
    # same file name as predict_manga.py, so shear_test.py can read it
    out.to_csv(out_dir / "predictions_manga.csv", index=False)
    pd.DataFrame(null).to_csv(out_dir / "uq_null_test_summary.csv", index=False)
    print(f"\n  Saved: {out_dir / 'predictions_manga_uq.csv'} "
          f"(also as predictions_manga.csv), uq_null_test_summary.csv")


if __name__ == "__main__":
    main()
