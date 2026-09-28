#!/usr/bin/env python3
"""
evaluate_kl_model.py
====================
Load a trained TwoStreamShearNet checkpoint and run evaluations without
retraining.

By default it evaluates on the held-out test split that train_kl_model.py
saved next to the checkpoint (split_test.csv), using the same npix /
images_root / use_original_image settings stored in the checkpoint.

Evaluations (run in order, see EVALUATIONS)
-------------------------------------------
  1. Basic metrics (loss, MSE, RMSE, eval time)
  2. Absolute error percentiles
  3. Fractional error percentiles
  4. Diagnostic plots (scatter, pred vs true, residual histograms)
  5. Error analysis: relates per-galaxy errors to galaxy properties
       - from the TNG API (cached): log M*, log SFR, log sSFR, gas fraction,
         stellar half-mass radius, v_max, metallicity, N gas particles
       - from the FITS files: rendered inclination and θ_int (FITS header),
         velmap fill fraction, |v|_95, stellar half-light radius
     and writes, per component, to <outdir>/error_analysis_<g1|g2>/:
       pred_vs_true_colored_<c>.png, abs_err_vs_galaxy_props_<c>.png,
       abs_err_vs_row_props_<c>.png, gallery_worst_best_<c>.png,
       correlations_<c>.csv, galaxy_errors_<c>.csv, rows_with_props.csv

NOTE on inclination: rows added by expand_shear_draws.py get new random
`inclination` / `theta_int` values in the CSV, but the images were rendered
once with the ORIGINAL values.  The error analysis therefore reads the
rendered values from the FITS header instead of the CSV.

Adding a new evaluation
-----------------------
Write a function with the signature

    def eval_something(preds, labels, ctx) -> dict:

where preds / labels are (N, 2) arrays [g+, g×] and ctx is a dict holding
"df", "out_dir", "name", "loss", "avg_eval_time", "images_root",
"tng_cache" and "args".  Print whatever you like, and return a dict of
scalars to be added to the summary CSV.  Then append it to EVALUATIONS.

Usage
-----
  # Test split saved during training (TNG properties need an API key)
  export TNG_API_KEY=...
  python evaluate_kl_model.py --checkpoint ./kl_model_output/best_model.pt

  # Any other CSV with subhalo_id, snap, g1, g2 columns
  python evaluate_kl_model.py --checkpoint ./kl_model_output/best_model.pt \
      --split_csv ./kl_model_output/split_val.csv --name val

  # Error analysis for both components, FITS-derived properties only
  python evaluate_kl_model.py --checkpoint ... --error_components g1 g2 --no_tng
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.io import fits
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from train_kl_model import (
    KLShearDataset,
    TwoStreamShearNet,
    filter_to_existing,
    plot_pred_vs_true,
    plot_residual_hist,
    plot_shear_scatter,
    run_epoch,
)

EPS  = 0.035               # KL shape noise floor (Xu+2022 §3.3)
PCTS = [50, 75, 90, 95, 99]
COMPONENTS = [(0, "g+", "g1"), (1, "g×", "g2")]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluations  (each returns a dict of scalars for the summary CSV)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_basic_metrics(preds, labels, ctx):
    out = {
        "n_samples":        len(labels),
        "n_unique_gal":     int(ctx["df"]["subhalo_id"].nunique()),
        "smooth_l1":        float(ctx["loss"]),
        "avg_eval_time_ms": float(ctx["avg_eval_time"] * 1e3),
    }
    print(f"  Samples             : {out['n_samples']} "
          f"({out['n_unique_gal']} unique galaxies)")
    print(f"  Smooth-L1 loss      : {out['smooth_l1']:.4f}")
    print(f"  Avg eval time       : {out['avg_eval_time_ms']:.3f} ms/sample")
    for col, sym, tag in COMPONENTS:
        mse = float(np.mean((preds[:, col] - labels[:, col]) ** 2))
        out[f"mse_{tag}"]  = mse
        out[f"rmse_{tag}"] = float(np.sqrt(mse))
        print(f"  MSE  {sym:<3}            : {mse:.4e}")
        print(f"  RMSE {sym:<3}            : {np.sqrt(mse):.4e}")
    return out


def eval_abs_error_percentiles(preds, labels, ctx):
    out = {}
    errs = {tag: np.abs(preds[:, c] - labels[:, c]) for c, _, tag in COMPONENTS}

    print("\n  Absolute error percentiles  |pred − true|")
    print(f"  {'Percentile':>12}  {'g+':>10}  {'g×':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for p in PCTS:
        g1p = float(np.percentile(errs["g1"], p))
        g2p = float(np.percentile(errs["g2"], p))
        out[f"abs_err_pct{p:02d}_g1"] = g1p
        out[f"abs_err_pct{p:02d}_g2"] = g2p
        print(f"  {p:>11}th  {g1p:>10.4f}  {g2p:>10.4f}")
    return out


def eval_frac_error_percentiles(preds, labels, ctx):
    out, fracs = {}, {}
    for col, _, tag in COMPONENTS:
        true, pred = labels[:, col], preds[:, col]
        valid = np.abs(true) >= EPS
        fracs[tag] = np.abs((pred[valid] - true[valid]) / np.abs(true[valid]))
        out[f"frac_err_n_excluded_{tag}"] = int((~valid).sum())

    print(f"\n  Fractional error percentiles  |pred − true| / |true|")
    print(f"  (only galaxies with |g_true| ≥ {EPS})")
    print(f"  excluded: g+ {out['frac_err_n_excluded_g1']}, "
          f"g× {out['frac_err_n_excluded_g2']}")
    print(f"  {'Percentile':>12}  {'g+':>10}  {'g×':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")
    for p in PCTS:
        vals = []
        for tag in ("g1", "g2"):
            v = float(np.percentile(fracs[tag], p)) if fracs[tag].size else np.nan
            out[f"frac_err_pct{p:02d}_{tag}"] = v
            vals.append(v)
        print(f"  {p:>11}th  {vals[0]*100:>9.1f}%  {vals[1]*100:>9.1f}%")
    return out


def eval_plots(preds, labels, ctx):
    if ctx["args"].skip_plots:
        return {}
    print("\n  Generating diagnostic plots …")
    name, out_dir = ctx["name"], ctx["out_dir"]
    plot_shear_scatter(preds, labels, name, out_dir)
    plot_pred_vs_true( preds, labels, name, out_dir)
    plot_residual_hist(preds, labels, name, out_dir, eps=EPS)
    return {}


def eval_error_analysis(preds, labels, ctx):
    """Relate per-galaxy errors to TNG / FITS galaxy properties."""
    args = ctx["args"]
    if args.skip_error_analysis or not args.error_components:
        return {}
    print("\n── Error analysis ─────────────────────────────────────────")
    df = build_property_table(ctx["df"], preds, ctx["images_root"],
                              args, ctx["tng_cache"])
    df.to_csv(ctx["out_dir"] / "rows_with_props.csv", index=False)

    out = {}
    for comp in args.error_components:
        corr = analyze_component(df, comp, ctx["images_root"],
                                 ctx["out_dir"] / f"error_analysis_{comp}", args)
        if corr is not None and len(corr):
            out[f"err_top_property_{comp}"] = corr.iloc[0]["property"]
            out[f"err_top_rho_{comp}"]      = float(corr.iloc[0]["spearman_rho"])
    return out


# Add new evaluation functions here; they run in order.
EVALUATIONS = [
    eval_basic_metrics,
    eval_abs_error_percentiles,
    eval_frac_error_percentiles,
    eval_plots,
    eval_error_analysis,
]


# ═══════════════════════════════════════════════════════════════════════════════
# Error analysis: galaxy properties
# ═══════════════════════════════════════════════════════════════════════════════

H_LITTLE = 0.6774          # TNG little h

# Full TNG50 snapshots -> redshift (used if the CSV has no redshift column)
SNAP_REDSHIFT = {33: 2.0, 40: 1.5, 50: 1.0, 59: 0.7, 67: 0.5,
                 72: 0.4, 78: 0.3, 84: 0.2, 91: 0.1, 99: 0.0}

ERR_COMPONENTS = {"g1": ("g+", "g1", "g1_pred"),
                  "g2": ("g×", "g2", "g2_pred")}

TNG_FIELDS = ["mass_stars", "mass_gas", "sfr", "halfmassrad_stars",
              "vmax", "starmetallicity", "gasmetallicity",
              "len_stars", "len_gas"]

KEYS = ["sim", "snap", "subhalo_id"]

# (column, plot label); anything missing from the data is skipped
GALAXY_PROPS = [
    ("log_mstar",           "log M* [M☉]"),
    ("log_sfr",             "log SFR [M☉/yr]"),
    ("log_ssfr",            "log sSFR [1/yr]"),
    ("gas_frac",            "M_gas / (M_gas + M*)"),
    ("r_half_star_kpc",     "r½ stars [kpc]"),
    ("tng_vmax",            "v_max [km/s]"),
    ("tng_starmetallicity", "Z*"),
    ("log_len_gas",         "log N gas particles"),
    ("redshift",            "z"),
    ("incl_img_deg",        "inclination (rendered) [deg]"),
    ("sin2theta",           "sin 2θ_int (rendered)"),
    ("cos2theta",           "cos 2θ_int (rendered)"),
    ("vel_fill_frac",       "velmap fill fraction"),
    ("v_amp_kms",           "|v| 95th pct [km/s]"),
    ("r50_light_kpc",       "r50 light [kpc]"),
]


def fits_features(images_root: Path, snap: int, sid: int) -> dict:
    """Cheap per-galaxy features read from the FITS files on disk."""
    d   = images_root / f"snap{snap}" / f"galaxy_{sid}"
    out = {}
    pixscale = np.nan

    vel_path = d / f"galaxy_{sid}_velmap_original.fits"
    if vel_path.exists():
        with fits.open(str(vel_path)) as hdul:
            v   = hdul[0].data.astype(np.float64)
            hdr = hdul[0].header
        out["incl_img"]  = hdr.get("INCL_RAD",  np.nan)
        out["theta_img"] = hdr.get("THETA_INT", np.nan)
        pixscale         = hdr.get("PIXSCALE",  np.nan)
        finite = np.isfinite(v)
        out["vel_fill_frac"] = float(finite.mean())
        out["v_amp_kms"] = (float(np.percentile(np.abs(v[finite]), 95))
                            if finite.any() else np.nan)

    img_path = d / f"galaxy_{sid}_image_original.fits"
    if img_path.exists():
        with fits.open(str(img_path)) as hdul:
            img = np.nan_to_num(hdul[0].data.astype(np.float64))
            if not np.isfinite(pixscale):
                pixscale = hdul[0].header.get("PIXSCALE", np.nan)
        c      = (img.shape[0] - 1) / 2.0
        yy, xx = np.indices(img.shape)
        r      = np.hypot(xx - c, yy - c).ravel()
        order  = np.argsort(r)
        cum    = np.cumsum(img.ravel()[order])
        if cum[-1] > 0:
            r50 = r[order][np.searchsorted(cum, 0.5 * cum[-1])]
            out["r50_light_kpc"] = float(r50 * pixscale)

    return out


def fetch_tng_props(gals: pd.DataFrame, api_key: str,
                    cache_path: Path) -> pd.DataFrame:
    """Fetch subhalo catalogue entries, caching results to CSV."""
    # Imported here so evaluation works without requests/h5py when --no_tng
    from generate_kl_tng50 import BASE_URL, tng_get

    cols  = KEYS + [f"tng_{f}" for f in TNG_FIELDS]
    cache = (pd.read_csv(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=cols))
    cache = cache.astype({"sim": str, "snap": int, "subhalo_id": int})
    done  = set(zip(cache["sim"], cache["snap"], cache["subhalo_id"]))

    todo = [g for g in gals.itertuples(index=False)
            if (g.sim, g.snap, g.subhalo_id) not in done]
    print(f"  TNG properties: {len(done)} cached, {len(todo)} to fetch "
          f"(cache: {cache_path})")

    api_key = api_key.strip().strip('"').strip("'")
    headers, new_rows = {"api-key": api_key}, []

    # Preflight: test the key once on the first galaxy instead of letting
    # tng_get retry every galaxy against an auth error.
    if todo:
        import requests
        g   = todo[0]
        url = f"{BASE_URL}{g.sim}/snapshots/{g.snap}/subhalos/{g.subhalo_id}/"
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            print(f"  [ERROR] Cannot reach the TNG API ({exc}). "
                  f"Skipping TNG properties.")
            return cache
        if r.status_code in (401, 403):
            masked = f"{api_key[:4]}…({len(api_key)} chars)"
            print(f"  [ERROR] TNG API rejected the key {masked} "
                  f"with HTTP {r.status_code}: {r.text[:200].strip()}")
            print("          Check the key on your TNG account page and pass it "
                  "with --apikey. Skipping TNG properties.")
            return cache

    for i, g in enumerate(todo, 1):
        url = f"{BASE_URL}{g.sim}/snapshots/{g.snap}/subhalos/{g.subhalo_id}/"
        try:
            meta = tng_get(url, headers=headers).json()
        except Exception as exc:
            print(f"  [WARN] {g.sim} snap {g.snap} id {g.subhalo_id}: {exc}")
            continue
        row = {"sim": g.sim, "snap": g.snap, "subhalo_id": g.subhalo_id}
        for f in TNG_FIELDS:
            row[f"tng_{f}"] = meta.get(f, np.nan)
        new_rows.append(row)

        if i % 25 == 0 or i == len(todo):
            print(f"    fetched {i}/{len(todo)}")
            pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True
                      ).to_csv(cache_path, index=False)
        time.sleep(0.2)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return cache


def add_derived(df: pd.DataFrame) -> None:
    """Physical-unit and derived columns (in place)."""
    # TNG columns can arrive as object dtype (concat onto an empty cache
    # frame), which numpy ufuncs like log10 reject, so force numeric.
    for c in [c for c in df.columns if c.startswith("tng_")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        if "tng_mass_stars" in df:
            a     = 1.0 / (1.0 + df["redshift"])
            mstar = df["tng_mass_stars"] * 1e10 / H_LITTLE
            mgas  = df["tng_mass_gas"]   * 1e10 / H_LITTLE
            sfr   = df["tng_sfr"].clip(lower=1e-3)   # floor for quenched
            df["log_mstar"]       = np.log10(mstar.where(mstar > 0))
            df["log_sfr"]         = np.log10(sfr)
            df["log_ssfr"]        = np.log10(sfr / mstar.where(mstar > 0))
            df["gas_frac"]        = mgas / (mgas + mstar)
            df["r_half_star_kpc"] = df["tng_halfmassrad_stars"] * a / H_LITTLE
            df["log_len_gas"]     = np.log10(df["tng_len_gas"].clip(lower=1))
        if "incl_img" in df:
            df["incl_img_deg"] = np.degrees(df["incl_img"])
        if "theta_img" in df:
            df["sin2theta"] = np.sin(2 * df["theta_img"])
            df["cos2theta"] = np.cos(2 * df["theta_img"])


def build_property_table(df_split, preds, images_root, args, cache_path):
    """Per-row predictions joined with FITS and (optionally) TNG properties."""
    df = df_split.copy()
    df["g1_pred"] = preds[:, 0]
    df["g2_pred"] = preds[:, 1]
    if "sim" not in df:
        df["sim"] = args.sim
    df = df.astype({"sim": str, "snap": int, "subhalo_id": int})
    if "redshift" not in df:
        df["redshift"] = df["snap"].map(SNAP_REDSHIFT)

    gals = df[KEYS].drop_duplicates().reset_index(drop=True)

    print(f"  Reading FITS-derived properties for {len(gals)} galaxies …")
    feats = [{"sim": g.sim, "snap": g.snap, "subhalo_id": g.subhalo_id,
              **fits_features(images_root, g.snap, g.subhalo_id)}
             for g in gals.itertuples(index=False)]
    df = df.merge(pd.DataFrame(feats), on=KEYS, how="left")

    if "inclination" in df and "incl_img" in df:
        mismatch = (df["inclination"] - df["incl_img"]).abs() > 1e-4
        if mismatch.any():
            print(f"  [NOTE] {int(mismatch.sum())}/{len(df)} rows have a CSV "
                  f"inclination that differs from the rendered image "
                  f"(expected for expand_shear_draws rows). Using FITS values.")

    api_key = args.apikey or os.environ.get("TNG_API_KEY")
    if args.no_tng:
        print("  Skipping TNG API (--no_tng).")
    elif not api_key:
        print("  No TNG API key (--apikey / TNG_API_KEY); "
              "using FITS properties only.")
    else:
        props = fetch_tng_props(gals, api_key, cache_path)
        df = df.merge(props, on=KEYS, how="left")

    add_derived(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Error analysis: stats + plots
# ═══════════════════════════════════════════════════════════════════════════════

def correlations(table, ycol, props, level):
    rows = []
    for c, label in props:
        ok = table[c].notna() & table[ycol].notna()
        if ok.sum() < 5:
            continue
        rho, p = spearmanr(table.loc[ok, c], table.loc[ok, ycol])
        rows.append({"property": c, "label": label, "level": level,
                     "n": int(ok.sum()), "spearman_rho": rho, "p_value": p})
    return rows


def binned_median(x, y, nbins=8):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    nbins = max(2, min(nbins, len(x) // 5))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return [], []
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    bins = [i for i in range(len(edges) - 1) if (idx == i).any()]
    return ([np.median(x[idx == i]) for i in bins],
            [np.median(y[idx == i]) for i in bins])


def _grid(n, w=4.2, hgt=3.8):
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, hgt * nrow),
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def plot_colored_pred_vs_true(df, props, true_col, pred_col, sym, out_path):
    fig, axes = _grid(len(props))
    t, p = df[true_col].to_numpy(), df[pred_col].to_numpy()
    lim  = max(np.abs(t).max(), np.abs(p).max()) * 1.1
    for ax, (c, label) in zip(axes, props):
        vals = df[c].to_numpy(dtype=float)
        ok   = np.isfinite(vals)
        ax.scatter(t[~ok], p[~ok], s=5, c="lightgrey", rasterized=True)
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        sc = ax.scatter(t[ok], p[ok], c=vals[ok], cmap="viridis",
                        vmin=vmin, vmax=vmax, s=6, alpha=0.7, rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(label, fontsize=8)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{sym} true"); ax.set_ylabel(f"{sym} pred")
        ax.set_title(label, fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_err_vs_props(table, ycol, ylabel, props, corr, out_path):
    lookup = {r["property"]: r for r in corr}
    fig, axes = _grid(len(props))
    for ax, (c, label) in zip(axes, props):
        x = table[c].to_numpy(dtype=float)
        y = table[ycol].to_numpy(dtype=float)
        ax.scatter(x, y, s=8, alpha=0.5, color="#2E86AB", rasterized=True)
        xc, yc = binned_median(x, y)
        ax.plot(xc, yc, "-o", color="#E84855", ms=3, lw=1.5)
        ax.set_yscale("log")
        r = lookup.get(c)
        stat = f"\nρ={r['spearman_rho']:+.2f}  p={r['p_value']:.1e}" if r else ""
        ax.set_title(label + stat, fontsize=9)
        ax.set_xlabel(label, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_gallery(gal, images_root, n, sym, out_path):
    n = min(n, len(gal) // 2)
    if n < 1:
        return
    groups = [(gal.nlargest(n, "med_abs_err"), "worst"),
              (gal.nsmallest(n, "med_abs_err"), "best")]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for col0, (sub, tag) in zip([0, 2], groups):
        for r, (_, g) in enumerate(sub.iterrows()):
            sid, snap = int(g["subhalo_id"]), int(g["snap"])
            d = images_root / f"snap{snap}" / f"galaxy_{sid}"
            ax_i, ax_v = axes[r, col0], axes[r, col0 + 1]
            try:
                img = np.nan_to_num(fits.getdata(d / f"galaxy_{sid}_image_original.fits"))
                floor = 1e-3 * img.max() if img.max() > 0 else 1.0
                ax_i.imshow(np.log10(img + floor), origin="lower", cmap="gray")
                v  = fits.getdata(d / f"galaxy_{sid}_velmap_original.fits")
                vm = np.nanpercentile(np.abs(v), 95) if np.isfinite(v).any() else 1.0
                ax_v.imshow(v, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm)
            except FileNotFoundError:
                pass
            ax_i.set_title(f"{tag} #{r+1}  id {sid} (snap {snap})\n"
                           f"median |Δ{sym}| = {g['med_abs_err']:.3f}", fontsize=8)
            extra = []
            for col, fmt in [("incl_img_deg", "i={:.0f}°"),
                             ("log_mstar",    "logM*={:.1f}"),
                             ("vel_fill_frac", "fill={:.2f}")]:
                if col in g and np.isfinite(g[col]):
                    extra.append(fmt.format(g[col]))
            ax_v.set_title("velmap  " + "  ".join(extra), fontsize=8)
            for ax in (ax_i, ax_v):
                ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def analyze_component(df, comp, images_root, out_dir, args):
    """Correlations, tables and plots for one shear component."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sym, true_col, pred_col = ERR_COMPONENTS[comp]
    other = "g1" if comp == "g2" else "g2"
    other_sym = ERR_COMPONENTS[other][0]

    df = df.copy()
    df["err"]     = df[pred_col] - df[true_col]
    df["abs_err"] = df["err"].abs()
    df["g_abs_true"]        = np.hypot(df["g1"], df["g2"])
    df[f"{other}_true_abs"] = df[other].abs()

    # Skip properties that are missing or constant (e.g. redshift, one snap)
    gal_props = [(c, l) for c, l in GALAXY_PROPS
                 if c in df and df[c].notna().sum() >= 5 and df[c].nunique() > 1]
    row_props = [("g_abs_true",        "|g| true"),
                 (true_col,            f"{sym} true"),
                 (f"{other}_true_abs", f"|{other_sym}| true")]

    gal = df.groupby(KEYS, as_index=False).agg(
        med_abs_err=("abs_err", "median"),
        mean_err=("err", "mean"),            # per-galaxy bias
        n_draws=("err", "size"),
        **{c: (c, "first") for c, _ in gal_props},
    ).sort_values("med_abs_err", ascending=False)
    gal.to_csv(out_dir / f"galaxy_errors_{comp}.csv", index=False)

    # Galaxy-level: one point per galaxy, so repeated shear draws don't
    # inflate significance.  Row-level: quantities that change per draw.
    corr_gal = correlations(gal, "med_abs_err", gal_props, "galaxy")
    corr_row = correlations(df,  "abs_err",     row_props, "row")
    corr = pd.DataFrame(corr_gal + corr_row)
    if len(corr):
        corr = corr.reindex(corr["spearman_rho"].abs()
                            .sort_values(ascending=False).index)
        corr.to_csv(out_dir / f"correlations_{comp}.csv", index=False)
        print(f"\n  Spearman correlation with |Δ{sym}|  (sorted by |ρ|)")
        print(f"  {'property':<22} {'level':<7} {'n':>5} {'ρ':>7} {'p':>9}")
        for _, r in corr.iterrows():
            print(f"  {r['property']:<22} {r['level']:<7} {r['n']:>5} "
                  f"{r['spearman_rho']:>+7.3f} {r['p_value']:>9.1e}")

    show = ["subhalo_id", "snap", "n_draws", "med_abs_err", "mean_err"] + \
           [c for c in ["log_mstar", "log_sfr", "gas_frac", "incl_img_deg",
                        "vel_fill_frac", "v_amp_kms"] if c in gal]
    print(f"\n  Worst {args.n_worst} galaxies by median |Δ{sym}|")
    print(gal[show].head(args.n_worst).to_string(index=False, float_format="%.3f"))

    if not args.skip_plots:
        print(f"\n  Generating {sym} error-analysis plots …")
        if gal_props:
            plot_colored_pred_vs_true(df, gal_props, true_col, pred_col, sym,
                                      out_dir / f"pred_vs_true_colored_{comp}.png")
            plot_err_vs_props(gal, "med_abs_err", f"median |Δ{sym}| per galaxy",
                              gal_props, corr_gal,
                              out_dir / f"abs_err_vs_galaxy_props_{comp}.png")
        plot_err_vs_props(df, "abs_err", f"|Δ{sym}|", row_props, corr_row,
                          out_dir / f"abs_err_vs_row_props_{comp}.png")
        plot_gallery(gal, images_root, args.n_gallery, sym,
                     out_dir / f"gallery_worst_best_{comp}.png")
    return corr


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained KL shear model.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pt from train_kl_model.py")
    p.add_argument("--split_csv", default=None,
                   help="CSV to evaluate on (default: split_test.csv next to "
                        "the checkpoint)")
    p.add_argument("--name", default=None,
                   help="Label for this evaluation, used in file names "
                        "(default: derived from the CSV name)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: <checkpoint dir>/eval_<name>)")
    p.add_argument("--images_root", default=None,
                   help="Override images_root stored in the checkpoint")
    p.add_argument("--npix", type=int, default=None,
                   help="Override npix stored in the checkpoint")
    p.add_argument("--use_original_image", action="store_true", default=None,
                   help="Force on-the-fly shearing of image_original.fits "
                        "(default: whatever the checkpoint was trained with)")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_plots",  action="store_true")

    ea = p.add_argument_group("error analysis")
    ea.add_argument("--error_components", nargs="*", default=["g2"],
                    choices=["g1", "g2"],
                    help="Components to analyse (default: g2 = g×)")
    ea.add_argument("--skip_error_analysis", action="store_true")
    ea.add_argument("--apikey", default=None,
                    help="TNG API key (or env TNG_API_KEY)")
    ea.add_argument("--no_tng", action="store_true",
                    help="Skip the TNG API; use FITS-derived properties only")
    ea.add_argument("--sim", default="TNG50-1",
                    help="Simulation name if the CSV has no 'sim' column")
    ea.add_argument("--tng_cache", default=None,
                    help="TNG property cache CSV (default: "
                         "<checkpoint dir>/tng_subhalo_props.csv, shared "
                         "across evaluation runs)")
    ea.add_argument("--n_gallery", type=int, default=6)
    ea.add_argument("--n_worst",   type=int, default=15)
    return p.parse_args()


def main():
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Checkpoint + stored training args ───────────────────────────────────
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    images_root = Path(args.images_root or train_args["images_root"])
    npix        = args.npix or train_args.get("npix", 128)
    use_orig    = (args.use_original_image if args.use_original_image is not None
                   else train_args.get("use_original_image", False))

    print(f"Checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch')}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f})")
    print(f"images_root={images_root}  npix={npix}  "
          f"use_original_image={use_orig}")

    # ── Data ─────────────────────────────────────────────────────────────────
    split_csv = Path(args.split_csv) if args.split_csv \
        else ckpt_path.parent / "split_test.csv"
    name    = args.name or split_csv.stem.replace("split_", "")
    out_dir = Path(args.outdir) if args.outdir \
        else ckpt_path.parent / f"eval_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)
    df = filter_to_existing(df, images_root, use_orig)
    print(f"Evaluating on {split_csv} ({len(df)} rows) as '{name}'")

    loader = DataLoader(
        KLShearDataset(df, images_root, npix, use_orig),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    # pretrained=False: weights come from the checkpoint, no need to download
    model = TwoStreamShearNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\n── Evaluation: {name} ─────────────────────────────────────")
    loss, _, _, preds, labels, avg_eval_time = run_epoch(
        model, loader, nn.SmoothL1Loss(), None, device, training=False
    )

    # Save raw predictions so new analyses can be done without re-running
    pred_df = df.copy()
    pred_df["g1_pred"] = preds[:, 0]
    pred_df["g2_pred"] = preds[:, 1]
    pred_df.to_csv(out_dir / f"predictions_{name}.csv", index=False)

    # ── Evaluations ──────────────────────────────────────────────────────────
    tng_cache = (Path(args.tng_cache) if args.tng_cache
                 else ckpt_path.parent / "tng_subhalo_props.csv")
    ctx = {"df": df, "out_dir": out_dir, "name": name, "loss": loss,
           "avg_eval_time": avg_eval_time, "images_root": images_root,
           "tng_cache": tng_cache, "args": args}
    summary = {"checkpoint": str(ckpt_path), "split_csv": str(split_csv),
               "best_epoch": ckpt.get("epoch")}
    for fn in EVALUATIONS:
        summary.update(fn(preds, labels, ctx) or {})

    pd.DataFrame([summary]).to_csv(out_dir / f"summary_{name}.csv", index=False)
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()