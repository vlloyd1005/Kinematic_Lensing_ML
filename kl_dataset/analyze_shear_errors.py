#!/usr/bin/env python3
"""
analyze_shear_errors.py
=======================
Diagnose what drives shear-prediction errors (default: g×) by joining the
per-row predictions from evaluate_kl_model.py with galaxy properties:

  From the TNG API (subhalo catalogue, cached to CSV after first fetch):
    log M*, log SFR, log sSFR, gas fraction, stellar half-mass radius,
    v_max, stellar / gas metallicity

  From the FITS files on disk (no API needed):
    inclination and θ_int actually used to render the images (FITS header),
    velocity-map fill fraction (fraction of pixels with gas),
    velocity amplitude |v|_95, half-light radius of the stellar image

NOTE on inclination: rows added by expand_shear_draws.py get new random
`inclination` / `theta_int` values in the CSV, but the images were rendered
once with the ORIGINAL values.  So the CSV columns can be wrong for those
rows.  This script reads the true values from the FITS header instead.

Outputs (in <predictions dir>/error_analysis_<component>/):
  pred_vs_true_colored_<c>.png    pred vs true, one panel per property
  abs_err_vs_galaxy_props_<c>.png per-galaxy median |error| vs property
  abs_err_vs_row_props_<c>.png    per-row |error| vs shear-dependent quantities
  gallery_worst_best_<c>.png      images + velmaps of worst / best galaxies
  correlations_<c>.csv            Spearman ρ of |error| with each property
  galaxy_errors_<c>.csv           per-galaxy error table with all properties
  rows_with_props_<c>.csv         per-row predictions joined with properties
  tng_subhalo_props.csv           API cache (reused on later runs)

Usage
-----
  export TNG_API_KEY=...
  python analyze_shear_errors.py \
      --predictions ./kl_model_output/eval_test/predictions_test.csv \
      --images_root /path/to/kl_dataset/images

  # g+ instead of g×, or skip the API entirely
  python analyze_shear_errors.py ... --component g1
  python analyze_shear_errors.py ... --no_tng
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
from astropy.io import fits
from scipy.stats import spearmanr

from generate_kl_tng50 import BASE_URL, h, tng_get

# Full TNG50 snapshots -> redshift (used if the CSV has no redshift column)
SNAP_REDSHIFT = {33: 2.0, 40: 1.5, 50: 1.0, 59: 0.7, 67: 0.5,
                 72: 0.4, 78: 0.3, 84: 0.2, 91: 0.1, 99: 0.0}

COMPONENTS = {"g1": ("g+", "g1", "g1_pred"),
              "g2": ("g×", "g2", "g2_pred")}

TNG_FIELDS = ["mass_stars", "mass_gas", "sfr", "halfmassrad_stars",
              "vmax", "starmetallicity", "gasmetallicity",
              "len_stars", "len_gas"]

KEYS = ["sim", "snap", "subhalo_id"]

# (column, plot label); anything missing from the data is skipped
GALAXY_PROPS = [
    ("log_mstar",        "log M* [M☉]"),
    ("log_sfr",          "log SFR [M☉/yr]"),
    ("log_ssfr",         "log sSFR [1/yr]"),
    ("gas_frac",         "M_gas / (M_gas + M*)"),
    ("r_half_star_kpc",  "r½ stars [kpc]"),
    ("tng_vmax",         "v_max [km/s]"),
    ("tng_starmetallicity", "Z* "),
    ("log_len_gas",      "log N gas particles"),
    ("redshift",         "z"),
    ("incl_img_deg",     "inclination (rendered) [deg]"),
    ("sin2theta",        "sin 2θ_int (rendered)"),
    ("cos2theta",        "cos 2θ_int (rendered)"),
    ("vel_fill_frac",    "velmap fill fraction"),
    ("v_amp_kms",        "|v| 95th pct [km/s]"),
    ("r50_light_kpc",    "r50 light [kpc]"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Property sources
# ═══════════════════════════════════════════════════════════════════════════════

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
    cols = KEYS + [f"tng_{f}" for f in TNG_FIELDS]
    cache = (pd.read_csv(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=cols))
    cache = cache.astype({"sim": str, "snap": int, "subhalo_id": int})
    done  = set(zip(cache["sim"], cache["snap"], cache["subhalo_id"]))

    todo = [g for g in gals.itertuples(index=False)
            if (g.sim, g.snap, g.subhalo_id) not in done]
    print(f"TNG properties: {len(done)} cached, {len(todo)} to fetch")

    headers, new_rows = {"api-key": api_key}, []
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
            print(f"  fetched {i}/{len(todo)}")
            pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True
                      ).to_csv(cache_path, index=False)
        time.sleep(0.2)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return cache


def add_derived(df: pd.DataFrame) -> None:
    """Physical-unit and derived columns (in place)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if "tng_mass_stars" in df:
            a     = 1.0 / (1.0 + df["redshift"])
            mstar = df["tng_mass_stars"] * 1e10 / h
            mgas  = df["tng_mass_gas"]   * 1e10 / h
            sfr   = df["tng_sfr"].clip(lower=1e-3)   # floor for quenched
            df["log_mstar"]       = np.log10(mstar.where(mstar > 0))
            df["log_sfr"]         = np.log10(sfr)
            df["log_ssfr"]        = np.log10(sfr / mstar.where(mstar > 0))
            df["gas_frac"]        = mgas / (mgas + mstar)
            df["r_half_star_kpc"] = df["tng_halfmassrad_stars"] * a / h
            df["log_len_gas"]     = np.log10(df["tng_len_gas"].clip(lower=1))
        if "incl_img" in df:
            df["incl_img_deg"] = np.degrees(df["incl_img"])
        if "theta_img" in df:
            df["sin2theta"] = np.sin(2 * df["theta_img"])
            df["cos2theta"] = np.cos(2 * df["theta_img"])


# ═══════════════════════════════════════════════════════════════════════════════
# Stats + plots
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
    xc  = [np.median(x[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()]
    yc  = [np.median(y[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()]
    return xc, yc


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
        ax.plot(xc, yc, "-o", color="#E84855", ms=3, lw=1.5, label="binned median")
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
            if "incl_img_deg" in g and np.isfinite(g["incl_img_deg"]):
                extra.append(f"i={g['incl_img_deg']:.0f}°")
            if "log_mstar" in g and np.isfinite(g["log_mstar"]):
                extra.append(f"logM*={g['log_mstar']:.1f}")
            if "vel_fill_frac" in g and np.isfinite(g["vel_fill_frac"]):
                extra.append(f"fill={g['vel_fill_frac']:.2f}")
            ax_v.set_title("velmap  " + "  ".join(extra), fontsize=8)
            for ax in (ax_i, ax_v):
                ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Relate shear errors to galaxy properties.")
    p.add_argument("--predictions", required=True,
                   help="predictions_<name>.csv from evaluate_kl_model.py")
    p.add_argument("--images_root", required=True)
    p.add_argument("--component",   choices=["g1", "g2"], default="g2",
                   help="g2 = g× (default), g1 = g+")
    p.add_argument("--outdir",      default=None)
    p.add_argument("--apikey",      default=None,
                   help="TNG API key (or env TNG_API_KEY)")
    p.add_argument("--sim",         default="TNG50-1",
                   help="Simulation name if the CSV has no 'sim' column")
    p.add_argument("--cache",       default=None,
                   help="Path of TNG property cache CSV "
                        "(default: <outdir>/tng_subhalo_props.csv)")
    p.add_argument("--no_tng",      action="store_true",
                   help="Skip API calls; use FITS-derived properties only")
    p.add_argument("--n_gallery",   type=int, default=6)
    p.add_argument("--n_worst",     type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    sym, true_col, pred_col = COMPONENTS[args.component]
    other = "g1" if args.component == "g2" else "g2"
    other_sym = COMPONENTS[other][0]

    preds_path  = Path(args.predictions)
    images_root = Path(args.images_root)
    out_dir = (Path(args.outdir) if args.outdir
               else preds_path.parent / f"error_analysis_{args.component}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Predictions ──────────────────────────────────────────────────────────
    df = pd.read_csv(preds_path)
    if "sim" not in df:
        df["sim"] = args.sim
    df = df.astype({"sim": str, "snap": int, "subhalo_id": int})
    if "redshift" not in df:
        df["redshift"] = df["snap"].map(SNAP_REDSHIFT)

    df["err"]     = df[pred_col] - df[true_col]
    df["abs_err"] = df["err"].abs()
    df["g_abs_true"]          = np.hypot(df["g1"], df["g2"])
    df[f"{other}_true_abs"]   = df[other].abs()

    gals = df[KEYS].drop_duplicates().reset_index(drop=True)
    print(f"{len(df)} rows, {len(gals)} unique galaxies, component {sym}")

    # ── FITS-derived properties ──────────────────────────────────────────────
    print("Reading FITS-derived properties …")
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

    # ── TNG catalogue properties ─────────────────────────────────────────────
    api_key = args.apikey or os.environ.get("TNG_API_KEY")
    if args.no_tng:
        print("Skipping TNG API (--no_tng).")
    elif not api_key:
        print("No TNG API key (--apikey / TNG_API_KEY); using FITS properties only.")
    else:
        cache_path = Path(args.cache) if args.cache else out_dir / "tng_subhalo_props.csv"
        props = fetch_tng_props(gals, api_key, cache_path)
        df = df.merge(props, on=KEYS, how="left")

    add_derived(df)
    df.to_csv(out_dir / f"rows_with_props_{args.component}.csv", index=False)

    # ── Per-galaxy table ─────────────────────────────────────────────────────
    gal_props = [(c, l) for c, l in GALAXY_PROPS
                 if c in df and df[c].notna().sum() >= 5]
    row_props = [("g_abs_true",        "|g| true"),
                 (true_col,            f"{sym} true"),
                 (f"{other}_true_abs", f"|{other_sym}| true")]

    gal = df.groupby(KEYS, as_index=False).agg(
        med_abs_err=("abs_err", "median"),
        mean_err=("err", "mean"),            # per-galaxy bias
        n_draws=("err", "size"),
        **{c: (c, "first") for c, _ in gal_props},
    )
    gal = gal.sort_values("med_abs_err", ascending=False)
    gal.to_csv(out_dir / f"galaxy_errors_{args.component}.csv", index=False)

    # ── Correlations ─────────────────────────────────────────────────────────
    # Galaxy-level: one point per galaxy, so repeated shear draws don't
    # inflate significance.  Row-level: quantities that change per draw.
    corr_gal = correlations(gal, "med_abs_err", gal_props, "galaxy")
    corr_row = correlations(df,  "abs_err",     row_props, "row")
    corr = pd.DataFrame(corr_gal + corr_row)
    if len(corr):
        corr = corr.reindex(corr["spearman_rho"].abs().sort_values(ascending=False).index)
        corr.to_csv(out_dir / f"correlations_{args.component}.csv", index=False)
        print(f"\nSpearman correlation with |Δ{sym}|  (sorted by |ρ|)")
        print(f"  {'property':<22} {'level':<7} {'n':>5} {'ρ':>7} {'p':>9}")
        for _, r in corr.iterrows():
            print(f"  {r['property']:<22} {r['level']:<7} {r['n']:>5} "
                  f"{r['spearman_rho']:>+7.3f} {r['p_value']:>9.1e}")

    show = ["subhalo_id", "snap", "n_draws", "med_abs_err", "mean_err"] + \
           [c for c in ["log_mstar", "log_sfr", "gas_frac", "incl_img_deg",
                        "vel_fill_frac", "v_amp_kms"] if c in gal]
    print(f"\nWorst {args.n_worst} galaxies by median |Δ{sym}|")
    print(gal[show].head(args.n_worst).to_string(index=False, float_format="%.3f"))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    c = args.component
    if gal_props:
        plot_colored_pred_vs_true(df, gal_props, true_col, pred_col, sym,
                                  out_dir / f"pred_vs_true_colored_{c}.png")
        plot_err_vs_props(gal, "med_abs_err", f"median |Δ{sym}| per galaxy",
                          gal_props, corr_gal,
                          out_dir / f"abs_err_vs_galaxy_props_{c}.png")
    plot_err_vs_props(df, "abs_err", f"|Δ{sym}|", row_props, corr_row,
                      out_dir / f"abs_err_vs_row_props_{c}.png")
    plot_gallery(gal, images_root, args.n_gallery, sym,
                 out_dir / f"gallery_worst_best_{c}.png")

    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()