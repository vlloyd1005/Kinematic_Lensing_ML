#!/usr/bin/env python3
"""
saliency_manga.py
=================
SmoothGrad saliency maps for a random set of SDSS/MaNGA galaxies, using the
inputs predict_manga.py already prepared, so the network sees exactly what
it saw when making the predictions.

Saliency is computed with the same code as evaluate_kl_model.py
(_saliency_maps / _saliency_stats), so the maps and the border /
velocity-channel statistics can be compared directly with those for TNG.
Real galaxies have no true shear, so the figures show the predictions
(and the 8-fold rotation/flip mean from predict_manga) instead of errors.

One statistic is specific to real data: how much of the velocity-map
saliency sits on the edge of the IFU footprint, relative to that edge's
share of the area.  A ratio well above 1 means the model reads the shape
of the hexagonal fibre bundle rather than the kinematics.  The footprint
is also drawn (lime) on the velocity panels.

Outputs, in <pred_dir>/saliency/:
  saliency_manga_<mask>.png                overview: inputs + ∂g/∂photo, ∂g/∂vel
  saliency_manga_<mask>_vel_channels_<c>.png   per velocity channel, g+ and g×
  saliency_manga_<mask>_galaxies.csv       per-galaxy statistics
  saliency_manga_<mask>_summary.csv        averages over the selected galaxies

Usage
-----
  python saliency_manga.py --checkpoint kl_dataset/model_output/best_model.pt \
      --pred_dir /content/manga_predictions
  # options: --vel_mask none|dap|strict  --n 5  --seed 1  --min_fill 0.1
  #          --mangaids 1-834 1-1009     (specific galaxies instead of random)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion

from train_kl_model import KLShearDataset, TwoStreamShearNet, apply_shear_numpy
from evaluate_kl_model import VEL_CHANNELS, _saliency_maps, _saliency_stats

COMPS = [("g1", "g+"), ("g2", "g×")]


def ifu_footprint(root, sid, crop):
    """
    Where the prepared velocity map has data, cropped exactly like the model
    input.  None if the map has no empty pixels (e.g. it was smoothed with
    --psf_mode match, which fills them), since then there is no edge.
    """
    path = root / "snap0" / f"galaxy_{sid}_draw0000" / f"galaxy_{sid}_velmap_original.fits"
    m = np.isfinite(fits.getdata(path)).astype(np.float32)
    if m.all():
        return None
    return apply_shear_numpy(m, 0.0, 0.0, crop) > 0.5


def ifu_stats(res, fp, width):
    """Share of velocity saliency on the footprint edge and outside it."""
    out = {}
    if fp is None or not fp.any():
        return out
    edge = binary_dilation(fp, iterations=width) & ~binary_erosion(fp, iterations=width)
    out["ifu_edge_area_frac"] = float(edge.mean())
    for t, _ in COMPS:
        s = res[f"sal_vel_{t}"].sum(0)
        out[f"vel_sal_ifu_edge_frac_{t}"] = float(s[edge].sum() / s.sum())
        out[f"vel_sal_ifu_edge_ratio_{t}"] = out[f"vel_sal_ifu_edge_frac_{t}"] / edge.mean()
        out[f"vel_sal_outside_ifu_frac_{t}"] = float(s[~fp].sum() / s.sum())
    return out


def _show(ax, img, cmap, title, sal=False, m=None, H=None, fp=None):
    if sal:
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0,
                  vmax=np.percentile(img, 99.5) or 1.0)
        ax.add_patch(plt.Rectangle((m - 0.5, m - 0.5), H - 2 * m, H - 2 * m,
                                   fill=False, ec="cyan", lw=0.6, ls="--"))
    else:
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    if fp is not None:
        ax.contour(fp, levels=[0.5], colors="lime", linewidths=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _label(row):
    return (f"{row['mangaid']} ({row['plateifu']})  fill {row['vel_fill_frac']:.2f}\n"
            f"g+ {row['g1_pred']:+.3f} (8-fold {row['g1_dihedral_mean']:+.3f})  "
            f"g× {row['g2_pred']:+.3f} (8-fold {row['g2_dihedral_mean']:+.3f})")


def plot_overview(results, df, fps, out_path, title, m, H):
    n = len(results)
    fig, axes = plt.subplots(n, 6, figsize=(16, 2.9 * n + 0.6), squeeze=False)
    for ax, r, (_, row), fp in zip(axes, results, df.iterrows(), fps):
        _show(ax[0], r["photo"], "gray", _label(row))
        _show(ax[1], r["vel"][0], "RdBu_r", "v_obs input", fp=fp)
        _show(ax[2], r["sal_photo_g1"],      "magma", "∂g+/∂photo", True, m, H)
        _show(ax[3], r["sal_vel_g1"].sum(0), "magma", "∂g+/∂vel",   True, m, H, fp)
        _show(ax[4], r["sal_photo_g2"],      "magma", "∂g×/∂photo", True, m, H)
        _show(ax[5], r["sal_vel_g2"].sum(0), "magma", "∂g×/∂vel",   True, m, H, fp)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_channels(results, df, fps, comp, sym, out_path, title, m, H):
    n = len(results)
    fig, axes = plt.subplots(n, 6, figsize=(16, 2.9 * n + 0.6), squeeze=False)
    for ax, r, (_, row), fp in zip(axes, results, df.iterrows(), fps):
        for c, name in enumerate(VEL_CHANNELS):
            _show(ax[c], r["vel"][c], "RdBu_r",
                  f"{row['mangaid']}\n{name} input" if c == 0 else f"{name} input", fp=fp)
            _show(ax[3 + c], r[f"sal_vel_{comp}"][c], "magma",
                  f"∂{sym}/∂{name}", True, m, H, fp)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Saliency maps for MaNGA galaxies.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pred_dir", default="/content/manga_predictions",
                   help="predict_manga.py output folder")
    p.add_argument("--vel_mask", default="strict", choices=["none", "dap", "strict"])
    p.add_argument("--n", type=int, default=5, help="number of random galaxies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mangaids", nargs="*", default=None,
                   help="specific galaxies instead of a random draw")
    p.add_argument("--min_fill", type=float, default=0.0,
                   help="only draw galaxies with velmap fill ≥ this (e.g. 0.1)")
    p.add_argument("--psf_mode", default="match", choices=["match", "train", "none"],
                   help="must match the predict_manga.py run (default match)")
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_noise", type=float, default=0.05)
    p.add_argument("--saliency_border", type=float, default=0.10)
    p.add_argument("--ifu_edge_px", type=int, default=2,
                   help="half-width of the IFU edge band [model px]")
    return p.parse_args()


def main():
    a = parse_args()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir = Path(a.pred_dir)
    root     = pred_dir / f"inputs_{a.vel_mask}"
    out_dir  = pred_dir / "saliency"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    ta   = ckpt.get("args", {})
    npix, crop = ta.get("npix", 128), ta.get("crop_frac", 1.0)

    # ── Pick galaxies ───────────────────────────────────────────────────────
    pred = pd.read_csv(pred_dir / "predictions_manga.csv")
    df = pred[(pred["vel_mask"] == a.vel_mask) & (pred["vel_fill_frac"] >= a.min_fill)]
    if a.mangaids:
        df = df[df["mangaid"].isin(a.mangaids)]
        missing = set(a.mangaids) - set(df["mangaid"])
        if missing:
            print(f"[WARN] not found for mask '{a.vel_mask}': {sorted(missing)}")
    else:
        df = df.sample(min(a.n, len(df)), random_state=a.seed)
    if df.empty:
        raise SystemExit("No galaxies to show; check --vel_mask / --min_fill / --mangaids.")
    df = df.reset_index(drop=True)
    print(f"{len(df)} galaxies ({a.vel_mask} mask): {', '.join(df['mangaid'])}")

    # Same dataset settings predict_manga used for these inputs
    ds_df = df.assign(subhalo_id=df["index"].astype(int), snap=0, draw_idx=0,
                      g1=0.0, g2=0.0)
    smooth = ta.get("smooth_sigma", 0.0) if a.psf_mode == "train" else 0.0
    ds = KLShearDataset(ds_df, root, npix, ta.get("use_original_image", False),
                        smooth_sigma=smooth, smooth_target=ta.get("smooth_target", "both"),
                        crop_frac=crop)

    model = TwoStreamShearNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Saliency (evaluate_kl_model code) ───────────────────────────────────
    results = _saliency_maps(model, ds, device, ds_df, range(len(df)),
                             np.zeros(len(df)), a.smoothgrad_samples,
                             a.smoothgrad_noise)

    # Sanity check: same input -> same prediction as predict_manga
    for r, (_, row) in zip(results, df.iterrows()):
        d = np.abs(r["pred"] - row[["g1_pred", "g2_pred"]].to_numpy(float)).max()
        if d > 1e-3:
            print(f"  [WARN] {row['mangaid']}: prediction differs from "
                  f"predictions_manga.csv by {d:.4f}; is --psf_mode the same as "
                  f"in the predict_manga run?")

    print(f"\n── Saliency, {a.vel_mask} mask ──")
    summary, m, H = _saliency_stats(results, a.saliency_border, a.vel_mask)

    fps  = [ifu_footprint(root, int(sid), crop) for sid in ds_df["subhalo_id"]]
    rows = []
    for r, (_, row), fp in zip(results, df.iterrows(), fps):
        rec = {"mangaid": row["mangaid"], "plateifu": row["plateifu"],
               "vel_fill_frac": row["vel_fill_frac"],
               "g1_pred": r["pred"][0], "g2_pred": r["pred"][1]}
        for t, _ in COMPS:
            for stream in ("photo", "vel"):
                s_map = r[f"sal_{stream}_{t}"]
                s_map = s_map.sum(0) if stream == "vel" else s_map
                rec[f"sal_total_{stream}_{t}"] = float(s_map.sum())
            rec[f"sal_photo_frac_{t}"] = (rec[f"sal_total_photo_{t}"] /
                                          (rec[f"sal_total_photo_{t}"] +
                                           rec[f"sal_total_vel_{t}"]))
        rec.update(ifu_stats(r, fp, a.ifu_edge_px))
        rows.append(rec)
    gal = pd.DataFrame(rows)

    if "ifu_edge_area_frac" in gal:
        print(f"  Velocity saliency on the IFU edge (±{a.ifu_edge_px} px; "
              f"ratio > 1 means edge-focused)")
        for t, sym in COMPS:
            fr, ra = gal[f"vel_sal_ifu_edge_frac_{t}"], gal[f"vel_sal_ifu_edge_ratio_{t}"]
            out = gal[f"vel_sal_outside_ifu_frac_{t}"]
            print(f"    {sym}: {fr.mean():.1%} on the edge (ratio {ra.mean():.2f}),  "
                  f"{out.mean():.1%} outside the footprint")
            summary[f"vel_sal_ifu_edge_ratio_{t}"] = float(ra.mean())
            summary[f"vel_sal_outside_ifu_frac_{t}"] = float(out.mean())
    for t, sym in COMPS:
        summary[f"sal_photo_frac_{t}"] = float(gal[f"sal_photo_frac_{t}"].mean())
        print(f"  {sym}: {summary[f'sal_photo_frac_{t}']:.0%} of saliency on the image, "
              f"{1 - summary[f'sal_photo_frac_{t}']:.0%} on the velocity map")

    stub = f"saliency_manga_{a.vel_mask}"
    gal.to_csv(out_dir / f"{stub}_galaxies.csv", index=False)
    pd.DataFrame([{"vel_mask": a.vel_mask, "n": len(df), "seed": a.seed,
                   "mangaids": ";".join(df["mangaid"]), **summary}]
                 ).to_csv(out_dir / f"{stub}_summary.csv", index=False)
    print(f"  Saved: {out_dir / (stub + '_galaxies.csv')}")

    # ── Figures ─────────────────────────────────────────────────────────────
    title = f"SDSS/MaNGA, {a.vel_mask} mask, {len(df)} galaxies (seed {a.seed})"
    plot_overview(results, df, fps, out_dir / f"{stub}.png", title, m, H)
    for t, sym in COMPS:
        plot_channels(results, df, fps, t, sym,
                      out_dir / f"{stub}_vel_channels_{t}.png",
                      f"{title}: {sym} per velocity channel", m, H)


if __name__ == "__main__":
    main()
