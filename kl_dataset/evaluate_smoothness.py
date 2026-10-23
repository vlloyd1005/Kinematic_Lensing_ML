#!/usr/bin/env python3
"""
evaluate_smoothness.py
======================
Compare saliency maps across smoothing runs on the SAME galaxies.

1. Run the reference model (default σ = 0) on its test split and pick
   galaxies from the best band of its error distribution (default: the
   0–20th percentile of |Δg×|), at most one row per galaxy.
2. Feed exactly those rows (same galaxy, same shear draw) to every model in
   --sigmas, each with its own smoothing / crop settings from its checkpoint.
3. For every σ, compute SmoothGrad saliency maps and save:
     <outdir>/sigma_<s>/saliency_<…>.png                overview per σ
     <outdir>/sigma_<s>/saliency_gx_vel_channels_<…>.png g× per velocity channel
     <outdir>/compare_gal<id>_snap<snap>.png             one galaxy, rows = σ
     <outdir>/saliency_vs_sigma.png                      border ratio and
                                                         velocity-channel shares vs σ
     <outdir>/saliency_vs_sigma.csv                      numbers behind it

Checkpoints are expected at <root>/sigma_<s>/best_model.pt, where <s> is
written exactly as passed to --sigmas (e.g. 0, 0.5, 1.5), matching the
folder names made by the SLURM array script.

Usage
-----
  python kl_dataset/evaluate_smoothness.py \
      --root kl_dataset/model_output_full \
      --sigmas 0 0.5 1 1.5 2 4
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
from torch.utils.data import DataLoader

from train_kl_model import (
    KLShearDataset,
    TwoStreamShearNet,
    filter_by_vel_fill,
    filter_to_existing,
    run_epoch,
)
from evaluate_kl_model import (
    VEL_CHANNELS,
    _pick_rows_by_error,
    _saliency_maps,
    _saliency_plots,
    _saliency_stats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_run(ckpt_path: Path, device, images_root_override=None):
    """Model + the data settings it was trained with."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta   = ckpt.get("args", {})
    cfg  = {
        "images_root":   Path(images_root_override or ta["images_root"]),
        "npix":          ta.get("npix", 128),
        "use_orig":      ta.get("use_original_image", False),
        "smooth_sigma":  float(ta.get("smooth_sigma", 0.0)),
        "smooth_target": ta.get("smooth_target", "both"),
        "crop_frac":     float(ta.get("crop_frac", 1.0)),
        "min_vel_fill":  float(ta.get("min_vel_fill", 0.0)),
    }
    model = TwoStreamShearNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def make_dataset(df, cfg):
    return KLShearDataset(df, cfg["images_root"], cfg["npix"], cfg["use_orig"],
                          smooth_sigma=cfg["smooth_sigma"],
                          smooth_target=cfg["smooth_target"],
                          crop_frac=cfg["crop_frac"])


def predict(model, df, cfg, device, batch_size, num_workers):
    loader = DataLoader(make_dataset(df, cfg), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    _, _, _, preds, labels, _ = run_epoch(model, loader, nn.SmoothL1Loss(),
                                          None, device, training=False)
    return preds, labels


def check_not_in_training(run_dir: Path, sel: pd.DataFrame, label: str):
    """Warn if a selected galaxy was in this run's train/val split."""
    keys = set(zip(sel["subhalo_id"], sel["snap"]))
    for split in ("split_train.csv", "split_val.csv"):
        f = run_dir / split
        if f.exists():
            seen = pd.read_csv(f, usecols=["subhalo_id", "snap"])
            hit  = keys & set(zip(seen["subhalo_id"], seen["snap"]))
            if hit:
                print(f"  [WARN] {label}: {len(hit)} selected galaxies are in "
                      f"its {split}: {sorted(hit)[:5]}")


def plot_galaxy_comparison(per_sigma, j, sym, out_path):
    """One galaxy; rows = σ; inputs + saliency for both components."""
    labels = list(per_sigma)
    fig, axes = plt.subplots(len(labels), 6, figsize=(16, 2.7 * len(labels) + 0.6),
                             squeeze=False)
    first = per_sigma[labels[0]]["results"][j]
    for r_i, s in enumerate(labels):
        r = per_sigma[s]["results"][j]
        panels = [
            (r["photo"], "gray", f"σ = {s} px  photo\n"
             f"g+ {r['true'][0]:+.3f}→{r['pred'][0]:+.3f}  "
             f"g× {r['true'][1]:+.3f}→{r['pred'][1]:+.3f}", False),
            (r["vel"][0], "RdBu_r", f"v_obs   |Δ{sym}|={r['err']:.4f}", False),
            (r["sal_photo_g1"],      "magma", "∂g+/∂photo", True),
            (r["sal_vel_g1"].sum(0), "magma", "∂g+/∂vel",   True),
            (r["sal_photo_g2"],      "magma", "∂g×/∂photo", True),
            (r["sal_vel_g2"].sum(0), "magma", "∂g×/∂vel",   True),
        ]
        for c, (img, cmap, title, is_sal) in enumerate(panels):
            ax = axes[r_i, c]
            if is_sal:
                ax.imshow(img, origin="lower", cmap=cmap, vmin=0,
                          vmax=np.percentile(img, 99.5) or 1.0)
            else:
                ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"id {first['sid']} (snap {first['snap']}) across smoothing",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_stats_vs_sigma(table: pd.DataFrame, out_path: Path):
    sig = table["sigma"].to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for t, sym, ax in (("g1", "g+", axes[0]), ("g2", "g×", axes[1])):
        for n, color in zip(VEL_CHANNELS, ["#2E86AB", "#E84855", "#3BB273"]):
            ax.plot(sig, table[f"share_{n}_{t}"], "-o", color=color, label=n)
        ax.set_title(f"{sym}: velocity-channel share of saliency")
        ax.set_ylim(0, 1); ax.set_ylabel("share")

    ax = axes[2]
    for t, sym, ls in (("g1", "g+", "-"), ("g2", "g×", "--")):
        for stream, color in (("photo", "#555555"), ("vel", "#E84855")):
            ax.plot(sig, table[f"border_ratio_{stream}_{t}"], ls, marker="o",
                    color=color, label=f"{sym} {stream}")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_title("Border saliency ÷ border area (1 = no edge focus)")
    ax.set_ylabel("ratio")

    for ax in axes:
        ax.set_xlabel("smoothing σ [px]"); ax.set_xticks(sig)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Saliency across smoothing runs.")
    p.add_argument("--root", default="kl_dataset/model_output_full",
                   help="Folder containing sigma_<s>/best_model.pt")
    p.add_argument("--sigmas", nargs="+", default=["0", "0.5", "1", "1.5", "2", "4"],
                   help="Sigma folder labels, written as in the folder names")
    p.add_argument("--ref_sigma", default="0",
                   help="Run whose errors choose the galaxies (default 0)")
    p.add_argument("--percentile", type=float, default=20,
                   help="Use the 0–<percentile> band of the reference errors")
    p.add_argument("--pick", choices=["random", "lowest"], default="random",
                   help="random: random galaxies within the band; "
                        "lowest: the very best galaxies")
    p.add_argument("--component", choices=["g1", "g2"], default="g2",
                   help="Component whose error ranks galaxies (default g×)")
    p.add_argument("--n_galaxies", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_csv", default=None,
                   help="Default: split_test.csv of the reference run")
    p.add_argument("--outdir", default=None,
                   help="Default: <root>/saliency_smoothness")
    p.add_argument("--images_root", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_noise", type=float, default=0.05)
    p.add_argument("--saliency_border", type=float, default=0.10)
    return p.parse_args()


def main():
    args   = parse_args()
    root   = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir) if args.outdir else root / "saliency_smoothness"
    out_dir.mkdir(parents=True, exist_ok=True)
    col = 0 if args.component == "g1" else 1
    sym = "g+" if args.component == "g1" else "g×"
    print(f"Device: {device}")

    # ── 1. Choose galaxies from the reference run ────────────────────────────
    ref_dir = root / f"sigma_{args.ref_sigma}"
    ref_model, ref_cfg, _ = load_run(ref_dir / "best_model.pt", device,
                                     args.images_root)
    split_csv = Path(args.split_csv) if args.split_csv else ref_dir / "split_test.csv"
    df = pd.read_csv(split_csv)
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)
    df = filter_to_existing(df, ref_cfg["images_root"], ref_cfg["use_orig"])
    df = filter_by_vel_fill(df, ref_cfg["images_root"], ref_cfg["min_vel_fill"])

    print(f"\nReference σ = {args.ref_sigma}: predicting {len(df)} rows "
          f"from {split_csv}")
    preds, labels = predict(ref_model, df, ref_cfg, device,
                            args.batch_size, args.num_workers)
    err = np.abs(preds[:, col] - labels[:, col])
    del ref_model

    if args.pick == "lowest":
        order = pd.DataFrame({"i": np.arange(len(df)), "e": err,
                              "sid": df["subhalo_id"], "snap": df["snap"]})
        order = order.sort_values("e").drop_duplicates(["sid", "snap"])
        idxs  = order["i"].to_numpy()[:args.n_galaxies]
        e_hi  = float(np.percentile(err, args.percentile))
        idxs  = idxs[err[idxs] <= e_hi]
    else:
        rng = np.random.default_rng(args.seed)
        _, _, _, e_hi, idxs = _pick_rows_by_error(
            df, err, [0, args.percentile], args.n_galaxies, rng)[0]

    sel = df.loc[idxs].reset_index(drop=True)
    print(f"Selected {len(sel)} galaxies from the 0–{args.percentile:g} "
          f"percentile of |Δ{sym}| (≤ {e_hi:.4f}) at σ = {args.ref_sigma}:")
    for (_, r), e in zip(sel.iterrows(), err[idxs]):
        print(f"  id {int(r['subhalo_id'])} snap {int(r['snap'])}  "
              f"g+={r['g1']:+.4f} g×={r['g2']:+.4f}  |Δ{sym}|={e:.4f}")
    sel.to_csv(out_dir / "selected_galaxies.csv", index=False)
    if len(sel) == 0:
        raise SystemExit("No galaxies selected.")

    # ── 2–3. Saliency for every σ on the same rows ──────────────────────────
    per_sigma, rows, crops = {}, [], set()
    for s in args.sigmas:
        run_dir = root / f"sigma_{s}"
        ckpt    = run_dir / "best_model.pt"
        if not ckpt.exists():
            print(f"\n[WARN] {ckpt} not found; skipping σ = {s}")
            continue
        print(f"\n── σ = {s} ─────────────────────────────────────────")
        model, cfg, _ = load_run(ckpt, device, args.images_root)
        crops.add(cfg["crop_frac"])
        if abs(cfg["smooth_sigma"] - float(s)) > 1e-9:
            print(f"  [WARN] folder says σ = {s} but checkpoint was trained "
                  f"with smooth_sigma = {cfg['smooth_sigma']}")
        check_not_in_training(run_dir, sel, f"σ = {s}")

        p_s, l_s = predict(model, sel, cfg, device, args.batch_size, 0)
        err_s = np.abs(p_s[:, col] - l_s[:, col])

        results = _saliency_maps(model, make_dataset(sel, cfg), device, sel,
                                 range(len(sel)), err_s,
                                 args.smoothgrad_samples, args.smoothgrad_noise)
        stats, m, H = _saliency_stats(results, args.saliency_border, f"sigma{s}")

        sig_dir = out_dir / f"sigma_{s}"
        sig_dir.mkdir(exist_ok=True)
        _saliency_plots(results, sig_dir,
                        f"sigma{s}_{args.component}_p0-{args.percentile:g}",
                        f"σ = {s} px — galaxies from the best "
                        f"{args.percentile:g}% at σ = {args.ref_sigma} "
                        f"(|Δ{sym}| ≤ {e_hi:.4f})", sym, m, H)
        per_sigma[s] = {"results": results}

        area = stats["saliency_border_area_frac"]
        row = {"sigma": float(s), f"mean_abs_err_{args.component}": float(err_s.mean())}
        for t in ("g1", "g2"):
            for stream in ("photo", "vel"):
                row[f"border_ratio_{stream}_{t}"] = (
                    stats[f"saliency_border_frac_{stream}_{t}_sigma{s}"] / area)
            for n in VEL_CHANNELS:
                row[f"share_{n}_{t}"] = stats[f"saliency_vel_share_{n}_{t}_sigma{s}"]
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(crops) > 1:
        print(f"\n[WARN] Runs use different crop_frac values {sorted(crops)}; "
              f"they are not directly comparable.")
    if not per_sigma:
        raise SystemExit("No runs found.")

    # ── Cross-σ outputs ─────────────────────────────────────────────────────
    print("\nGenerating cross-σ figures …")
    for j in range(len(sel)):
        r0 = per_sigma[next(iter(per_sigma))]["results"][j]
        plot_galaxy_comparison(per_sigma, j, sym,
                               out_dir / f"compare_gal{r0['sid']}_snap{r0['snap']}.png")

    table = pd.DataFrame(rows).sort_values("sigma")
    table.to_csv(out_dir / "saliency_vs_sigma.csv", index=False)
    if len(table) > 1:
        plot_stats_vs_sigma(table, out_dir / "saliency_vs_sigma.png")

    print(f"\n  {'σ':>5}  {'mean|Δ' + sym + '|':>11}  " +
          "  ".join(f"{n:>7}" for n in VEL_CHANNELS) + "   border(vel g×)")
    for _, r in table.iterrows():
        print(f"  {r['sigma']:>5g}  {r[f'mean_abs_err_{args.component}']:>11.4f}  " +
              "  ".join(f"{r[f'share_{n}_g2']:>7.0%}" for n in VEL_CHANNELS) +
              f"   {r['border_ratio_vel_g2']:.2f}")
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
