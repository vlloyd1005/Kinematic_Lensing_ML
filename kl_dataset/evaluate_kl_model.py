#!/usr/bin/env python3
"""
evaluate_kl_model.py
====================
Load a trained TwoStreamShearNet checkpoint and run evaluations without
retraining.

By default it evaluates on the held-out test split that train_kl_model.py
saved next to the checkpoint (split_test.csv), using the same npix /
images_root / use_original_image settings stored in the checkpoint.

Adding a new evaluation
-----------------------
Write a function with the signature

    def eval_something(preds, labels, ctx) -> dict:

where preds / labels are (N, 2) arrays [g+, g×] and ctx is a dict holding
"df", "out_dir", "name", "loss", "avg_eval_time" and "args".  Print whatever
you like, and return a dict of scalars to be added to the summary CSV.
Then append it to the EVALUATIONS list near the bottom of this file.

Usage
-----
  # Test split saved during training
  python evaluate_kl_model.py --checkpoint ./kl_model_output/best_model.pt

  # Any other CSV with subhalo_id, snap, g1, g2 columns
  python evaluate_kl_model.py --checkpoint ./kl_model_output/best_model.pt \
      --split_csv ./kl_model_output/split_val.csv --name val
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# Add new evaluation functions here; they run in order.
EVALUATIONS = [
    eval_basic_metrics,
    eval_abs_error_percentiles,
    eval_frac_error_percentiles,
    eval_plots,
]


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
    ctx = {"df": df, "out_dir": out_dir, "name": name, "loss": loss,
           "avg_eval_time": avg_eval_time, "args": args}
    summary = {"checkpoint": str(ckpt_path), "split_csv": str(split_csv),
               "best_epoch": ckpt.get("epoch")}
    for fn in EVALUATIONS:
        summary.update(fn(preds, labels, ctx) or {})

    pd.DataFrame([summary]).to_csv(out_dir / f"summary_{name}.csv", index=False)
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()