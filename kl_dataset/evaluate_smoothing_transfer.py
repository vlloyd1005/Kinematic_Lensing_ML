#!/usr/bin/env python3
"""
evaluate_smoothing_transfer.py
==============================
Apply ONE model (default: the one trained without smoothing, σ = 0) to inputs
smoothed by different amounts, to see how it degrades and where it looks.

1. Run the model on its own test split at its training smoothing and pick
   galaxies from the best band of its errors (default: 0–20th percentile of
   |Δg×|), at most one row per galaxy.
2. For every σ in --sigmas, smooth the inputs by σ and:
     - evaluate the whole test set   → error vs σ curve
     - compute SmoothGrad saliency for the selected rows (same galaxies,
       same shear draws at every σ)
3. If retrained runs exist at <root>/sigma_<s>/test_summary.csv, their test
   RMSE is plotted alongside, showing the cost of NOT training on smoothed
   data.  All runs share the same test galaxies (same CSV, seed and cuts).

Outputs (default <root>/smoothing_transfer/):
  selected_galaxies.csv
  transfer_metrics.csv           fixed-model test metrics per σ (+ retrained)
  error_vs_sigma.png             g+ and g× RMSE / median error vs σ
  sigma_<s>/pred_vs_true_*.png   whole test set, same axes for every σ
  sigma_<s>/saliency_*.png       overview and g× velocity-channel figures
  compare_gal<id>_snap<snap>.png one galaxy, rows = σ
  saliency_vs_sigma.png / .csv   channel shares and border ratio vs σ

Usage
-----
  python kl_dataset/evaluate_smoothing_transfer.py \
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

from train_kl_model import filter_by_vel_fill, filter_to_existing
from evaluate_kl_model import (
    VEL_CHANNELS,
    _pick_rows_by_error,
    _saliency_maps,
    _saliency_plots,
    _saliency_stats,
)
from evaluate_smoothness import (
    load_run,
    make_dataset,
    plot_galaxy_comparison,
    plot_stats_vs_sigma,
    predict,
)

KL_NOISE_FLOOR = 0.035
COMPS = [(0, "g1", "g+", "#2E86AB"), (1, "g2", "g×", "#E84855")]


def plot_error_vs_sigma(table: pd.DataFrame, model_sigma: str, out_path: Path):
    sig = table["sigma"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (_, tag, sym, color) in zip(axes, COMPS):
        ax.plot(sig, table[f"rmse_{tag}"], "-o", color=color,
                label=f"σ={model_sigma} model, RMSE")
        ax.plot(sig, table[f"abs_err_pct50_{tag}"], ":o", color=color, ms=4,
                label=f"σ={model_sigma} model, median |Δ|")
        rt = table[f"retrained_rmse_{tag}"]
        if rt.notna().any():
            ok = rt.notna().to_numpy()
            ax.plot(sig[ok], rt[ok], "--s", color="k", ms=5,
                    label="retrained at each σ, RMSE")
        ax.axhline(KL_NOISE_FLOOR, color="grey", ls=":", lw=1.2,
                   label=f"KL noise floor ({KL_NOISE_FLOOR})")
        ax.set_title(f"{sym}: error vs input smoothing")
        ax.set_xlabel("input smoothing σ [px]"); ax.set_xticks(sig)
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("absolute error")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_pred_vs_true_fixed(preds, labels, sigma, model_sigma, out_path: Path):
    """
    Predicted vs true for g+ and g×, with axis limits set by the TRUE shear
    range so every σ gets identical axes and can be compared directly.
    Predictions outside the axes are counted in the title.
    """
    lim = np.abs(labels).max() * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (c, tag, sym, color) in zip(axes, COMPS):
        t, pr = labels[:, c], preds[:, c]
        mse   = float(np.mean((pr - t) ** 2))
        n_out = int((np.abs(pr) > lim).sum())
        ax.scatter(t, np.clip(pr, -lim, lim), s=6, alpha=0.5, color=color,
                   rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{sym} true"); ax.set_ylabel(f"{sym} predicted")
        ax.set_title(f"{sym}  [σ={model_sigma} model, input σ = {sigma} px]\n"
                     f"MSE={mse:.2e}" + (f"   {n_out} outside axes" if n_out else ""),
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.parent.name}/{out_path.name}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed model applied to differently smoothed inputs.")
    p.add_argument("--root", default="kl_dataset/model_output_full",
                   help="Folder containing sigma_<s>/ run folders")
    p.add_argument("--model_sigma", default="0",
                   help="Folder label of the fixed model (default: sigma_0)")
    p.add_argument("--sigmas", nargs="+", default=["0", "0.5", "1", "1.5", "2", "4"],
                   help="Input smoothings to apply (px); also used to find "
                        "retrained runs at <root>/sigma_<s>")
    p.add_argument("--smooth_target", choices=["both", "photo", "vel"],
                   default="both", help="Which inputs to smooth")
    p.add_argument("--percentile", type=float, default=20,
                   help="Pick galaxies from the 0–<percentile> error band")
    p.add_argument("--pick", choices=["random", "lowest"], default="random")
    p.add_argument("--component", choices=["g1", "g2"], default="g2")
    p.add_argument("--n_galaxies", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_csv", default=None,
                   help="Default: split_test.csv of the fixed model's run")
    p.add_argument("--outdir", default=None,
                   help="Default: <root>/smoothing_transfer")
    p.add_argument("--images_root", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_noise", type=float, default=0.05)
    p.add_argument("--saliency_border", type=float, default=0.10)
    return p.parse_args()


def main():
    args    = parse_args()
    root    = Path(args.root)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir) if args.outdir else root / "smoothing_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    col = 0 if args.component == "g1" else 1
    sym = "g+" if args.component == "g1" else "g×"
    print(f"Device: {device}")

    # ── Fixed model and its test set ─────────────────────────────────────────
    run_dir = root / f"sigma_{args.model_sigma}"
    model, base_cfg, _ = load_run(run_dir / "best_model.pt", device,
                                  args.images_root)
    print(f"Fixed model: {run_dir / 'best_model.pt'} "
          f"(trained with smooth_sigma = {base_cfg['smooth_sigma']}, "
          f"crop_frac = {base_cfg['crop_frac']})")

    split_csv = Path(args.split_csv) if args.split_csv else run_dir / "split_test.csv"
    df = pd.read_csv(split_csv)
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)
    df = filter_to_existing(df, base_cfg["images_root"], base_cfg["use_orig"])
    df = filter_by_vel_fill(df, base_cfg["images_root"], base_cfg["min_vel_fill"])

    # ── 1. Choose galaxies at the model's own training smoothing ────────────
    print(f"\nPredicting {len(df)} test rows at the training smoothing …")
    ref_p, ref_l = predict(model, df, base_cfg, device,
                           args.batch_size, args.num_workers)
    err = np.abs(ref_p[:, col] - ref_l[:, col])

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
          f"percentile of |Δ{sym}| (≤ {e_hi:.4f}):")
    for (_, r), e in zip(sel.iterrows(), err[idxs]):
        print(f"  id {int(r['subhalo_id'])} snap {int(r['snap'])}  "
              f"g+={r['g1']:+.4f} g×={r['g2']:+.4f}  |Δ{sym}|={e:.4f}")
    sel.to_csv(out_dir / "selected_galaxies.csv", index=False)
    if len(sel) == 0:
        raise SystemExit("No galaxies selected.")

    # ── 2. Every input smoothing ─────────────────────────────────────────────
    metrics, sal_rows, per_sigma = [], [], {}
    for s in args.sigmas:
        print(f"\n── input σ = {s} px ───────────────────────────────────")
        cfg = {**base_cfg, "smooth_sigma": float(s),
               "smooth_target": args.smooth_target}

        # whole test set
        p_all, l_all = predict(model, df, cfg, device,
                               args.batch_size, args.num_workers)
        m = {"sigma": float(s)}
        for c, tag, _, _ in COMPS:
            e_all = np.abs(p_all[:, c] - l_all[:, c])
            m[f"mse_{tag}"]  = float(np.mean(e_all ** 2))
            m[f"rmse_{tag}"] = float(np.sqrt(m[f"mse_{tag}"]))
            for pct in (50, 90):
                m[f"abs_err_pct{pct}_{tag}"] = float(np.percentile(e_all, pct))
        summ = root / f"sigma_{s}" / "test_summary.csv"
        if summ.exists():
            ts = pd.read_csv(summ).iloc[0]
            for _, tag, _, _ in COMPS:
                m[f"retrained_rmse_{tag}"] = float(ts[f"test_rmse_{tag}"])
            if "crop_frac" in ts and abs(float(ts["crop_frac"]) - base_cfg["crop_frac"]) > 1e-9:
                print(f"  [WARN] retrained σ = {s} used crop_frac "
                      f"{ts['crop_frac']} vs {base_cfg['crop_frac']}")
        else:
            for _, tag, _, _ in COMPS:
                m[f"retrained_rmse_{tag}"] = np.nan
        metrics.append(m)
        print(f"  test RMSE  g+ {m['rmse_g1']:.4e}   g× {m['rmse_g2']:.4e}")

        sig_dir = out_dir / f"sigma_{s}"
        sig_dir.mkdir(exist_ok=True)
        plot_pred_vs_true_fixed(p_all, l_all, s, args.model_sigma,
                                sig_dir / f"pred_vs_true_input_sigma{s}.png")

        # saliency on the selected rows
        p_s, l_s = predict(model, sel, cfg, device, args.batch_size, 0)
        err_s = np.abs(p_s[:, col] - l_s[:, col])
        results = _saliency_maps(model, make_dataset(sel, cfg), device, sel,
                                 range(len(sel)), err_s,
                                 args.smoothgrad_samples, args.smoothgrad_noise)
        stats, mb, H = _saliency_stats(results, args.saliency_border, f"sigma{s}")
        _saliency_plots(results, sig_dir,
                        f"input_sigma{s}_{args.component}_p0-{args.percentile:g}",
                        f"σ={args.model_sigma} model on inputs smoothed by "
                        f"σ = {s} px — best {args.percentile:g}% galaxies "
                        f"(|Δ{sym}| ≤ {e_hi:.4f} unsmoothed)", sym, mb, H)
        per_sigma[s] = {"results": results}

        area = stats["saliency_border_area_frac"]
        row = {"sigma": float(s), f"mean_abs_err_{args.component}": float(err_s.mean())}
        for t in ("g1", "g2"):
            for stream in ("photo", "vel"):
                row[f"border_ratio_{stream}_{t}"] = (
                    stats[f"saliency_border_frac_{stream}_{t}_sigma{s}"] / area)
            for n in VEL_CHANNELS:
                row[f"share_{n}_{t}"] = stats[f"saliency_vel_share_{n}_{t}_sigma{s}"]
        sal_rows.append(row)

    # ── 3. Cross-σ outputs ──────────────────────────────────────────────────
    print("\nGenerating cross-σ figures …")
    mt = pd.DataFrame(metrics).sort_values("sigma")
    mt.to_csv(out_dir / "transfer_metrics.csv", index=False)
    plot_error_vs_sigma(mt, args.model_sigma, out_dir / "error_vs_sigma.png")

    for j in range(len(sel)):
        r0 = per_sigma[args.sigmas[0]]["results"][j]
        plot_galaxy_comparison(per_sigma, j, sym,
                               out_dir / f"compare_gal{r0['sid']}_snap{r0['snap']}.png")

    st = pd.DataFrame(sal_rows).sort_values("sigma")
    st.to_csv(out_dir / "saliency_vs_sigma.csv", index=False)
    if len(st) > 1:
        plot_stats_vs_sigma(st, out_dir / "saliency_vs_sigma.png")

    print(f"\n  {'σ in':>5}  {'RMSE g+':>10} {'RMSE g×':>10}   "
          f"{'retrained g+':>12} {'retrained g×':>12}")
    for _, r in mt.iterrows():
        rt1 = f"{r['retrained_rmse_g1']:.4e}" if pd.notna(r["retrained_rmse_g1"]) else "—"
        rt2 = f"{r['retrained_rmse_g2']:.4e}" if pd.notna(r["retrained_rmse_g2"]) else "—"
        print(f"  {r['sigma']:>5g}  {r['rmse_g1']:>10.4e} {r['rmse_g2']:>10.4e}   "
              f"{rt1:>12} {rt2:>12}")
    print(f"\nAll outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()