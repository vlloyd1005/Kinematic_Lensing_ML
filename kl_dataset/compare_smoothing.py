#!/usr/bin/env python3
"""
compare_smoothing.py
====================
Collect test_summary.csv from every run under --root (one sub-folder per
smoothing sigma, e.g. sigma_0, sigma_2, ...) and plot how the test error
changes with Gaussian smoothing.

Outputs (in --root):
  smoothing_comparison.csv   one row per run
  mse_vs_smoothing.png       MSE per component, and % change vs no smoothing
  error_vs_smoothing.png     subplots for g+ and g×: absolute-error
                             percentiles vs sigma, with the KL noise floor

Usage
-----
  python compare_smoothing.py --root kl_dataset/model_output_smooth
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KL_NOISE_FLOOR = 0.035          # per-galaxy KL shape noise (Xu+2022 §3.3)
COMPS = [("g1", "g+", "#2E86AB"), ("g2", "g×", "#E84855")]
PCTS  = [50, 75, 90, 95]


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/test_summary.csv")):
        s = pd.read_csv(f).iloc[0].to_dict()
        if "smooth_sigma" not in s or pd.isna(s.get("smooth_sigma")):
            m = re.search(r"sigma_([\d.]+)", f.parent.name)
            if not m:
                print(f"  [skip] {f.parent.name}: no smooth_sigma in summary "
                      f"or folder name")
                continue
            s["smooth_sigma"] = float(m.group(1))
        s["run"] = f.parent.name
        rows.append(s)
    if not rows:
        raise SystemExit(f"No */test_summary.csv found under {root}")
    return pd.DataFrame(rows).sort_values("smooth_sigma").reset_index(drop=True)


def add_kpc_axis(ax, kpc_per_pix):
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * kpc_per_pix,
                                               lambda x: x / kpc_per_pix))
    sec.set_xlabel("σ [kpc]", fontsize=9)


def main():
    p = argparse.ArgumentParser(description="Compare smoothing runs.")
    p.add_argument("--root", required=True,
                   help="Folder containing one sub-folder per run")
    p.add_argument("--kpc_per_pix", type=float, default=None,
                   help="Physical pixel scale of the model input (default: "
                        "kpc_per_pix from test_summary.csv, else 30/128)")
    args = p.parse_args()
    root = Path(args.root)

    df = load_runs(root)
    sig = df["smooth_sigma"].to_numpy(float)
    df.to_csv(root / "smoothing_comparison.csv", index=False)
    if args.kpc_per_pix is None:
        args.kpc_per_pix = (float(df["kpc_per_pix"].iloc[0])
                            if "kpc_per_pix" in df else 30.0 / 128)

    ref = df.iloc[0]                # smallest sigma (ideally 0) as reference
    print(f"Reference run: {ref['run']} (σ = {ref['smooth_sigma']} px)\n")
    print(f"  {'run':<14} {'σ px':>5} {'σ kpc':>6} {'MSE g+':>10} {'MSE g×':>10}"
          f" {'Δ g+':>7} {'Δ g×':>7}")
    for _, r in df.iterrows():
        d1 = 100 * (r["test_mse_g1"] / ref["test_mse_g1"] - 1)
        d2 = 100 * (r["test_mse_g2"] / ref["test_mse_g2"] - 1)
        print(f"  {r['run']:<14} {r['smooth_sigma']:>5g} "
              f"{r['smooth_sigma'] * args.kpc_per_pix:>6.2f} "
              f"{r['test_mse_g1']:>10.3e} {r['test_mse_g2']:>10.3e} "
              f"{d1:>+6.0f}% {d2:>+6.0f}%")

    # ── Figure 1: MSE and % change ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for tag, sym, color in COMPS:
        mse = df[f"test_mse_{tag}"].to_numpy(float)
        axes[0].plot(sig, mse, "-o", color=color, label=sym)
        axes[1].plot(sig, 100 * (mse / mse[0] - 1), "-o", color=color, label=sym)
    axes[0].set_ylabel("Test MSE"); axes[0].set_yscale("log")
    axes[0].set_title("Test MSE vs smoothing")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel(f"% change in MSE vs σ = {sig[0]:g} px")
    axes[1].set_title("Change in MSE vs smoothing")
    for ax in axes:
        ax.set_xlabel("smoothing σ [px]"); ax.set_xticks(sig)
        ax.grid(alpha=0.3); ax.legend()
        add_kpc_axis(ax, args.kpc_per_pix)
    fig.tight_layout()
    fig.savefig(root / "mse_vs_smoothing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: mse_vs_smoothing.png")

    # ── Figure 2: error percentiles, one subplot per component ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    shades = plt.cm.viridis(np.linspace(0.1, 0.85, len(PCTS)))
    for ax, (tag, sym, _) in zip(axes, COMPS):
        for pct, c in zip(PCTS, shades):
            col = f"abs_err_pct{pct:02d}_{tag}"
            if col in df:
                ax.plot(sig, df[col], "-o", color=c, ms=4, label=f"{pct}th pct")
        rmse = df[f"test_rmse_{tag}"]
        ax.plot(sig, rmse, "k--", lw=1.2, label="RMSE")
        ax.axhline(KL_NOISE_FLOOR, color="grey", ls=":", lw=1.2,
                   label=f"KL noise floor ({KL_NOISE_FLOOR})")
        ax.set_title(f"{sym}: |pred − true| vs smoothing")
        ax.set_xlabel("smoothing σ [px]"); ax.set_xticks(sig)
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        add_kpc_axis(ax, args.kpc_per_pix)
    axes[0].set_ylabel("absolute error")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(root / "error_vs_smoothing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: error_vs_smoothing.png")
    print(f"  Saved: smoothing_comparison.csv")


if __name__ == "__main__":
    main()
