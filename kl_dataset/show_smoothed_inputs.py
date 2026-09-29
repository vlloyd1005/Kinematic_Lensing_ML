#!/usr/bin/env python3
"""
show_smoothed_inputs.py
=======================
Plot the model inputs (stellar image, v_obs, v_asym, v_sym) for a few random
galaxies at one Gaussian smoothing sigma, exactly as KLShearDataset feeds
them to the network.

The same --seed picks the same galaxies (and the same shear draw) for every
sigma, so figures from different runs can be compared directly.  All panels
use the fixed [0, 1] colour range of the normalised inputs.

Usage
-----
  python show_smoothed_inputs.py \
      --images_root "kl_dataset/images" \
      --csv         "kl_dataset/data/dataset_plan_expanded_2draws.csv" \
      --smooth_sigma 2 \
      --outdir      "kl_dataset/model_output_smooth/sigma_2"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_kl_model import KLShearDataset, filter_by_vel_fill, filter_to_existing


def main():
    p = argparse.ArgumentParser(description="Plot smoothed model inputs.")
    p.add_argument("--images_root", required=True)
    p.add_argument("--csv", required=True,
                   help="Dataset CSV (use the same one for every sigma)")
    p.add_argument("--smooth_sigma", type=float, default=0.0,
                   help="Gaussian sigma in pixels of the npix image")
    p.add_argument("--smooth_target", choices=["both", "photo", "vel"],
                   default="both")
    p.add_argument("--crop_frac", type=float, default=0.75,
                   help="Central fraction kept after shearing (as in training)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--npix", type=int, default=128)
    p.add_argument("--n_galaxies", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_vel_fill", type=float, default=0.5,
                   help="Same galaxy cut as training (0 = off)")
    p.add_argument("--use_original_image", action="store_true",
                   help="Shear image_original.fits on the fly (as in training)")
    args = p.parse_args()

    images_root = Path(args.images_root)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["subhalo_id"] != -1].reset_index(drop=True)
    df = filter_to_existing(df, images_root, args.use_original_image)
    df = filter_by_vel_fill(df, images_root, args.min_vel_fill)

    # One row (first shear draw) per galaxy, then a seeded random subset
    gals = df.drop_duplicates(["subhalo_id", "snap"])
    rng  = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(gals.index.to_numpy(),
                              size=min(args.n_galaxies, len(gals)),
                              replace=False))

    ds = KLShearDataset(df, images_root, args.npix, args.use_original_image,
                        smooth_sigma=args.smooth_sigma,
                        smooth_target=args.smooth_target,
                        crop_frac=args.crop_frac)

    kpc_per_pix = 30.0 * args.crop_frac / args.npix
    panels = [("stellar image", "gray"), ("v_obs", "RdBu_r"),
              ("v_asym (g× carrier)", "RdBu_r"), ("v_sym (rotation)", "RdBu_r")]

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.1 * n), squeeze=False)
    for r, i in enumerate(idxs):
        photo, vel, lab = ds[int(i)]
        imgs = [photo[0].numpy(), vel[0].numpy(), vel[1].numpy(), vel[2].numpy()]
        row  = df.loc[int(i)]
        for c, ((title, cmap), img) in enumerate(zip(panels, imgs)):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(f"id {int(row['subhalo_id'])} (snap {int(row['snap'])})"
                             f"  g+={lab[0]:+.3f} g×={lab[1]:+.3f}\n{title}",
                             fontsize=8)
            else:
                ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Model inputs, smoothing σ = {args.smooth_sigma:g} px "
                 f"({args.smooth_sigma * kpc_per_pix:.2f} kpc) on "
                 f"{args.smooth_target}, crop {args.crop_frac:g}", fontsize=11)
    fig.tight_layout()
    fname = out_dir / f"inputs_sigma{args.smooth_sigma:g}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
