#!/usr/bin/env python3
"""
expand_shear_draws.py
=====================
Takes an existing dataset_plan_with_ids.csv and expands it to N draws per
subhalo by sampling new (g1, g2, inclination, theta_int) for each draw.

Since inclination and PA change per draw, each draw produces a genuinely
different projected image — the face-on alignment in generate_kl_tng50.py
always starts from the stellar angular momentum frame, then applies whatever
(inclination, theta_int) the CSV row specifies.  This means you CAN reuse
the same galaxy without new API calls by re-running generate_kl_tng50.py
on the expanded CSV: the particle data is already on disk as FITS originals,
but the rendered images need to be regenerated for each new viewing angle.

Workflow
--------
  # 1. Expand the CSV with new draws (varies shear + inclination + PA)
  python expand_shear_draws.py \
      --input   dataset_plan_with_ids.csv \
      --output  dataset_plan_expanded_10draws.csv \
      --n_draws 10 \
      --seed    42

  # 2. Regenerate FITS for all draws.
  #    The generator reads (inclination, theta_int, g1, g2) from the CSV
  #    and applies them to the already-downloaded particle data.
  #    Use --outdir with draw_idx subdirectories (handled automatically).
  sbatch kl_generate_images.sh   # point PLAN_CSV at the expanded CSV

Note on directory naming
------------------------
Each draw gets its own output directory:
    snap{N}/galaxy_{sid}_draw{k:04d}/
so draw 0 is the same projection as the original single-draw FITS,
draw 1–9 are new projections.  The training script resolves paths via
_gal_dir(images_root, snap, sid, draw_idx) which is already in train_kl_model.py.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def draw_shear_batch(n, rng, sigma_g=0.05):
    """Draw (g1, g2) pairs with |g| < 0.2."""
    g1 = np.zeros(n); g2 = np.zeros(n)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        nr = remaining.sum()
        g1[remaining] = rng.normal(0, sigma_g, nr)
        g2[remaining] = rng.normal(0, sigma_g, nr)
        remaining = np.sqrt(g1**2 + g2**2) >= 0.2
    return g1, g2


def draw_inclination_batch(n, rng, i_min=0.2, i_max=1.4):
    """
    Geometric (sin i) prior, clipped to [i_min, i_max] rad.
    i_min = 0.2 rad (11°): face-on limit — v_minor → 0, g× unmeasurable.
    i_max = 1.4 rad (80°): edge-on limit — Hα extincted by dust lane.
    """
    cos_i = rng.uniform(np.cos(i_max), np.cos(i_min), n)
    return np.arccos(cos_i)


def draw_theta_batch(n, rng):
    """Position angle uniform over [-π, π)."""
    return rng.uniform(-np.pi, np.pi, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True,
                   help="Existing dataset_plan_with_ids.csv")
    p.add_argument("--output",      required=True,
                   help="Output expanded CSV path")
    p.add_argument("--n_draws",     type=int, default=10,
                   help="Draws per (subhalo, snap) pair (default: 10)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--images_root", default=None,
                   help="Optional: filter to subhalos with existing "
                        "image_original.fits on disk")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    df  = pd.read_csv(args.input)
    df  = df[df["subhalo_id"] != -1].reset_index(drop=True)
    print(f"Input: {len(df)} rows, {df['subhalo_id'].nunique()} unique subhalos")

    if args.images_root:
        root = Path(args.images_root)
        mask = []
        for _, row in df.iterrows():
            sid, snap = int(row["subhalo_id"]), int(row["snap"])
            mask.append(
                (root / f"snap{snap}" / f"galaxy_{sid}" /
                 f"galaxy_{sid}_image_original.fits").exists()
            )
        df = df[mask].reset_index(drop=True)
        print(f"After filtering to existing originals: {len(df)} rows")

    # One base row per unique (subhalo_id, snap)
    base = df.drop_duplicates(subset=["subhalo_id", "snap"]).reset_index(drop=True)
    print(f"Unique (subhalo, snap) pairs: {len(base)}")

    rows = []
    for _, row in base.iterrows():
        g1_arr    = np.zeros(args.n_draws)
        g2_arr    = np.zeros(args.n_draws)
        g1_arr, g2_arr = draw_shear_batch(args.n_draws, rng)
        inc_arr   = draw_inclination_batch(args.n_draws, rng)
        theta_arr = draw_theta_batch(args.n_draws, rng)

        for k in range(args.n_draws):
            new_row = row.to_dict()
            new_row["g1"]          = round(float(g1_arr[k]),    8)
            new_row["g2"]          = round(float(g2_arr[k]),    8)
            new_row["inclination"] = round(float(inc_arr[k]),   8)
            new_row["theta_int"]   = round(float(theta_arr[k]), 8)
            new_row["draw_idx"]    = k
            rows.append(new_row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print(f"\nExpanded: {len(out)} rows "
          f"({len(base)} galaxies × {args.n_draws} draws)")
    print(f"Saved → {args.output}")
    print()
    print("Each draw has a unique (g1, g2, inclination, theta_int).")
    print("The generator will write to galaxy_{sid}_draw{k:04d}/ directories.")
    print("Run generate_kl_tng50.py on this CSV to render all FITS files.")

    # Summary statistics of drawn parameters
    print("\nDrawn parameter ranges:")
    for col, label in [("g1", "g1"), ("g2", "g2"),
                       ("inclination", "incl (rad)"),
                       ("theta_int",   "theta_int (rad)")]:
        if col in out:
            print(f"  {label:<15} "
                  f"min={out[col].min():+.3f}  "
                  f"mean={out[col].mean():+.3f}  "
                  f"max={out[col].max():+.3f}")


if __name__ == "__main__":
    main()