#!/usr/bin/env python3
"""
expand_shear_draws.py
=====================
Takes an existing dataset_plan_with_ids.csv (which has one shear draw per
subhalo) and expands it to N draws per subhalo by adding new (g1, g2,
theta_int, inclination) samples for each galaxy.

This is the correct way to scale up the training set cheaply:
  - No new API calls needed (FITS images already exist on disk)
  - No data leakage (GroupShuffleSplit keeps all rows for a subhalo together)
  - Each row is a genuinely different training example (different shear)

Eason's approach: 50 galaxies × 10,000 shear draws = 500,000 rows
Your approach after expansion: ~780 galaxies × N draws

The generate_kl_tng50.py script only needs to be re-run for NEW subhalos.
For additional shear draws on existing galaxies, the training script handles
everything at load time — it applies the shear transformation in the Dataset
class, so you only need new CSV rows, not new FITS files.

HOWEVER: there is an important subtlety. Your current generate_kl_tng50.py
saves galaxy_{ID}_image_SHEARED.fits with the shear baked in at generation
time. This means each new shear draw DOES need a new FITS file.

Two options:
  Option A (recommended): switch training to apply shear at load time from
    the ORIGINAL (unsheared) image. This way you only need one FITS file per
    galaxy and can draw infinite shear values. See --mode augment below.

  Option B: re-run generate_kl_tng50.py for each new draw (expensive, many
    API calls, much slower).

This script implements Option A: it generates the expanded CSV and also
prints instructions for updating the training script to apply shear on-the-fly.

Usage
-----
  python expand_shear_draws.py \
      --input   dataset_plan_with_ids.csv \
      --output  dataset_plan_expanded.csv \
      --n_draws 20 \
      --seed    42

  # This gives you ~780 × 20 = ~15,600 training rows
  # using only the FITS files you already have on disk.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def draw_shear_batch(n: int, rng: np.random.Generator,
                     sigma_g: float = 0.05) -> tuple:
    """Draw (g1, g2) pairs, rejecting |g| >= 0.2."""
    g1 = np.zeros(n); g2 = np.zeros(n)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        nr = remaining.sum()
        g1[remaining] = rng.normal(0, sigma_g, nr)
        g2[remaining] = rng.normal(0, sigma_g, nr)
        remaining = np.sqrt(g1**2 + g2**2) >= 0.2
    return g1, g2

### To Do: currently uses pixel mapping for shear, but I think switching 
### to particle mapping would be more accurate
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True,
                   help="Existing dataset_plan_with_ids.csv")
    p.add_argument("--output",   required=True,
                   help="Output expanded CSV path")
    p.add_argument("--n_draws",  type=int, default=20,
                   help="Total shear draws per subhalo (default: 20). "
                        "The original draw counts as 1, so n_draws=20 adds 19 new rows.")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--images_root", default=None,
                   help="Optional: check that image_original.fits exists for each subhalo")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    df  = pd.read_csv(args.input)
    df  = df[df["subhalo_id"] != -1].reset_index(drop=True)

    print(f"Input: {len(df)} rows, {df['subhalo_id'].nunique()} unique subhalos")

    # Optionally filter to subhalos that have their original FITS on disk
    if args.images_root:
        root = Path(args.images_root)
        has_file = []
        for _, row in df.iterrows():
            sid  = int(row["subhalo_id"])
            snap = int(row["snap"])
            orig = root / f"snap{snap}" / f"galaxy_{sid}" / \
                   f"galaxy_{sid}_image_original.fits"
            has_file.append(orig.exists())
        df = df[has_file].reset_index(drop=True)
        print(f"After filtering to existing originals: {len(df)} rows")

    # Deduplicate to one row per unique (subhalo_id, snap) — we'll expand from there
    base = df.drop_duplicates(subset=["subhalo_id", "snap"]).reset_index(drop=True)
    print(f"Unique (subhalo, snap) pairs: {len(base)}")

    rows = []
    for _, row in base.iterrows():
        g1_arr, g2_arr = draw_shear_batch(args.n_draws, rng)
        inc_arr   = np.arccos(rng.uniform(np.cos(1.4), np.cos(0.2), args.n_draws))
        theta_arr = rng.uniform(-np.pi / 2, np.pi / 2, args.n_draws)

        for k in range(args.n_draws):
            new_row = row.to_dict()
            new_row["g1"]          = round(float(g1_arr[k]),   8)
            new_row["g2"]          = round(float(g2_arr[k]),   8)
            new_row["inclination"] = round(float(inc_arr[k]),  8)
            new_row["theta_int"]   = round(float(theta_arr[k]),8)
            new_row["draw_idx"]    = k
            rows.append(new_row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print(f"\nExpanded: {len(out)} rows ({len(base)} galaxies × {args.n_draws} draws)")
    print(f"Saved → {args.output}")
    print()
    print("=" * 60)
    print("IMPORTANT: update your training script to use")
    print("image_original.fits + on-the-fly shear (see note below)")
    print("=" * 60)
    print("""
The expanded CSV has new (g1, g2) values per row, but your existing
FITS files have shear baked in at generation time. To use the expanded
CSV correctly, update KLShearDataset in train_kl_model.py to:

  1. Load galaxy_{ID}_image_ORIGINAL.fits  (not sheared)
  2. Apply shear at load time using shear_image_tensor()

Replace the photo_path line in __getitem__ with:
    photo_path = gal_dir / f"galaxy_{sid}_image_original.fits"

And after loading, apply shear:
    photo = apply_shear_numpy(photo, float(row['g1']), float(row['g2']))

where apply_shear_numpy uses scipy.ndimage.map_coordinates as in
generate_kl_tng50.py's shear_image_remap() function.

This gives you unlimited shear augmentation with zero new API calls.
""")


if __name__ == "__main__":
    main()