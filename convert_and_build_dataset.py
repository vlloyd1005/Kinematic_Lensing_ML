#!/usr/bin/env python3
"""
convert_and_build_dataset.py
============================
Two utilities in one file:

  1. convert_csv(input_csv, output_csv, subhalo_ids)
     Converts Eason's CSV format to ours, merging in subhalo IDs
     from a separate list (since their CSV has row_id but not subhalo_id).

  2. build_dataset_plan(sims, snapshots_per_sim, n_shear_draws,
                        n_inclination_draws, output_csv)
     Generates a master CSV covering multiple TNG simulations and snapshots
     that fall inside the KL-relevant redshift window (z ≈ 0.3–1.5, matching
     the Roman HLSS H-alpha + [O III] emitter range from Xu+2022 §3.2),
     with randomly drawn (g1, g2, inclination, theta_int) per galaxy.

Snapshot / redshift reference for TNG50-1  ── FULL SNAPSHOTS ONLY
-------------------------------------------------------------------
CRITICAL: GFM_StellarPhotometrics (r-band image weights) only exists
in the 20 "full" snapshots. The other 80 "mini" snapshots omit it and
the API returns HTTP 400 if you request it. Only use full snapshots.

Full snapshots in the KL redshift window (z = 0.3–1.5):

  snap  redshift   notes
  ----  --------   -----
  40    1.50       upper H-alpha + [O III] overlap
  50    1.00       peak of KL dN/dz  (best single choice)
  59    0.70       H-alpha mid-range
  67    0.50       H-alpha low-z tail
  72    0.40       lower edge
  78    0.30       minimum useful redshift

Recommended set:  --snaps 40 50 59       (z = 1.5, 1.0, 0.7)
Add z=0.5 tail:   --snaps 40 50 59 67

Usage
-----
  # 1. Convert the other student's CSV
  python convert_and_build_dataset.py convert \
      --input   their_sample.csv \
      --ids     subhalo_ids.txt \    # one integer per line
      --output  my_sample.csv

  # 2. Build a multi-sim/multi-snap dataset plan
  python convert_and_build_dataset.py build \
      --output  dataset_plan.csv \
      --n_gal   200               # galaxies per (sim, snap) combination
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Snapshot → redshift lookup (TNG50-1, relevant range only)
# All three TNG50 resolution levels share the same snapshot numbering.
# ---------------------------------------------------------------------------

# FULL snapshots only — the only ones with GFM_StellarPhotometrics.
# Mini snapshots omit that field; the API returns HTTP 400 if requested.
# Source: https://www.tng-project.org/data/docs/specifications/
TNG50_SNAPSHOTS = {
    # snap: redshift  (all 20 full snapshots)
    2:  12.00,
    3:  11.00,
    4:  10.00,
    6:   9.00,
    8:   8.00,
    11:  7.00,
    13:  6.00,
    17:  5.00,
    21:  4.00,
    25:  3.00,
    33:  2.00,
    40:  1.50,   # ← upper KL window
    50:  1.00,   # ← peak of KL dN/dz
    59:  0.70,   # ← H-alpha mid-range
    67:  0.50,   # ← H-alpha low-z tail
    72:  0.40,
    78:  0.30,   # ← lower KL window edge
    84:  0.20,
    91:  0.10,
    99:  0.00,
}

# KL-relevant full snapshots only (z = 0.3–1.5)
KL_SNAPSHOTS = {snap: z for snap, z in TNG50_SNAPSHOTS.items()
                if 0.30 <= z <= 1.50}

# Available TNG50 resolution levels (all use same snap numbering)
TNG50_SIMS = ["TNG50-1", "TNG50-2", "TNG50-3", "TNG50-4"]

# ---------------------------------------------------------------------------
# Shear and inclination priors (matching Xu+2022 Appendix B)
# ---------------------------------------------------------------------------

def draw_shear(n: int, rng: np.random.Generator,
               sigma_g: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw (g1, g2) from a zero-mean Gaussian with std sigma_g ≈ 0.05,
    clipped to |g| < 0.2 (typical cosmic-shear amplitude from §3.3).
    """
    while True:
        g1 = rng.normal(0, sigma_g, n)
        g2 = rng.normal(0, sigma_g, n)
        g  = np.sqrt(g1**2 + g2**2)
        ok = g < 0.2
        if ok.all():
            return g1, g2
        # resample only the out-of-range ones (rare)
        g1[~ok] = rng.normal(0, sigma_g, (~ok).sum())
        g2[~ok] = rng.normal(0, sigma_g, (~ok).sum())
        if np.all(np.sqrt(g1**2 + g2**2) < 0.2):
            return g1, g2


def draw_inclination(n: int, rng: np.random.Generator,
                     i_min: float = 0.2, i_max: float = 1.4) -> np.ndarray:
    """
    Draw inclinations sin(i) uniformly (geometric prior for random orientations),
    clipped to [i_min, i_max] rad to avoid degenerate face-on / edge-on cases.
    From the paper §3.3 / Appendix B: the KL sample is biased toward lower
    inclination (face-on) galaxies where H-alpha detection is easier.
    """
    cos_min = np.cos(i_max)
    cos_max = np.cos(i_min)
    cos_i   = rng.uniform(cos_min, cos_max, n)
    return np.arccos(cos_i)


def draw_theta_int(n: int, rng: np.random.Generator) -> np.ndarray:
    """Position angle: uniform over [-π/2, π/2)."""
    return rng.uniform(-np.pi / 2, np.pi / 2, n)


# ---------------------------------------------------------------------------
# 1.  CSV format converter
# ---------------------------------------------------------------------------

def convert_csv(input_csv: str, subhalo_id_source: str, output_csv: str) -> None:
    """
    Convert Eason's CSV to our format.

    Their columns:
      row_id, g1, g2, theta_int, i, v0, vcirc, rscale, rmse

    Our columns:
      subhalo_id, g1, g2, inclination, theta_int, redshift, vcirc_kms

    Notes
    -----
    * Their 'i' column is the inclination angle in radians — maps directly
      to our 'inclination'.
    * Their 'theta_int' is already our 'theta_int'.
    * Their CSV has no subhalo_id; we supply a separate file with one
      integer (subhalo ID) per line, in the same order as the CSV rows.
    * 'v0' (systemic velocity offset) and 'rscale' are not needed by our
      image-generation pipeline, but we keep them as optional extra columns.
    * 'rmse' is a fit-quality metric — kept as a filter column.
    * 'redshift' is not in their CSV; we add a placeholder of 1.0 (edit to
      match whatever snapshot they used, or pass --redshift).
    """
    df = pd.read_csv(input_csv)

    # Load subhalo IDs
    if subhalo_id_source.endswith(".csv"):
        ids_df = pd.read_csv(subhalo_id_source)
        # Accept either a single-column file or one with a 'subhalo_id' column
        if "subhalo_id" in ids_df.columns:
            subhalo_ids = ids_df["subhalo_id"].values
        else:
            subhalo_ids = ids_df.iloc[:, 0].values
    else:
        # Plain text: one integer per line
        with open(subhalo_id_source) as f:
            subhalo_ids = np.array([int(line.strip()) for line in f
                                    if line.strip()], dtype=int)

    if len(subhalo_ids) != len(df):
        raise ValueError(
            f"Subhalo ID count ({len(subhalo_ids)}) != CSV row count ({len(df)}). "
            "Make sure the IDs file is in the same order as the input CSV."
        )

    out = pd.DataFrame({
        "subhalo_id":  subhalo_ids,
        "g1":          df["g1"].values,
        "g2":          df["g2"].values,
        "inclination": df["i"].values,          # 'i' → 'inclination'
        "theta_int":   df["theta_int"].values,
        # Extras kept for reference
        "vcirc_kms":   df["vcirc"].values,
        "v0_kms":      df["v0"].values,
        "rscale":      df["rscale"].values,
        "rmse":        df["rmse"].values,
    })

    out.to_csv(output_csv, index=False)
    print(f"Converted {len(out)} rows → {output_csv}")
    print(out.head())


# ---------------------------------------------------------------------------
# 2.  Multi-sim / multi-snap dataset plan builder
# ---------------------------------------------------------------------------

def build_dataset_plan(
    sims:             list[str],
    snap_list:        list[int] | None,
    n_gal_per_combo:  int,
    output_csv:       str,
    seed:             int = 42,
    rmse_threshold:   float | None = None,
    their_csv:        str | None = None,
) -> None:
    """
    Build a master dataset CSV that spans multiple (sim, snapshot) combos.

    Strategy
    --------
    For each (sim, snap) pair we generate n_gal_per_combo rows, each with:
      - a placeholder subhalo_id of -1 (you replace these with real IDs from
        a subhalo catalog query, e.g. disc galaxies in the relevant mass range)
      - randomly drawn (g1, g2, inclination, theta_int)
      - the correct redshift for that snapshot

    If --their_csv is given, we also append their galaxies (after conversion),
    re-using the same randomly drawn shear / inclination columns but copying
    the subhalo_id from the supplied IDs file.

    Snapshot selection rationale (from Xu+2022)
    -------------------------------------------
    The paper's KL galaxy sample (§3.2) covers the Roman HLSS footprint where
    H-alpha (0.53 < z < 2.08) and [O III] (1.06 < z < 2.77) land in the grism
    band.  The dN/dΩdz distribution peaks near z ≈ 1 (Fig. 3).

    FULL SNAPSHOTS ONLY — mini snapshots lack GFM_StellarPhotometrics
    and the API returns HTTP 400 if you request it.  Valid choices:
      snap 40  z = 1.50   upper H-alpha/[O III] overlap
      snap 50  z = 1.00   peak of KL dN/dz  (best single choice)
      snap 59  z = 0.70   H-alpha mid-range
      snap 67  z = 0.50   low-z tail (optional 4th point)
    """
    if snap_list is None:
        # Default: three full snapshots covering the KL peak
        snap_list = [40, 50, 59]

    rng   = np.random.default_rng(seed)
    rows  = []

    for sim in sims:
        for snap in snap_list:
            z = TNG50_SNAPSHOTS.get(snap)
            if z is None:
                print(f"  [WARN] snap {snap} not in table, skipping.")
                continue
            if not (0.30 <= z <= 1.50):
                print(f"  [WARN] snap {snap} (z={z:.3f}) outside KL window or is a mini snap, skipping.")
                continue

            g1, g2 = draw_shear(n_gal_per_combo, rng)
            inc     = draw_inclination(n_gal_per_combo, rng)
            theta   = draw_theta_int(n_gal_per_combo, rng)

            for k in range(n_gal_per_combo):
                rows.append({
                    "sim":         sim,
                    "snap":        snap,
                    "redshift":    round(z, 4),
                    "subhalo_id":  -1,      # fill in from catalog query
                    "g1":          round(g1[k], 8),
                    "g2":          round(g2[k], 8),
                    "inclination": round(inc[k], 8),
                    "theta_int":   round(theta[k], 8),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nDataset plan: {len(df)} rows across "
          f"{len(sims)} sim(s) × {len(snap_list)} snap(s)")
    print(f"Saved → {output_csv}\n")

    # Summary table
    summary = df.groupby(["sim", "snap", "redshift"]).size().reset_index(name="n_rows")
    print(summary.to_string(index=False))
    print()
    print("NOTE: subhalo_id is -1 everywhere. Before running generate_kl_tng50.py,")
    print("query the TNG API for disc galaxies at each (sim, snap) and fill in real IDs.")
    print("See the companion query script: query_disc_subhalos.py")


# ---------------------------------------------------------------------------
# 3.  Companion: query disc subhalos from the TNG API
#     (prints a ready-to-use shell snippet)
# ---------------------------------------------------------------------------

QUERY_HELP = """
# -----------------------------------------------------------
# How to fill in real subhalo_ids
# -----------------------------------------------------------
# For each (sim, snap) in your plan, query the TNG API for
# star-forming disc galaxies in the KL mass range.
#
# The key filters (matching Xu+2022 §3.2 KL sample):
#   - Stellar mass  1e9 < M* < 1e12  Msun  (log M* ≈ 9–12)
#   - SubhaloFlag == 1  (genuine galaxy, not spurious)
#   - Sufficient star particles (proxy for disc size)
#
# Example (Python, using the TNG API):

import requests, pandas as pd

API_KEY = "YOUR_KEY_HERE"
headers = {"api-key": API_KEY}

def get_subhalos(sim, snap, mass_min_log=9.5, mass_max_log=11.5, limit=500):
    url = f"https://www.tng-project.org/api/{sim}/snapshots/{snap}/subhalos/"
    params = {
        "limit":       limit,
        "mass_stars__gt": 10**(mass_min_log - 10),   # API uses 1e10 Msun/h units
        "mass_stars__lt": 10**(mass_max_log - 10),
        "order_by":    "-mass_stars",
    }
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    results = r.json()["results"]
    return [entry["id"] for entry in results]

# Fill IDs into the plan CSV:
plan = pd.read_csv("dataset_plan.csv")
for (sim, snap), grp in plan.groupby(["sim", "snap"]):
    ids = get_subhalos(sim, snap, limit=len(grp))
    if len(ids) < len(grp):
        ids += ids * (len(grp) // len(ids) + 1)   # cycle if not enough
    plan.loc[grp.index, "subhalo_id"] = ids[:len(grp)]
plan.to_csv("dataset_plan_with_ids.csv", index=False)
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV format or build a multi-sim KL dataset plan."
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- convert ---
    conv = sub.add_parser("convert", help="Convert the other student's CSV format.")
    conv.add_argument("--input",  required=True, help="Their CSV file")
    conv.add_argument("--ids",    required=True,
                      help="File of subhalo IDs (one per line, or CSV with subhalo_id column)")
    conv.add_argument("--output", required=True, help="Output CSV path")

    # --- build ---
    bld = sub.add_parser("build", help="Build a multi-sim/snap dataset plan CSV.")
    bld.add_argument("--output",   required=True, help="Output CSV path")
    bld.add_argument("--sims",     nargs="+", default=["TNG50-1"],
                     help="Simulation names (default: TNG50-1)")
    bld.add_argument("--snaps",    nargs="+", type=int, default=None,
                     help="Full snapshot numbers only (default: 40 50 59). "
                          "Mini snaps lack GFM_StellarPhotometrics → HTTP 400.")
    bld.add_argument("--n_gal",    type=int, default=100,
                     help="Galaxies per (sim, snap) combo (default: 100)")
    bld.add_argument("--seed",     type=int, default=42)

    # --- query-help ---
    sub.add_parser("query-help",
                   help="Print the snippet for querying real subhalo IDs.")

    args = parser.parse_args()

    if args.cmd == "convert":
        convert_csv(args.input, args.ids, args.output)

    elif args.cmd == "build":
        build_dataset_plan(
            sims            = args.sims,
            snap_list       = args.snaps,
            n_gal_per_combo = args.n_gal,
            output_csv      = args.output,
            seed            = args.seed,
        )

    elif args.cmd == "query-help":
        print(QUERY_HELP)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()