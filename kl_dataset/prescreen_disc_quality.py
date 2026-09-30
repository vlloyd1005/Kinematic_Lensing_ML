#!/usr/bin/env python3
"""
prescreen_disc_quality.py
=========================
Quickly estimate how many galaxies in a candidate plan CSV will pass the
disc quality cuts (κ_rot > 0.5, V_rot/σ > 1.0) WITHOUT running the full
image generation pipeline.

For each subhalo this script makes only 2 API calls:
  1. Subhalo metadata (halfmassrad_stars, mass_stars, sfr)
  2. Stellar particle cutout (Coordinates, Velocities, Masses,
     GFM_StellarFormationTime)

No gas cutout, no rendering, no FITS writing.  ~2-3s per galaxy vs ~30s
for the full pipeline.  Use this to calibrate how many candidates to fetch
before committing to kl_generate_images.sh.

Usage
-----
  # Screen the first 200 candidates at snap 50
  python prescreen_disc_quality.py \
      --csv   dataset_plan_with_ids.csv \
      --snap  50 \
      --n     200

  # Screen all candidates in the plan
  python prescreen_disc_quality.py \
      --csv   dataset_plan_with_ids.csv \
      --snap  50

Output
------
  prescreen_snap{snap}.csv  — one row per galaxy with κ_rot, V/σ, SFR,
                               log M*, and PASS/FAIL columns
  Prints a summary table showing pass rates vs κ_rot and V/σ thresholds
  so you can choose thresholds before running the full generator.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

# ── Copy the disc quality functions from generate_kl_tng50.py ────────────────
# (duplicated here so this script runs standalone without the full pipeline)

H0   = 67.74
h    = H0 / 100.0
BASE_URL = "https://www.tng-project.org/api/"


def tng_get(url, params=None, headers=None, max_retries=4):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=90)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"    rate-limited — waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("max_retries exceeded")


def download_cutout(sim, snap, subhalo_id, particle_type, fields, headers):
    pt_map = {"stars": "PartType4"}
    pt_str = pt_map[particle_type]
    url    = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/cutout.hdf5"
    params = {particle_type: ",".join(fields)}
    r = tng_get(url, params=params, headers=headers)
    try:
        with h5py.File(io.BytesIO(r.content), "r") as f:
            if pt_str not in f:
                return None
            return {field: f[pt_str][field][:] for field in fields
                    if field in f[pt_str]}
    except Exception:
        return None


def _angular_momentum_ez(mass, pos, vel, r_max=None):
    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        if idx.sum() == 0:
            raise ValueError("no particles within r_max")
        pos, vel, mass = pos[idx], vel[idx], mass[idx]
    J  = np.sum(mass[:, None] * np.cross(pos, vel), axis=0)
    Jn = np.linalg.norm(J)
    if Jn == 0:
        raise ValueError("zero angular momentum")
    return J / Jn


def kappa_rotation(mass, pos, vel, r_max=None):
    """κ_rot = E_rot / E_kin.  > 0.5 indicates a disc."""
    ez   = _angular_momentum_ez(mass, pos, vel, r_max=r_max)
    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        pos, vel, mass = pos[idx], vel[idx], mass[idx]
    ji     = np.cross(pos, vel)
    jz     = ji @ ez
    pos_z  = (pos @ ez)[:, None] * ez
    r_cyl  = np.maximum(np.linalg.norm(pos - pos_z, axis=1), 1e-9)
    E_rot  = 0.5 * np.sum(mass * (jz / r_cyl) ** 2)
    E_kin  = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
    return float(E_rot / E_kin) if E_kin > 0 else 0.0


def vrot_over_sigma(mass, pos, vel, r_max=None):
    """V_rot/σ within r_max.  Returns (ratio, v_rot, sigma)."""
    ez = _angular_momentum_ez(mass, pos, vel, r_max=r_max)
    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        if idx.sum() < 10:
            raise ValueError("too few particles")
        pos, vel, mass = pos[idx], vel[idx], mass[idx]
    pos_z  = (pos @ ez)[:, None] * ez
    e_phi  = np.cross(ez, pos - pos_z)
    norm   = np.linalg.norm(e_phi, axis=1)
    good   = norm > 1e-8
    e_phi[good]  /= norm[good, None]
    e_phi[~good]  = 0.0
    v_phi  = np.einsum("ij,ij->i", vel, e_phi)
    M_tot  = mass.sum()
    v_rot  = float(np.sum(mass * v_phi) / M_tot)
    v_mean = v_rot * e_phi
    sigma  = float(np.sqrt(np.sum(mass * np.sum((vel - v_mean)**2, axis=1)) / M_tot))
    ratio  = float(v_rot / sigma) if sigma > 0 else np.inf
    return ratio, v_rot, sigma


def screen_one(subhalo_id, sim, snap, scale_factor, headers,
               kappa_min=0.5, vrot_sigma_min=1.0, min_particles=100):
    """
    Download stellar particles for one galaxy and run disc quality checks.
    Returns a dict of measured properties.
    """
    result = {"subhalo_id": subhalo_id, "snap": snap, "sim": sim,
              "kappa_rot": np.nan, "vrot_sigma": np.nan,
              "vrot_kms": np.nan, "sigma_kms": np.nan,
              "n_star": 0, "r_half_kpc": np.nan,
              "log_mstar": np.nan, "sfr": np.nan,
              "passes_kappa": False, "passes_vsig": False,
              "passes_npart": False, "passes_all": False,
              "fail_reason": "not_run"}

    # 1. Metadata
    try:
        url  = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/"
        meta = tng_get(url, headers=headers).json()
        a    = scale_factor
        r_half = float(meta["halfmassrad_stars"]) * a / h   # physical kpc
        mstar  = float(meta["mass_stars"]) * 1e10 / h
        sfr    = float(meta.get("sfr", 0.0))
        result["r_half_kpc"] = r_half
        result["log_mstar"]  = np.log10(mstar) if mstar > 0 else np.nan
        result["sfr"]        = sfr
    except Exception as exc:
        result["fail_reason"] = f"metadata_error: {exc}"
        return result

    # 2. Stellar cutout (no gas needed for disc quality)
    try:
        star_data = download_cutout(
            sim, snap, subhalo_id, "stars",
            ["Coordinates", "Velocities", "Masses", "GFM_StellarFormationTime"],
            headers)
    except Exception as exc:
        result["fail_reason"] = f"cutout_error: {exc}"
        return result

    if star_data is None or len(star_data.get("Coordinates", [])) == 0:
        result["fail_reason"] = "no_stellar_particles"
        return result

    # Remove wind particles
    sft = star_data["GFM_StellarFormationTime"]
    ok  = sft > 0
    if ok.sum() == 0:
        result["fail_reason"] = "all_wind_particles"
        return result
    coords = star_data["Coordinates"][ok].astype(np.float64) * a / h
    vel   = star_data["Velocities"][ok].astype(np.float64) * np.sqrt(a)
    mass   = star_data["Masses"][ok].astype(np.float64) * 1e10 / h
    result["n_star"] = int(ok.sum())

    # Centre on CoM of particles within r_half (not global CoM)
    # and subtract velocity of particles within the same aperture.
    # Using the global bulk velocity is wrong: halo particles pull the mean
    # away from the disc, leaving a residual that inflates sigma within
    # any sub-aperture.
    r_norm = np.linalg.norm(coords, axis=1)
    inner  = r_norm < r_half
    if inner.sum() < 20:
        inner = np.ones(len(coords), dtype=bool)  # fallback: use all

    pos_com = np.average(coords[inner], weights=mass[inner], axis=0)
    coords -= pos_com
    v_bulk  = np.average(vel[inner],    weights=mass[inner], axis=0)
    vel    -= v_bulk

    # Disc quality checks at 1×r_half (not 2× — avoids stellar halo contamination)
    # Note: prescreener uses stellar V/σ only (no gas cutout to keep it fast).
    # The full generator uses SFR-weighted gas V/σ which will be cleaner.
    # Expect prescreener V/σ to be slightly conservative vs the full pipeline.
    r_cut = 1.0 * r_half

    # Particle count
    result["passes_npart"] = result["n_star"] >= min_particles
    if not result["passes_npart"]:
        result["fail_reason"] = f"n_star={result['n_star']} < {min_particles}"

    # κ_rot threshold: looser at high-z (discs more turbulent at z=1)
    z_local = 1.0 / scale_factor - 1.0
    kappa_min_z = float(np.clip(kappa_min + 0.1 * (1.0 - min(z_local, 1.0)),
                                kappa_min, 0.5))
    result["kappa_min_used"] = round(kappa_min_z, 3)

    # κ_rot
    try:
        kappa = kappa_rotation(mass, coords, vel, r_max=r_cut)
        result["kappa_rot"]    = round(kappa, 4)
        result["passes_kappa"] = kappa >= kappa_min_z
    except Exception as exc:
        result["fail_reason"] = f"kappa_error: {exc}"

    # V_rot/σ
    try:
        vs, vr, sg = vrot_over_sigma(mass, coords, vel, r_max=r_cut)
        result["vrot_sigma"]  = round(vs, 3)
        result["vrot_kms"]    = round(vr, 2)
        result["sigma_kms"]   = round(sg, 2)
        result["passes_vsig"] = vs >= vrot_sigma_min
    except Exception as exc:
        result["fail_reason"] = f"vsig_error: {exc}"

    result["passes_all"] = (result["passes_npart"] and
                            result["passes_kappa"] and
                            result["passes_vsig"])
    if result["passes_all"]:
        result["fail_reason"] = ""
    elif not result["fail_reason"] or result["fail_reason"] == "not_run":
        # Summarise which checks failed
        fails = []
        if not result["passes_kappa"]: fails.append(f"κ={result['kappa_rot']:.2f}")
        if not result["passes_vsig"]:  fails.append(f"V/σ={result['vrot_sigma']:.2f}")
        if not result["passes_npart"]: fails.append(f"n={result['n_star']}")
        result["fail_reason"] = "  ".join(fails)

    return result


def print_summary(df, kappa_min, vrot_sigma_min):
    """Print pass rates and a threshold sensitivity table."""
    n = len(df)
    n_pass = df["passes_all"].sum()
    print(f"\n{'═'*60}")
    print(f"Screened {n} galaxies at snap {df['snap'].iloc[0]}")
    print(f"{'═'*60}")
    print(f"  Pass all cuts  : {n_pass}/{n}  ({100*n_pass/n:.0f}%)")
    print(f"  Fail κ_rot     : {(~df['passes_kappa']).sum()}")
    print(f"  Fail V/σ       : {(~df['passes_vsig']).sum()}")
    print(f"  Fail n_star    : {(~df['passes_npart']).sum()}")
    print(f"  Errors         : {df['fail_reason'].str.contains('error').sum()}")

    # Distribution of κ_rot and V/σ for passing galaxies
    ok = df["passes_all"]
    if ok.any():
        print(f"\n  For {ok.sum()} PASSING galaxies:")
        print(f"    κ_rot   : "
              f"median={df.loc[ok,'kappa_rot'].median():.3f}  "
              f"min={df.loc[ok,'kappa_rot'].min():.3f}  "
              f"max={df.loc[ok,'kappa_rot'].max():.3f}")
        print(f"    V/σ     : "
              f"median={df.loc[ok,'vrot_sigma'].median():.2f}  "
              f"min={df.loc[ok,'vrot_sigma'].min():.2f}  "
              f"max={df.loc[ok,'vrot_sigma'].max():.2f}")
        print(f"    V_rot   : "
              f"median={df.loc[ok,'vrot_kms'].median():.1f} km/s")
        print(f"    σ       : "
              f"median={df.loc[ok,'sigma_kms'].median():.1f} km/s")

    # Threshold sensitivity table — how many would pass at different cuts?
    print(f"\n  Threshold sensitivity (κ_rot × V/σ):")
    print(f"  {'κ_min':>6}  {'V/σ_min':>7}  {'n_pass':>6}  {'pass%':>6}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")
    for km in [0.3, 0.4, 0.5, 0.6]:
        for vm in [0.5, 1.0, 1.5, 2.0]:
            n_p = ((df["kappa_rot"] >= km) &
                   (df["vrot_sigma"] >= vm) &
                   df["passes_npart"]).sum()
            marker = " ←" if (abs(km - kappa_min) < 0.01 and
                               abs(vm - vrot_sigma_min) < 0.01) else ""
            print(f"  {km:>6.1f}  {vm:>7.1f}  {n_p:>6}  "
                  f"{100*n_p/n:>5.0f}%{marker}")
    print(f"{'═'*60}\n")

    # Extrapolation: if you need N passing galaxies, fetch M candidates
    if n_pass > 0:
        pass_rate = n_pass / n
        print(f"  Pass rate = {pass_rate:.0%}.  "
              f"To get N passing galaxies, fetch N / {pass_rate:.2f} candidates.")
        for target in [100, 200, 300, 500]:
            needed = int(np.ceil(target / pass_rate))
            print(f"    target={target:4d} → fetch ≈ {needed:4d} candidates")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Pre-screen disc quality for TNG50 candidates without "
                    "running the full image generation pipeline.")
    p.add_argument("--csv",         required=True,
                   help="dataset_plan_with_ids.csv (must have subhalo_id, snap, sim)")
    p.add_argument("--snap",        type=int, required=True,
                   help="Which snapshot to screen")
    p.add_argument("--n",           type=int, default=None,
                   help="Max galaxies to screen (default: all in CSV for this snap)")
    p.add_argument("--sim",         default="TNG50-1")
    p.add_argument("--kappa_min",   type=float, default=0.4,
                   help="Minimum κ_rot (default 0.4; scaled up to 0.5 at z=0)")
    p.add_argument("--vrot_sigma_min", type=float, default=1.0)
    p.add_argument("--min_particles",  type=int,   default=100)
    p.add_argument("--output",      default=None,
                   help="Output CSV path (default: prescreen_snap{snap}.csv)")
    p.add_argument("--resume",      action="store_true",
                   help="Load existing output CSV and skip already-screened galaxies")
    args = p.parse_args()

    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        sys.exit("Set TNG_API_KEY environment variable")
    headers = {"api-key": api_key}

    # Snapshot redshift table (full snapshots only)
    SNAP_Z = {40: 1.50, 50: 1.00, 59: 0.70, 67: 0.50,
              72: 0.40, 78: 0.30, 84: 0.20, 91: 0.10, 99: 0.00}
    z = SNAP_Z.get(args.snap)
    if z is None:
        sys.exit(f"Snap {args.snap} not in table. Add it or choose from {sorted(SNAP_Z)}")
    scale_factor = 1.0 / (1.0 + z)
    print(f"Snap {args.snap}  z={z:.2f}  a={scale_factor:.4f}\n")

    # Load candidates for this snap
    df_plan = pd.read_csv(args.csv)
    df_plan = df_plan[(df_plan["snap"] == args.snap) &
                      (df_plan["subhalo_id"] != -1)]
    df_plan = df_plan.drop_duplicates("subhalo_id").reset_index(drop=True)
    if args.n:
        df_plan = df_plan.head(args.n)
    print(f"Candidates to screen: {len(df_plan)}")

    out_path = Path(args.output or f"prescreen_snap{args.snap}.csv")

    # Resume: skip already-screened galaxies
    done_ids = set()
    existing_rows = []
    if args.resume and out_path.exists():
        df_done = pd.read_csv(out_path)
        done_ids = set(df_done["subhalo_id"].tolist())
        existing_rows = df_done.to_dict("records")
        print(f"Resuming: {len(done_ids)} already screened")

    results = list(existing_rows)
    todo    = df_plan[~df_plan["subhalo_id"].isin(done_ids)]
    print(f"Remaining: {len(todo)}\n")

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        sid = int(row["subhalo_id"])
        sim = str(row.get("sim", args.sim))
        print(f"[{i}/{len(todo)}] subhalo {sid} …", end="  ", flush=True)

        result = screen_one(sid, sim, args.snap, scale_factor, headers,
                            args.kappa_min, args.vrot_sigma_min,
                            args.min_particles)
        results.append(result)

        status = (f"κ={result['kappa_rot']:.3f}  "
                  f"V/σ={result['vrot_sigma']:.2f}  "
                  f"n={result['n_star']}  "
                  f"{'✓' if result['passes_all'] else '✗ ' + result['fail_reason']}")
        print(status, flush=True)

        # Save incrementally every 10 galaxies
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

        time.sleep(0.3)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    print_summary(df_out, args.kappa_min, args.vrot_sigma_min)


if __name__ == "__main__":
    main()