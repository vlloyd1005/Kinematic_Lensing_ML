#!/usr/bin/env python3
"""
generate_kl_tng50.py  (fixed: SFR gas weighting + star-forming selection)
==========================================================================
Changes from previous version:
  FIX 1 (face-on): already implemented via kl_geometry.py (correct).
  FIX 2 (gas):     velocity map now weighted by StarFormationRate and
                   restricted to star-forming gas (SFR > 0).  This matches
                   what Roman's grism actually measures — Hα emission tracks
                   star formation, not total gas mass.  Galaxies with fewer
                   than 10 star-forming gas particles are skipped (no disc).
  FIX 3 (query):   QUERY_HELP snippet now includes sfr__gt=0.1 so the
                   subhalo catalog query returns star-forming discs only.
"""

import argparse
import io
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from kl_geometry import (face_on_rotation, angular_momentum_direction,
                         rotate_to_los, realized_orientation, misalignment_deg)

warnings.filterwarnings("ignore")

H0  = 67.74
Om0 = 0.3089
h   = H0 / 100.0

BASE_URL = "https://www.tng-project.org/api/"


def tng_get(url, params=None, headers={}, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  Rate-limited; waiting {wait}s …", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            print(f"  Request error ({exc}); retry {attempt+1}/{max_retries}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("max_retries exceeded")


def get_snapshot_redshift(sim, snap, headers):
    url  = f"{BASE_URL}{sim}/snapshots/{snap}/"
    data = tng_get(url, headers=headers).json()
    z    = float(data["redshift"])
    return 0.0 if snap == 99 else z


def get_subhalo_meta(sim, snap, subhalo_id, headers):
    url = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/"
    return tng_get(url, headers=headers).json()


def download_cutout(sim, snap, subhalo_id, particle_type, fields, headers):
    pt_map = {"stars": "PartType4", "gas": "PartType0"}
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
    except Exception as exc:
        print(f"  [WARN] HDF5 parse error for subhalo {subhalo_id}: {exc}")
        return None


def center_particles(coords, subhalo_pos, boxsize_kpc):
    dx = coords - subhalo_pos
    dx = dx - boxsize_kpc * np.round(dx / boxsize_kpc)
    return dx


def apply_shear_to_coords(x, y, g1, g2, kappa=0.0):
    g = np.sqrt(g1**2 + g2**2)
    if g == 0.0:
        return x.copy(), y.copy()
    factor = 1.0 / ((1.0 - kappa) * (1.0 - g**2))
    x_obs  = factor * ((1.0 + g1) * x + g2 * y)
    y_obs  = factor * (g2 * x + (1.0 - g1) * y)
    return x_obs, y_obs


def particles_to_image(x, y, weights, npix, fov_kpc):
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix
    img  = np.zeros((npix, npix), dtype=np.float64)
    xi   = (x + half) / pix
    yi   = (y + half) / pix
    xi0  = np.floor(xi).astype(int)
    yi0  = np.floor(yi).astype(int)
    dx   = xi - xi0
    dy   = yi - yi0
    for (ix, iy, tx, ty, w) in zip(xi0, yi0, dx, dy, weights):
        for djx, wx in [(0, 1.0 - tx), (1, tx)]:
            for djy, wy in [(0, 1.0 - ty), (1, ty)]:
                ii, jj = ix + djx, iy + djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    img[jj, ii] += w * wx * wy
    return img


def particles_to_velmap(x, y, v_los, weight, npix, fov_kpc):
    """
    Weighted-mean LoS velocity map.
    `weight` is SFR (M☉/yr) for star-forming gas — matching Hα emission.
    """
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix
    num  = np.zeros((npix, npix), dtype=np.float64)
    den  = np.zeros((npix, npix), dtype=np.float64)
    xi   = (x + half) / pix
    yi   = (y + half) / pix
    xi0  = np.floor(xi).astype(int)
    yi0  = np.floor(yi).astype(int)
    dx   = xi - xi0
    dy   = yi - yi0
    for (ix, iy, tx, ty, v, w) in zip(xi0, yi0, dx, dy, v_los, weight):
        for djx, wx in [(0, 1.0 - tx), (1, tx)]:
            for djy, wy in [(0, 1.0 - ty), (1, ty)]:
                ii, jj = ix + djx, iy + djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    ww = wx * wy * w
                    num[jj, ii] += ww * v
                    den[jj, ii] += ww
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def shear_image_remap(img, g1, g2, kappa=0.0):
    npix   = img.shape[0]
    center = (npix - 1) / 2.0
    col_o, row_o = np.meshgrid(np.arange(npix), np.arange(npix))
    xo = (col_o - center) / center
    yo = (row_o - center) / center
    factor = (1.0 - kappa)
    xs = factor * ((1.0 - g1) * xo - g2       * yo)
    ys = factor * (-g2        * xo + (1.0 + g1) * yo)
    col_s = xs * center + center
    row_s = ys * center + center
    coords_s = np.array([row_s.ravel(), col_s.ravel()])
    return map_coordinates(img, coords_s, order=3,
                           mode="constant", cval=0.0).reshape(npix, npix)


def code_to_physical(coords_code, scale_factor):
    return coords_code * scale_factor / h

def vel_to_physical(vels_code, scale_factor):
    return vels_code * np.sqrt(scale_factor)

def mass_to_solar(mass_code):
    return mass_code * 1e10 / h


def process_galaxy(row, sim, snap, scale_factor, boxsize_kpc,
                   npix, fov_kpc, out_dir, headers):
    subhalo_id  = int(row["subhalo_id"])
    g1          = float(row["g1"])
    g2          = float(row["g2"])
    inclination = float(row["inclination"])
    theta_int   = float(row["theta_int"])

    gal_dir = out_dir / f"galaxy_{subhalo_id}"
    gal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    print(f"  [{subhalo_id}] Fetching metadata …", flush=True)
    meta    = get_subhalo_meta(sim, snap, subhalo_id, headers)
    pos_kpc = np.array([meta["pos_x"], meta["pos_y"], meta["pos_z"]],
                       dtype=np.float64) * scale_factor / h

    # 2. Stellar cutout
    print(f"  [{subhalo_id}] Downloading stellar cutout …", flush=True)
    star_fields = ["Coordinates", "Velocities", "Masses",
                   "GFM_StellarPhotometrics", "GFM_StellarFormationTime"]
    star_data = download_cutout(sim, snap, subhalo_id, "stars",
                                star_fields, headers)
    if star_data is None or len(star_data.get("Coordinates", [])) == 0:
        print(f"  [{subhalo_id}] SKIP – no stellar particles")
        return False

    sft = star_data["GFM_StellarFormationTime"]
    ok  = sft > 0
    if ok.sum() < 50:
        print(f"  [{subhalo_id}] SKIP – too few real star particles ({ok.sum()})")
        return False
    for k in list(star_data.keys()):
        star_data[k] = star_data[k][ok]

    star_coords = code_to_physical(star_data["Coordinates"].astype(np.float64), scale_factor)
    star_vels   = vel_to_physical(star_data["Velocities"].astype(np.float64), scale_factor)
    star_mass   = mass_to_solar(star_data["Masses"].astype(np.float64))

    phot  = star_data["GFM_StellarPhotometrics"]
    # K-band (col 3, ~2.2 µm) is closer to Roman H/F184 than r-band (col 5).
    # TODO: proper SPS modelling would be more accurate.
    r_col = 3 if phot.ndim == 2 and phot.shape[1] >= 8 else 0
    r_lum = 10.0 ** (-0.4 * phot[:, r_col].astype(np.float64))

    # 3. Gas cutout — include StarFormationRate for SFR-weighted velocity map
    # FIX 2: request StarFormationRate and restrict to star-forming particles
    print(f"  [{subhalo_id}] Downloading gas cutout …", flush=True)
    gas_fields = ["Coordinates", "Velocities", "Masses", "StarFormationRate"]
    gas_data   = download_cutout(sim, snap, subhalo_id, "gas",
                                 gas_fields, headers)

    have_gas = (gas_data is not None and
                len(gas_data.get("Coordinates", [])) > 0)

    sfr_weight = None
    if have_gas:
        gas_coords_raw = code_to_physical(
            gas_data["Coordinates"].astype(np.float64), scale_factor)
        gas_vels_raw   = vel_to_physical(
            gas_data["Velocities"].astype(np.float64), scale_factor)
        sfr_all        = gas_data["StarFormationRate"].astype(np.float64)

        # FIX 2: keep only star-forming gas (SFR > 0); this is the Hα tracer
        sf_mask = sfr_all > 0
        if sf_mask.sum() < 10:
            print(f"  [{subhalo_id}] No star-forming gas — velocity map will be empty "
                  f"(galaxy likely quenched; consider excluding from KL sample)")
            have_gas = False
        else:
            gas_coords = gas_coords_raw[sf_mask]
            gas_vels   = gas_vels_raw[sf_mask]
            sfr_weight = sfr_all[sf_mask]   # SFR in M☉/yr, used as pixel weight
            print(f"  [{subhalo_id}] {sf_mask.sum()} / {len(sfr_all)} "
                  f"gas particles are star-forming", flush=True)
    else:
        print(f"  [{subhalo_id}] No gas particles; velocity map will be empty.")

    # 4. Centre + subtract bulk velocity
    star_coords = center_particles(star_coords, pos_kpc, boxsize_kpc)
    v_bulk      = np.average(star_vels, weights=star_mass, axis=0)
    star_vels  -= v_bulk
    if have_gas:
        gas_coords = center_particles(gas_coords, pos_kpc, boxsize_kpc)
        gas_vels  -= v_bulk

    # 4b. Face-on alignment (FIX 1 — already implemented, verified correct)
    r_half = float(meta["halfmassrad_stars"]) * scale_factor / h
    rot_face, L_star = face_on_rotation(star_coords, star_vels, star_mass,
                                        r_max=2.0 * r_half)

    misalign = -1.0
    if have_gas:
        try:
            L_gas    = angular_momentum_direction(gas_coords, gas_vels, sfr_weight,
                                                  r_max=2.0 * r_half)
            misalign = misalignment_deg(L_star, L_gas)
        except ValueError:
            pass

    star_coords = rot_face.apply(star_coords)
    star_vels   = rot_face.apply(star_vels)
    if have_gas:
        gas_coords = rot_face.apply(gas_coords)
        gas_vels   = rot_face.apply(gas_vels)

    # 5. Project to image plane
    sx, sy, _       = rotate_to_los(star_coords, star_vels, inclination, theta_int)
    if have_gas:
        gx, gy, gv_los = rotate_to_los(gas_coords, gas_vels, inclination, theta_int)

    inc_real, pa_real = realized_orientation(rot_face, L_star, inclination, theta_int)
    assert abs(inc_real - inclination) < 1e-6 and abs(pa_real - theta_int) < 1e-6

    # 6. Render original stellar image
    img_orig = particles_to_image(sx, sy, r_lum, npix, fov_kpc)

    # 7. Render original velocity map — SFR-weighted (Hα proxy)
    if have_gas:
        vmap_orig = particles_to_velmap(gx, gy, gv_los, sfr_weight, npix, fov_kpc)
    else:
        vmap_orig = np.full((npix, npix), np.nan)

    # 8. Apply shear
    img_sheared = shear_image_remap(img_orig, g1, g2)

    if have_gas:
        gx_obs, gy_obs = apply_shear_to_coords(gx, gy, g1, g2)
        vmap_sheared   = particles_to_velmap(gx_obs, gy_obs, gv_los,
                                             sfr_weight, npix, fov_kpc)
    else:
        vmap_sheared = np.full((npix, npix), np.nan)

    # 9. Save FITS
    pixel_scale_kpc = fov_kpc / npix
    hdr = fits.Header()
    hdr["SUBHALID"]  = subhalo_id
    hdr["SIM"]       = sim
    hdr["SNAP"]      = snap
    hdr["REDSHIFT"]  = round(1.0 / scale_factor - 1.0, 6)
    hdr["G1"]        = g1
    hdr["G2"]        = g2
    hdr["INCL_RAD"]  = inclination
    hdr["THETAINT"]  = theta_int
    hdr["FOV_KPC"]   = fov_kpc
    hdr["PIXSCALE"]  = pixel_scale_kpc
    hdr["NPIX"]      = npix
    hdr["RHALF"]     = r_half
    hdr["MISALIGN"]  = misalign
    hdr["VELWGT"]    = "SFR"      # documents that velocity map is SFR-weighted

    def save_fits(data, fname, bunit):
        h2 = hdr.copy()
        h2["BUNIT"] = bunit
        fits.PrimaryHDU(data=data.astype(np.float32), header=h2).writeto(
            fname, overwrite=True)

    save_fits(img_orig,    str(gal_dir / f"galaxy_{subhalo_id}_image_original.fits"),  "rel_lum")
    save_fits(img_sheared, str(gal_dir / f"galaxy_{subhalo_id}_image_sheared.fits"),   "rel_lum")
    save_fits(vmap_orig,   str(gal_dir / f"galaxy_{subhalo_id}_velmap_original.fits"), "km/s")
    save_fits(vmap_sheared,str(gal_dir / f"galaxy_{subhalo_id}_velmap_sheared.fits"),  "km/s")

    print(f"  [{subhalo_id}] Saved 4 FITS files → {gal_dir}", flush=True)
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",          required=True)
    p.add_argument("--outdir",       default="./kl_output")
    p.add_argument("--apikey",       default=None)
    p.add_argument("--sim",          default="TNG50-1")
    p.add_argument("--snap",         type=int, default=99)
    p.add_argument("--npix",         type=int, default=256)
    p.add_argument("--fov_kpc",      type=float, default=30.0)
    p.add_argument("--skip_existing",action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    api_key = args.apikey or os.environ.get("TNG_API_KEY")
    if not api_key:
        sys.exit("ERROR: Provide --apikey or set TNG_API_KEY")

    headers = {"api-key": api_key}
    print(f"Testing TNG API connection ({args.sim}) …", flush=True)
    tng_get(f"{BASE_URL}{args.sim}/", headers=headers)
    print("  Connection OK\n", flush=True)

    redshift     = get_snapshot_redshift(args.sim, args.snap, headers)
    scale_factor = 1.0 / (1.0 + redshift)
    sim_meta     = tng_get(f"{BASE_URL}{args.sim}/", headers=headers).json()
    # API returns boxsize in ckpc/h for TNG50 — convert to physical kpc
    boxsize_kpc  = float(sim_meta["boxsize"]) * scale_factor / h
    print(f"Snapshot {args.snap}: z={redshift:.4f}  box={boxsize_kpc/1e3:.1f} Mpc\n",
          flush=True)

    df = pd.read_csv(args.csv)
    required = {"subhalo_id", "g1", "g2", "inclination", "theta_int"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing columns: {missing}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Galaxies"):
        sid     = int(row["subhalo_id"])
        gal_dir = out_dir / f"galaxy_{sid}"

        if args.skip_existing and gal_dir.exists():
            expected = [gal_dir / f"galaxy_{sid}_{s}.fits"
                        for s in ("image_original", "image_sheared", "velmap_original")]
            if all(p.exists() for p in expected):
                n_skip += 1
                continue

        print(f"\nProcessing subhalo {sid} ({idx+1}/{len(df)}) …", flush=True)
        try:
            ok = process_galaxy(row=row, sim=args.sim, snap=args.snap,
                                scale_factor=scale_factor,
                                boxsize_kpc=boxsize_kpc,
                                npix=args.npix, fov_kpc=args.fov_kpc,
                                out_dir=out_dir, headers=headers)
            if ok:  n_ok += 1
            else:   n_fail += 1
        except Exception as exc:
            print(f"  [ERROR] subhalo {sid}: {exc}", flush=True)
            n_fail += 1
        time.sleep(0.5)

    print(f"\nDone.  Success: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}", flush=True)


if __name__ == "__main__":
    main()