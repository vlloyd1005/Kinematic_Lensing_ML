#!/usr/bin/env python3
"""
test_kl_pipeline.py
===================
End-to-end smoke test for the KL image-generation pipeline.

Runs the full chain for a tiny sample (default: 5 galaxies) without
requiring any pre-existing CSV.  Steps:

  1. Query TNG50-1 snapshot 50 (z ≈ 1.0, peak of KL dN/dz) for real
     star-forming disc galaxies in the KL mass range.
  2. Draw random (g1, g2, inclination, theta_int) for each galaxy.
  3. Download stellar + gas particle cutouts via the TNG web API.
  4. Render and save four FITS images per galaxy:
       *_image_original.fits   – unsheared r-band stellar image
       *_image_sheared.fits    – sheared stellar image
       *_velmap_original.fits  – unsheared LoS gas velocity map
       *_velmap_sheared.fits   – sheared LoS gas velocity map
  5. Save a summary CSV and a quick-look PNG contact sheet.

Usage
-----
  pip install numpy scipy astropy requests h5py pandas matplotlib tqdm

  python test_kl_pipeline.py --apikey YOUR_TNG_KEY
  python test_kl_pipeline.py --apikey YOUR_TNG_KEY --n_gal 10 --npix 128
  python test_kl_pipeline.py --apikey YOUR_TNG_KEY --snap 57  # z≈0.76

Verified TNG50-1 snapshot → redshift mapping (KL-relevant range):
  snap 41  z ≈ 1.41
  snap 42  z ≈ 1.36
  snap 50  z ≈ 1.00  ← peak of KL dN/dz  (default)
  snap 57  z ≈ 0.76
  snap 65  z ≈ 0.55
"""

import argparse
import io
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore")

# ── TNG cosmological constants ───────────────────────────────────────────────
H0  = 67.74
h   = H0 / 100.0
BASE_URL = "https://www.tng-project.org/api/"

# ── Snapshot → redshift (TNG50-1, verified against the downloads page) ───────
# Source: https://www.tng-project.org/data/downloads/TNG50-1/
# Full table lives in convert_and_build_dataset.py; we keep only the
# KL-relevant range (z ≈ 0.3–1.5) here for brevity.
SNAP_Z = {
    41: 1.41,
    42: 1.36,
    43: 1.30,
    44: 1.25,
    45: 1.21,
    46: 1.15,
    47: 1.11,
    48: 1.07,
    49: 1.04,
    50: 1.00,   # ← peak of KL dN/dz  (default)
    51: 0.95,
    52: 0.92,
    53: 0.89,
    54: 0.85,
    55: 0.82,
    56: 0.79,
    57: 0.76,
    58: 0.73,
    59: 0.70,
    60: 0.68,
    61: 0.64,
    62: 0.62,
    63: 0.60,
    64: 0.58,
    65: 0.55,
    66: 0.52,
    67: 0.50,
    68: 0.48,
    69: 0.46,
    70: 0.44,
    71: 0.42,
    72: 0.40,
    73: 0.38,
    74: 0.36,
    75: 0.35,
    76: 0.33,
    77: 0.31,
    78: 0.30,
    99: 0.00,   # z=0; useful for intrinsic-shape tests only
}


# ════════════════════════════════════════════════════════════════════════════
# TNG API helpers
# ════════════════════════════════════════════════════════════════════════════

def tng_get(url, params=None, headers=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 15))
                print(f"    rate-limited — waiting {wait}s …", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("max retries exceeded")


def query_disc_galaxies(sim, snap, headers, n=5,
                        log_mass_min=9.8, log_mass_max=11.2):
    """
    Return up to n subhalo IDs that are:
      • genuine galaxies (SubhaloFlag = 1 enforced server-side by the API)
      • in the KL stellar-mass range (log M* ≈ 9.8–11.2 Msun)
      • ordered by descending stellar mass for reproducibility

    The TNG API mass filter uses internal units of 1e10 Msun/h, so we convert.
    """
    mass_min = 10 ** (log_mass_min - 10) / h   # → code units
    mass_max = 10 ** (log_mass_max - 10) / h

    url    = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/"
    params = {
        "limit":           n * 3,          # fetch extra in case some lack gas
        "mass_stars__gt":  mass_min,
        "mass_stars__lt":  mass_max,
        "order_by":        "-mass_stars",
    }
    data = tng_get(url, params=params, headers=headers).json()
    ids  = [entry["id"] for entry in data.get("results", [])]
    print(f"  API returned {len(ids)} candidate subhalos (keeping up to {n})")
    return ids[:n * 3]   # return extras; we'll drop failures later


def get_subhalo_pos(sim, snap, subhalo_id, headers, scale_factor):
    """Return physical position (kpc) of the subhalo centre."""
    url  = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/"
    meta = tng_get(url, headers=headers).json()
    pos  = np.array([meta["pos_x"], meta["pos_y"], meta["pos_z"]],
                    dtype=np.float64)
    return pos * scale_factor / h   # comoving kpc/h → physical kpc


def download_cutout(sim, snap, subhalo_id, particle_type, fields, headers):
    """Stream an HDF5 cutout and return {field: array}. Returns None on failure."""
    pt_hdf5 = {"stars": "PartType4", "gas": "PartType0"}[particle_type]
    url      = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/cutout.hdf5"
    params   = {particle_type: ",".join(fields)}
    try:
        r = tng_get(url, params=params, headers=headers)
        with h5py.File(io.BytesIO(r.content), "r") as f:
            if pt_hdf5 not in f:
                return None
            return {field: f[pt_hdf5][field][:] for field in fields
                    if field in f[pt_hdf5]}
    except Exception as exc:
        print(f"    cutout error for {subhalo_id}: {exc}", flush=True)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Physics / geometry
# ════════════════════════════════════════════════════════════════════════════

def to_physical_coords(code, scale_factor):
    return code.astype(np.float64) * scale_factor / h

def to_physical_vel(code, scale_factor):
    return code.astype(np.float64) * np.sqrt(scale_factor)

def to_solar_mass(code):
    return code.astype(np.float64) * 1e10 / h

def centre(coords, pos_kpc, boxsize_kpc):
    dx = coords - pos_kpc
    return dx - boxsize_kpc * np.round(dx / boxsize_kpc)

def rotate_to_los(coords, vels, inclination, theta_int):
    """
    Rotate to image plane using R(phi=theta_int, i=inclination, psi=0).
    Returns x_im, y_im (kpc) and v_los (km/s).
    """
    phi, i_ = theta_int, inclination
    cp, sp   = np.cos(phi),  np.sin(phi)
    ci, si   = np.cos(i_),   np.sin(i_)

    R = np.array([
        [ cp,        sp,       0  ],
        [-ci*sp,     ci*cp,    si ],
        [ si*sp,    -si*cp,    ci ],
    ])

    r_rot = coords @ R.T
    v_rot = vels   @ R.T
    return r_rot[:, 0], r_rot[:, 1], -v_rot[:, 2]


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════

def particles_to_image(x, y, weights, npix, fov_kpc):
    """CIC deposit of weighted particles onto a square pixel grid."""
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix
    img  = np.zeros((npix, npix), dtype=np.float64)

    xi  = (x + half) / pix
    yi  = (y + half) / pix
    xi0 = np.floor(xi).astype(int)
    yi0 = np.floor(yi).astype(int)
    dx  = xi - xi0
    dy  = yi - yi0

    for ix, iy, tx, ty, w in zip(xi0, yi0, dx, dy, weights):
        for djx, wx in ((0, 1-tx), (1, tx)):
            for djy, wy in ((0, 1-ty), (1, ty)):
                ii, jj = ix+djx, iy+djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    img[jj, ii] += w * wx * wy
    return img


def particles_to_velmap(x, y, v_los, mass, npix, fov_kpc):
    """Mass-weighted mean LoS velocity per pixel. NaN where empty."""
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix
    num  = np.zeros((npix, npix))
    den  = np.zeros((npix, npix))

    xi  = (x + half) / pix
    yi  = (y + half) / pix
    xi0 = np.floor(xi).astype(int)
    yi0 = np.floor(yi).astype(int)
    dx  = xi - xi0
    dy  = yi - yi0

    for ix, iy, tx, ty, v, m in zip(xi0, yi0, dx, dy, v_los, mass):
        for djx, wx in ((0, 1-tx), (1, tx)):
            for djy, wy in ((0, 1-ty), (1, ty)):
                ii, jj = ix+djx, iy+djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    w = wx * wy * m
                    num[jj, ii] += w * v
                    den[jj, ii] += w

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def shear_image(img, g1, g2):
    """
    Remap source-plane image to observed plane via the lensing matrix A.
    A theta_O = theta_S  =>  we pull pixels from source coords.
    """
    npix   = img.shape[0]
    centre = (npix - 1) / 2.0
    col_o, row_o = np.meshgrid(np.arange(npix), np.arange(npix))
    xo = (col_o - centre) / centre
    yo = (row_o - centre) / centre
    # A = [[1-g1, -g2], [-g2, 1+g1]]
    xs = (1 - g1) * xo - g2       * yo
    ys = -g2      * xo + (1 + g1) * yo
    col_s = xs * centre + centre
    row_s = ys * centre + centre
    coords = np.array([row_s.ravel(), col_s.ravel()])
    return map_coordinates(img, coords, order=3,
                           mode="constant", cval=0.0).reshape(npix, npix)


def shear_velmap(gx, gy, v_los, mass, g1, g2, npix, fov_kpc):
    """Apply shear to gas particle coords then re-render velocity map."""
    g_sq   = g1**2 + g2**2
    factor = 1.0 / (1.0 - g_sq)
    gx_obs = factor * ((1 + g1) * gx + g2       * gy)
    gy_obs = factor * (g2       * gx + (1 - g1) * gy)
    return particles_to_velmap(gx_obs, gy_obs, v_los, mass, npix, fov_kpc)


# ════════════════════════════════════════════════════════════════════════════
# Per-galaxy pipeline
# ════════════════════════════════════════════════════════════════════════════

def process_one(subhalo_id, g1, g2, inclination, theta_int,
                sim, snap, scale_factor, boxsize_kpc,
                npix, fov_kpc, out_dir, headers):
    """
    Full pipeline for a single galaxy.
    Returns a dict of metadata on success, None on failure.
    """
    gal_dir = out_dir / f"galaxy_{subhalo_id}"
    gal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Subhalo position
    try:
        pos_kpc = get_subhalo_pos(sim, snap, subhalo_id, headers, scale_factor)
    except Exception as exc:
        print(f"    metadata error: {exc}", flush=True)
        return None

    # 2. Stellar cutout
    star_fields = ["Coordinates", "Velocities", "Masses",
                   "GFM_StellarPhotometrics", "GFM_StellarFormationTime"]
    star = download_cutout(sim, snap, subhalo_id, "stars", star_fields, headers)
    if star is None or len(star.get("Coordinates", [])) == 0:
        print("    no stellar data — skipping", flush=True)
        return None

    # Remove wind particles (GFM_StellarFormationTime <= 0)
    ok = star["GFM_StellarFormationTime"] > 0
    if ok.sum() < 50:
        print(f"    only {ok.sum()} real star particles — skipping", flush=True)
        return None
    for k in list(star.keys()):
        star[k] = star[k][ok]

    s_pos  = to_physical_coords(star["Coordinates"], scale_factor)
    s_vel  = to_physical_vel(star["Velocities"], scale_factor)
    s_mass = to_solar_mass(star["Masses"])

    # r-band luminosity: GFM_StellarPhotometrics columns are U,B,V,K,g,r,i,z
    phot   = star["GFM_StellarPhotometrics"].astype(np.float64)
    r_col  = 5 if phot.shape[1] >= 8 else 0
    r_lum  = 10.0 ** (-0.4 * phot[:, r_col])

    # 3. Gas cutout (velocity map)
    gas_fields = ["Coordinates", "Velocities", "Masses"]
    gas = download_cutout(sim, snap, subhalo_id, "gas", gas_fields, headers)
    have_gas = (gas is not None and len(gas.get("Coordinates", [])) > 10)

    if have_gas:
        g_pos  = to_physical_coords(gas["Coordinates"], scale_factor)
        g_vel  = to_physical_vel(gas["Velocities"], scale_factor)
        g_mass = to_solar_mass(gas["Masses"])

    # 4. Centre + subtract bulk velocity
    s_pos  = centre(s_pos,  pos_kpc, boxsize_kpc)
    v_bulk = np.average(s_vel, weights=s_mass, axis=0)
    s_vel -= v_bulk
    if have_gas:
        g_pos  = centre(g_pos, pos_kpc, boxsize_kpc)
        g_vel -= v_bulk

    # 5. Project to image plane
    sx, sy, _     = rotate_to_los(s_pos, s_vel, inclination, theta_int)
    if have_gas:
        gx, gy, gv = rotate_to_los(g_pos, g_vel, inclination, theta_int)

    # 6. Render images
    img_orig    = particles_to_image(sx, sy, r_lum, npix, fov_kpc)
    img_sheared = shear_image(img_orig, g1, g2)

    if have_gas:
        vmap_orig    = particles_to_velmap(gx, gy, gv, g_mass, npix, fov_kpc)
        vmap_sheared = shear_velmap(gx, gy, gv, g_mass, g1, g2, npix, fov_kpc)
    else:
        vmap_orig    = np.full((npix, npix), np.nan)
        vmap_sheared = np.full((npix, npix), np.nan)

    # 7. Save FITS
    hdr = fits.Header()
    hdr["SUBHALID"] = subhalo_id
    hdr["SIM"]      = sim
    hdr["SNAP"]     = snap
    hdr["REDSHIFT"] = round(1.0 / scale_factor - 1.0, 6)
    hdr["G1"]       = g1
    hdr["G2"]       = g2
    hdr["INCL_RAD"] = inclination
    hdr["THETAINT"] = theta_int
    hdr["FOV_KPC"]  = fov_kpc
    hdr["PIXSCALE"] = fov_kpc / npix
    hdr["NPIX"]     = npix

    def save(arr, name, bunit):
        h2 = hdr.copy()
        h2["BUNIT"] = bunit
        fits.PrimaryHDU(arr.astype(np.float32), header=h2).writeto(
            str(gal_dir / name), overwrite=True)

    save(img_orig,    f"galaxy_{subhalo_id}_image_original.fits",  "rel_lum")
    save(img_sheared, f"galaxy_{subhalo_id}_image_sheared.fits",   "rel_lum")
    save(vmap_orig,   f"galaxy_{subhalo_id}_velmap_original.fits", "km/s")
    save(vmap_sheared,f"galaxy_{subhalo_id}_velmap_sheared.fits",  "km/s")

    return dict(subhalo_id=subhalo_id, sim=sim, snap=snap,
                redshift=round(1/scale_factor-1, 4),
                g1=g1, g2=g2, inclination=inclination, theta_int=theta_int,
                n_stars=int(ok.sum()),
                n_gas=int(len(gas["Coordinates"])) if have_gas else 0,
                img_orig_peak=float(np.nanmax(img_orig)),
                vmap_max_kms=float(np.nanmax(np.abs(vmap_orig))))


# ════════════════════════════════════════════════════════════════════════════
# Contact sheet
# ════════════════════════════════════════════════════════════════════════════

def make_contact_sheet(results, out_dir, fov_kpc):
    """4-panel plot per galaxy: orig image | sheared | vel orig | vel sheared."""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n),
                             squeeze=False)
    titles = ["Original image", "Sheared image",
              "Vel map (orig)", "Vel map (sheared)"]
    cmaps  = ["inferno", "inferno", "RdBu_r", "RdBu_r"]

    for row, meta in enumerate(results):
        sid  = meta["subhalo_id"]
        gdir = out_dir / f"galaxy_{sid}"

        files = [
            gdir / f"galaxy_{sid}_image_original.fits",
            gdir / f"galaxy_{sid}_image_sheared.fits",
            gdir / f"galaxy_{sid}_velmap_original.fits",
            gdir / f"galaxy_{sid}_velmap_sheared.fits",
        ]

        for col, (fpath, title, cmap) in enumerate(zip(files, titles, cmaps)):
            ax = axes[row, col]
            if not fpath.exists():
                ax.set_visible(False)
                continue
            data = fits.getdata(str(fpath)).astype(float)
            if "vel" in fpath.name:
                vmax = np.nanpercentile(np.abs(data), 98)
                vmax = max(vmax, 10.0)
                ax.imshow(data, origin="lower", cmap=cmap,
                          vmin=-vmax, vmax=vmax)
            else:
                vmax = np.nanpercentile(data, 99.5)
                ax.imshow(np.log1p(data / max(vmax, 1e-10) * 100),
                          origin="lower", cmap=cmap)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"id={sid}\ng=({meta['g1']:.3f},{meta['g2']:.3f})\n"
                              f"i={meta['inclination']:.2f}rad", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"KL test sample  —  {meta['sim']} snap {meta['snap']} "
                 f"(z≈{meta['redshift']})  |  FoV {fov_kpc} kpc",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "contact_sheet.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nContact sheet saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end KL pipeline test on a small TNG50 sample."
    )
    p.add_argument("--apikey",  required=True,
                   help="TNG API key  (https://www.tng-project.org/users/profile/)")
    p.add_argument("--sim",     default="TNG50-1")
    p.add_argument("--snap",    type=int, default=50,
                   help="Snapshot number (default 50 = z≈1.0, peak of KL dN/dz)")
    p.add_argument("--n_gal",   type=int, default=5,
                   help="Number of galaxies to process (default 5)")
    p.add_argument("--npix",    type=int, default=128,
                   help="Image size in pixels (default 128; use 256 for full res)")
    p.add_argument("--fov_kpc", type=float, default=30.0,
                   help="Field of view in physical kpc (default 30)")
    p.add_argument("--outdir",  default="./kl_test_output")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    headers = {"api-key": args.apikey}
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── snapshot metadata ────────────────────────────────────────────────
    z = SNAP_Z.get(args.snap)
    if z is None:
        sys.exit(f"Snap {args.snap} not in SNAP_Z table. Add it or choose from: "
                 f"{sorted(SNAP_Z)}")
    scale_factor = 1.0 / (1.0 + z)
    print(f"\nSimulation : {args.sim}")
    print(f"Snapshot   : {args.snap}  →  z = {z:.3f},  a = {scale_factor:.4f}")

    # Box size from API
    sim_meta    = tng_get(f"{BASE_URL}{args.sim}/", headers=headers).json()
    boxsize_kpc = float(sim_meta["boxsize"]) * scale_factor / h * 1e3
    print(f"Box size   : {boxsize_kpc/1e3:.1f} Mpc physical\n")

    # ── query real subhalo IDs ───────────────────────────────────────────
    print(f"Querying {args.sim} snap {args.snap} for disc galaxies …")
    candidate_ids = query_disc_galaxies(
        args.sim, args.snap, headers, n=args.n_gal)

    if not candidate_ids:
        sys.exit("No subhalos returned by the API query. Check your API key "
                 "and that the simulation/snapshot exists.")

    # ── draw shear and orientation ───────────────────────────────────────
    def draw_shear(sigma=0.05):
        while True:
            g1, g2 = rng.normal(0, sigma), rng.normal(0, sigma)
            if (g1**2 + g2**2) < 0.04:   # |g| < 0.2
                return float(g1), float(g2)

    def draw_inclination():
        # sin(i) prior, avoid face-on (i<0.2) and edge-on (i>1.4)
        cos_i = rng.uniform(np.cos(1.4), np.cos(0.2))
        return float(np.arccos(cos_i))

    def draw_theta():
        return float(rng.uniform(-np.pi / 2, np.pi / 2))

    # ── run pipeline ─────────────────────────────────────────────────────
    results  = []
    attempts = 0

    for sid in candidate_ids:
        if len(results) >= args.n_gal:
            break
        attempts += 1
        g1, g2      = draw_shear()
        inclination = draw_inclination()
        theta_int   = draw_theta()

        print(f"\n[{len(results)+1}/{args.n_gal}] subhalo {sid}  "
              f"g=({g1:+.4f},{g2:+.4f})  i={inclination:.3f}rad  "
              f"θ={theta_int:.3f}rad", flush=True)

        meta = process_one(
            subhalo_id   = sid,
            g1=g1, g2=g2,
            inclination  = inclination,
            theta_int    = theta_int,
            sim          = args.sim,
            snap         = args.snap,
            scale_factor = scale_factor,
            boxsize_kpc  = boxsize_kpc,
            npix         = args.npix,
            fov_kpc      = args.fov_kpc,
            out_dir      = out_dir,
            headers      = headers,
        )

        if meta is not None:
            results.append(meta)
            print(f"    ✓  stars={meta['n_stars']}  gas={meta['n_gas']}  "
                  f"img_peak={meta['img_orig_peak']:.2e}  "
                  f"vmax={meta['vmap_max_kms']:.1f} km/s", flush=True)
        else:
            print("    ✗  skipped", flush=True)

        time.sleep(0.4)   # polite API delay

    # ── save summary CSV ─────────────────────────────────────────────────
    if results:
        df = pd.DataFrame(results)
        csv_path = out_dir / "test_sample.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"\nSummary CSV → {csv_path}")
        print(df[["subhalo_id","redshift","g1","g2",
                  "inclination","n_stars","n_gas",
                  "vmap_max_kms"]].to_string(index=False))
    else:
        print("\nNo galaxies successfully processed.")
        return

    # ── contact sheet ─────────────────────────────────────────────────────
    make_contact_sheet(results, out_dir, args.fov_kpc)

    print(f"\nDone.  {len(results)}/{attempts} galaxies succeeded.")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()