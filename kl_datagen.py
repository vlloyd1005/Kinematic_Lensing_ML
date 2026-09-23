#!/usr/bin/env python3
"""
generate_kl_tng50.py
====================
Synthesised data-generation script for Kinematic Lensing (KL) with TNG50.

For every galaxy listed in the input CSV this script:
  1.  Downloads the stellar + gas particle cutout from the TNG web API
      (no full simulation download required).
  2.  Builds a stellar-light image (r-band, GFM_StellarPhotometrics) and
      a LoS-velocity map from the gas particles.
  3.  Applies cosmic-shear distortion (g1, g2) following Eq. 1 / 10 of
      Xu et al. 2022 (arXiv:2201.00739) to produce the "sheared" image and
      a sheared velocity map.
  4.  Saves three FITS files per galaxy:
        galaxy_{ID}_image_original.fits   – unsheared stellar image
        galaxy_{ID}_image_sheared.fits    – sheared stellar image
        galaxy_{ID}_velmap_original.fits  – unsheared LoS velocity map

Usage
-----
  python generate_kl_tng50.py \
      --csv   sample.csv \
      --outdir ./output \
      --apikey YOUR_TNG_API_KEY \
      --snap   99 \
      --sim    TNG50-1 \
      --npix   256 \
      --fov_kpc 30

CSV format (required columns)
------------------------------
  subhalo_id   – TNG50-1 subhalo index
  g1           – shear component 1
  g2           – shear component 2
  inclination  – inclination angle [rad]  (used to orient the velocity map)
  theta_int    – intrinsic position angle [rad]

Optional columns (used when present):
  redshift, vcirc_kms

Dependencies
------------
  pip install numpy scipy astropy requests h5py pandas tqdm
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# TNG cosmological parameters (TNG50-1 / TNG50 runs share these)
# ---------------------------------------------------------------------------
H0   = 67.74   # km/s/Mpc
Om0  = 0.3089
OL0  = 0.6911
h    = H0 / 100.0

# ---------------------------------------------------------------------------
# Helpers: TNG web API
# ---------------------------------------------------------------------------

BASE_URL = "https://www.tng-project.org/api/"


def tng_get(url: str, params: dict | None = None, headers: dict = {},
            stream: bool = False, max_retries: int = 5) -> requests.Response:
    """Thin wrapper around requests.get with retry logic."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, stream=stream,
                             timeout=120)
            if r.status_code == 429:               # rate-limited
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


def get_snapshot_redshift(sim: str, snap: int, headers: dict) -> float:
    url = f"{BASE_URL}{sim}/snapshots/{snap}/"
    data = tng_get(url, headers=headers).json()
    z = float(data["redshift"])
    if snap == 99:          # TNG returns a tiny floating-point residual at z=0
        z = 0.0
    return z


def get_subhalo_meta(sim: str, snap: int, subhalo_id: int,
                     headers: dict) -> dict:
    """Return the subhalo JSON metadata dict."""
    url = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{subhalo_id}/"
    return tng_get(url, headers=headers).json()


def download_cutout(sim: str, snap: int, subhalo_id: int,
                    particle_type: str, fields: list[str],
                    headers: dict) -> dict | None:
    """
    Download a particle cutout via the TNG API and return a dict of arrays.

    particle_type: 'stars' or 'gas'
    fields:        list of HDF5 field names, e.g. ['Coordinates', 'Velocities']
    """
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


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def center_particles(coords: np.ndarray, subhalo_pos: np.ndarray,
                     boxsize_kpc: float) -> np.ndarray:
    """Centre particle positions on subhalo and correct for periodic wrap."""
    dx = coords - subhalo_pos
    dx = dx - boxsize_kpc * np.round(dx / boxsize_kpc)
    return dx


def rotate_to_los(coords: np.ndarray, vels: np.ndarray,
                  inclination: float, theta_int: float):
    """
    Rotate coordinates and velocities so that the galaxy disc lies in the
    image plane with the specified inclination and position angle.

    Parameters
    ----------
    coords      : (N, 3) physical kpc, centred on galaxy
    vels        : (N, 3) km/s (physical)
    inclination : inclination angle i  [rad]  (0 = face-on)
    theta_int   : position angle phi   [rad]

    Returns
    -------
    x_im, y_im : projected image-plane coordinates (kpc)
    v_los      : line-of-sight velocities (km/s, positive = receding)
    """
    # --- rotation matrix following Appendix A of Xu+2022 ---
    # We build R(phi, i, psi=0) explicitly.
    phi = theta_int
    i_  = inclination
    psi = 0.0

    cp, sp = np.cos(phi), np.sin(phi)
    ci, si = np.cos(i_),  np.sin(i_)
    cps, sps = np.cos(psi), np.sin(psi)

    R = np.array([
        [ cp*cps - ci*sp*sps,  -cp*sps - ci*sp*cps,  si*sp],
        [ sp*cps + ci*cp*sps,  -sp*sps + ci*cp*cps, -si*cp],
        [ si*sps,               si*cps,               ci   ],
    ])  # shape (3,3)

    # Project positions and velocities
    r_rot = coords @ R.T           # (N, 3) in source-plane frame
    v_rot = vels   @ R.T

    x_im  = r_rot[:, 0]            # RA-like axis (XS)
    y_im  = r_rot[:, 1]            # Dec-like axis (YS)
    v_los = -v_rot[:, 2]           # LoS velocity: sign matches Eq. A13

    return x_im, y_im, v_los


# ---------------------------------------------------------------------------
# Shear transformation  (Xu+2022 Eq. 1)
# ---------------------------------------------------------------------------

def apply_shear_to_coords(x: np.ndarray, y: np.ndarray,
                           g1: float, g2: float,
                           kappa: float = 0.0):
    """
    Map source-plane coordinates (xS, yS) = A · (xO, yO) to
    observe-plane coordinates given reduced shear (g1, g2).

    We invert: given source-plane (x, y) we want observed-plane coords.
    A = (1-kappa) [[1-g+, -gx],[-gx, 1+g+]]

    Since shear is small, A^{-1} ≈ I + [[g+, gx],[gx, -g+]] / (1-kappa).
    For lensing data-generation we go source → image, i.e. apply A^{-1}.
    """
    g = np.sqrt(g1**2 + g2**2)
    if g == 0.0:
        return x.copy(), y.copy()

    # Reduced shear: assume kappa=0 for cosmic-shear regime (weak lensing)
    # Full transformation: theta_O = A^{-1} theta_S
    # A^{-1} = 1/(1-kappa) * 1/(1-g^2) * [[1+g+, gx],[gx, 1-g+]]
    factor = 1.0 / ((1.0 - kappa) * (1.0 - g**2))
    x_obs = factor * ((1.0 + g1) * x + g2 * y)
    y_obs = factor * (g2 * x + (1.0 - g1) * y)
    return x_obs, y_obs


# ---------------------------------------------------------------------------
# Image rendering: scatter particles onto a 2-D pixel grid
# ---------------------------------------------------------------------------

def particles_to_image(x: np.ndarray, y: np.ndarray,
                       weights: np.ndarray,
                       npix: int, fov_kpc: float) -> np.ndarray:
    """
    Bin particles into a 2-D image using a simple CIC (Cloud-In-Cell) scheme.

    Parameters
    ----------
    x, y    : projected coordinates (kpc) centred on galaxy
    weights : per-particle weight (e.g. stellar luminosity or mass)
    npix    : output image side length in pixels
    fov_kpc : total field of view in kpc (image spans ±fov_kpc/2)

    Returns
    -------
    img : (npix, npix) float64 image
    """
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix               # kpc per pixel

    # Convert to pixel coordinates [0, npix)
    xi = (x + half) / pix
    yi = (y + half) / pix

    # CIC deposit
    img = np.zeros((npix, npix), dtype=np.float64)
    xi0 = np.floor(xi).astype(int)
    yi0 = np.floor(yi).astype(int)
    dx  = xi - xi0
    dy  = yi - yi0

    for (ix, iy, tx, ty, w) in zip(xi0, yi0, dx, dy, weights):
        for djx, wx in [(0, 1.0 - tx), (1, tx)]:
            for djy, wy in [(0, 1.0 - ty), (1, ty)]:
                ii, jj = ix + djx, iy + djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    img[jj, ii] += w * wx * wy   # (row=y, col=x)

    return img


def particles_to_velmap(x: np.ndarray, y: np.ndarray,
                        v_los: np.ndarray, mass: np.ndarray,
                        npix: int, fov_kpc: float) -> np.ndarray:
    """
    Build a mass-weighted LoS velocity map.  Returns the mean LoS velocity
    per pixel in km/s.  Pixels with no particles are NaN.
    """
    half = fov_kpc / 2.0
    pix  = fov_kpc / npix

    xi  = (x + half) / pix
    yi  = (y + half) / pix

    num = np.zeros((npix, npix), dtype=np.float64)   # sum(mass * v_los)
    den = np.zeros((npix, npix), dtype=np.float64)   # sum(mass)

    xi0 = np.floor(xi).astype(int)
    yi0 = np.floor(yi).astype(int)
    dx  = xi - xi0
    dy  = yi - yi0

    for (ix, iy, tx, ty, v, m) in zip(xi0, yi0, dx, dy, v_los, mass):
        for djx, wx in [(0, 1.0 - tx), (1, tx)]:
            for djy, wy in [(0, 1.0 - ty), (1, ty)]:
                ii, jj = ix + djx, iy + djy
                if 0 <= ii < npix and 0 <= jj < npix:
                    w = wx * wy * m
                    num[jj, ii] += w * v
                    den[jj, ii] += w

    with np.errstate(invalid="ignore", divide="ignore"):
        vmap = np.where(den > 0, num / den, np.nan)
    return vmap


def shear_image_remap(img: np.ndarray,
                      g1: float, g2: float,
                      kappa: float = 0.0) -> np.ndarray:
    """
    Remap a source-plane image to the observed plane by applying the
    lensing transformation at the pixel level.

    We use scipy map_coordinates for sub-pixel interpolation.
    """
    npix = img.shape[0]
    assert img.shape[0] == img.shape[1], "Image must be square"

    centre = (npix - 1) / 2.0

    # Build output-plane grid
    col_o, row_o = np.meshgrid(np.arange(npix), np.arange(npix))   # (npix,npix)

    # Convert to centred (−1 … +1) coordinates (arbitrary units)
    xo = (col_o - centre) / centre
    yo = (row_o - centre) / centre

    # Source-plane coordinates via A: theta_S = A theta_O
    factor = (1.0 - kappa)
    xs = factor * ((1.0 - g1) * xo - g2        * yo)
    ys = factor * (-g2        * xo + (1.0 + g1) * yo)

    # Back to pixel coordinates in the source image
    col_s = xs * centre + centre
    row_s = ys * centre + centre

    coords_s = np.array([row_s.ravel(), col_s.ravel()])
    img_sheared = map_coordinates(img, coords_s, order=3,
                                  mode="constant", cval=0.0)
    return img_sheared.reshape(npix, npix)


# ---------------------------------------------------------------------------
# Unit conversions (TNG outputs are in "code units")
# ---------------------------------------------------------------------------

def code_to_physical(coords_code: np.ndarray, scale_factor: float) -> np.ndarray:
    """Convert comoving kpc/h to physical kpc."""
    return coords_code * scale_factor / h


def vel_to_physical(vels_code: np.ndarray, scale_factor: float) -> np.ndarray:
    """Convert code velocities (km/s * sqrt(a)) to physical km/s."""
    return vels_code * np.sqrt(scale_factor)


def mass_to_solar(mass_code: np.ndarray) -> np.ndarray:
    """Convert TNG mass code units (1e10 M_sun/h) to solar masses."""
    return mass_code * 1e10 / h


# ---------------------------------------------------------------------------
# Per-galaxy processing
# ---------------------------------------------------------------------------

def process_galaxy(row: pd.Series, sim: str, snap: int,
                   scale_factor: float, boxsize_kpc: float,
                   npix: int, fov_kpc: float,
                   out_dir: Path, headers: dict) -> bool:
    """
    Full pipeline for one galaxy row from the CSV.
    Returns True on success, False on failure.
    """
    subhalo_id = int(row["subhalo_id"])
    g1         = float(row["g1"])
    g2         = float(row["g2"])
    inclination = float(row["inclination"])
    theta_int   = float(row["theta_int"])

    gal_dir = out_dir / f"galaxy_{subhalo_id}"
    gal_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Subhalo metadata (position, half-mass radius, etc.) -----------
    print(f"  [{subhalo_id}] Fetching metadata …", flush=True)
    meta = get_subhalo_meta(sim, snap, subhalo_id, headers)
    pos_kpc = np.array([meta["pos_x"], meta["pos_y"], meta["pos_z"]],
                        dtype=np.float64) * scale_factor / h

    # ---- 2. Stellar particle cutout ---------------------------------------
    print(f"  [{subhalo_id}] Downloading stellar cutout …", flush=True)
    star_fields = [
        "Coordinates",
        "Velocities",
        "Masses",
        "GFM_StellarPhotometrics",   # columns: U,B,V,K,g,r,i,z  (AB mags)
        "GFM_StellarFormationTime",  # negative = wind particle
    ]
    star_data = download_cutout(sim, snap, subhalo_id, "stars",
                                star_fields, headers)
    if star_data is None or len(star_data.get("Coordinates", [])) == 0:
        print(f"  [{subhalo_id}] SKIP – no stellar particles")
        return False

    # Remove wind particles
    sft = star_data["GFM_StellarFormationTime"]
    ok  = sft > 0
    if ok.sum() < 50:
        print(f"  [{subhalo_id}] SKIP – too few real star particles ({ok.sum()})")
        return False
    for k in list(star_data.keys()):
        star_data[k] = star_data[k][ok]

    # Physical units
    star_coords = code_to_physical(star_data["Coordinates"].astype(np.float64),
                                   scale_factor)
    star_vels   = vel_to_physical(star_data["Velocities"].astype(np.float64),
                                  scale_factor)
    star_mass   = mass_to_solar(star_data["Masses"].astype(np.float64))

    # r-band luminosity weight from GFM_StellarPhotometrics column 5 (index 5)
    # The photometrics array stores absolute AB mags per particle.
    # Convert mag → luminosity: L ∝ 10^(-0.4 * M)
    phot = star_data["GFM_StellarPhotometrics"]
    if phot.ndim == 2 and phot.shape[1] >= 8:
        r_mag   = phot[:, 5].astype(np.float64)   # r-band (SDSS)
    else:
        r_mag   = phot[:, 0].astype(np.float64)   # fallback: first band
    r_lum = 10.0 ** (-0.4 * r_mag)                # relative luminosity

    # ---- 3. Gas particle cutout (for velocity map) ------------------------
    print(f"  [{subhalo_id}] Downloading gas cutout …", flush=True)
    gas_fields = ["Coordinates", "Velocities", "Masses"]
    gas_data   = download_cutout(sim, snap, subhalo_id, "gas",
                                 gas_fields, headers)
    have_gas   = (gas_data is not None and
                  len(gas_data.get("Coordinates", [])) > 0)

    if have_gas:
        gas_coords = code_to_physical(gas_data["Coordinates"].astype(np.float64),
                                      scale_factor)
        gas_vels   = vel_to_physical(gas_data["Velocities"].astype(np.float64),
                                     scale_factor)
        gas_mass   = mass_to_solar(gas_data["Masses"].astype(np.float64))
    else:
        print(f"  [{subhalo_id}] No gas particles; velocity map will be empty.")

    # ---- 4. Centre and subtract bulk velocity (subhalo CoM) ---------------
    star_coords = center_particles(star_coords, pos_kpc, boxsize_kpc)
    # Bulk velocity: mass-weighted mean of stars
    v_bulk = np.average(star_vels, weights=star_mass, axis=0)
    star_vels -= v_bulk

    if have_gas:
        gas_coords = center_particles(gas_coords, pos_kpc, boxsize_kpc)
        gas_vels  -= v_bulk

    # ---- 5. Project to image plane (source plane, no shear) ---------------
    sx, sy, sv_los = rotate_to_los(star_coords, star_vels, inclination, theta_int)

    if have_gas:
        gx, gy, gv_los = rotate_to_los(gas_coords, gas_vels, inclination, theta_int)

    # ---- 6. Render ORIGINAL stellar image ---------------------------------
    img_orig = particles_to_image(sx, sy, r_lum, npix, fov_kpc)

    # ---- 7. Render ORIGINAL velocity map ----------------------------------
    if have_gas:
        vmap_orig = particles_to_velmap(gx, gy, gv_los, gas_mass, npix, fov_kpc)
    else:
        vmap_orig = np.full((npix, npix), np.nan)

    # ---- 8. Apply shear to source-plane image via pixel remapping ---------
    img_sheared = shear_image_remap(img_orig, g1, g2)

    # Alternatively, compute sheared velocity map by transforming gas coords
    if have_gas:
        gx_obs, gy_obs = apply_shear_to_coords(gx, gy, g1, g2)
        vmap_sheared   = particles_to_velmap(gx_obs, gy_obs, gv_los,
                                             gas_mass, npix, fov_kpc)
    else:
        vmap_sheared = np.full((npix, npix), np.nan)

    # ---- 9. Save FITS files -----------------------------------------------
    pixel_scale_kpc = fov_kpc / npix

    common_header = fits.Header()
    common_header["SUBHALID"] = subhalo_id
    common_header["SIM"]      = sim
    common_header["SNAP"]     = snap
    common_header["REDSHIFT"] = round(1.0 / scale_factor - 1.0, 6)
    common_header["G1"]       = g1
    common_header["G2"]       = g2
    common_header["INCL_RAD"] = inclination
    common_header["THETA_INT"] = theta_int
    common_header["FOV_KPC"]  = fov_kpc
    common_header["PIXSCALE"] = pixel_scale_kpc   # kpc/pixel
    common_header["NPIX"]     = npix
    common_header["BUNIT"]    = "rel_lum"

    def save_fits(data: np.ndarray, fname: str, header: fits.Header,
                  bunit: str = ""):
        h2 = header.copy()
        if bunit:
            h2["BUNIT"] = bunit
        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=h2)
        hdu.writeto(fname, overwrite=True)

    save_fits(img_orig,
              str(gal_dir / f"galaxy_{subhalo_id}_image_original.fits"),
              common_header, bunit="rel_lum")

    save_fits(img_sheared,
              str(gal_dir / f"galaxy_{subhalo_id}_image_sheared.fits"),
              common_header, bunit="rel_lum")

    vh = common_header.copy()
    vh["BUNIT"] = "km/s"
    save_fits(vmap_orig,
              str(gal_dir / f"galaxy_{subhalo_id}_velmap_original.fits"),
              vh, bunit="km/s")

    save_fits(vmap_sheared,
              str(gal_dir / f"galaxy_{subhalo_id}_velmap_sheared.fits"),
              vh, bunit="km/s")

    print(f"  [{subhalo_id}] Saved 4 FITS files → {gal_dir}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate KL training images from TNG50 via web API."
    )
    p.add_argument("--csv",     required=True,
                   help="Input CSV with columns: subhalo_id, g1, g2, inclination, theta_int")
    p.add_argument("--outdir",  default="./kl_output",
                   help="Root output directory  (default: ./kl_output)")
    p.add_argument("--apikey",  default=None,
                   help="TNG API key (or set env var TNG_API_KEY)")
    p.add_argument("--sim",     default="TNG50-1",
                   help="Simulation name  (default: TNG50-1)")
    p.add_argument("--snap",    type=int, default=99,
                   help="Snapshot number  (default: 99 = z≈0)")
    p.add_argument("--npix",    type=int, default=256,
                   help="Image size in pixels  (default: 256)")
    p.add_argument("--fov_kpc", type=float, default=30.0,
                   help="Field of view in physical kpc  (default: 30)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip galaxies whose output directory already exists")
    return p.parse_args()


def main():
    args = parse_args()

    # API key
    api_key = args.apikey or os.environ.get("TNG_API_KEY")
    if not api_key:
        sys.exit("ERROR: Provide --apikey or set the TNG_API_KEY environment variable.\n"
                 "Register at https://www.tng-project.org/users/register/")

    headers = {"api-key": api_key}

    # Test connection
    print(f"Testing TNG API connection ({args.sim}) …", flush=True)
    tng_get(f"{BASE_URL}{args.sim}/", headers=headers)
    print("  Connection OK\n", flush=True)

    # Snapshot metadata
    redshift     = get_snapshot_redshift(args.sim, args.snap, headers)
    scale_factor = 1.0 / (1.0 + redshift)
    # TNG box size for TNG50-1: 51.7 Mpc/h comoving → convert to physical kpc
    # We fetch it from the API
    sim_meta     = tng_get(f"{BASE_URL}{args.sim}/", headers=headers).json()
    boxsize_kpc  = float(sim_meta["boxsize"]) * scale_factor / h * 1e3  # Mpc/h → kpc
    print(f"Snapshot {args.snap}: z = {redshift:.4f}, "
          f"a = {scale_factor:.4f}, box = {boxsize_kpc/1e3:.1f} Mpc\n",
          flush=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    required_cols = {"subhalo_id", "g1", "g2", "inclination", "theta_int"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing required columns: {missing}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip, n_fail = 0, 0, 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Galaxies"):
        subhalo_id = int(row["subhalo_id"])
        gal_dir    = out_dir / f"galaxy_{subhalo_id}"

        if args.skip_existing and gal_dir.exists():
            expected = [
                gal_dir / f"galaxy_{subhalo_id}_image_original.fits",
                gal_dir / f"galaxy_{subhalo_id}_image_sheared.fits",
                gal_dir / f"galaxy_{subhalo_id}_velmap_original.fits",
            ]
            if all(p.exists() for p in expected):
                n_skip += 1
                continue

        print(f"\nProcessing subhalo {subhalo_id} ({idx+1}/{len(df)}) …",
              flush=True)
        try:
            ok = process_galaxy(
                row           = row,
                sim           = args.sim,
                snap          = args.snap,
                scale_factor  = scale_factor,
                boxsize_kpc   = boxsize_kpc,
                npix          = args.npix,
                fov_kpc       = args.fov_kpc,
                out_dir       = out_dir,
                headers       = headers,
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:
            print(f"  [ERROR] subhalo {subhalo_id}: {exc}", flush=True)
            n_fail += 1

        # Polite delay between API calls to avoid rate-limiting
        time.sleep(0.5)

    print(f"\nDone.  Success: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}",
          flush=True)


if __name__ == "__main__":
    main()