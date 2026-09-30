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


# ═══════════════════════════════════════════════════════════════════════════════
# Disc quality diagnostics
# (adapted from the reference implementation; integrated into pipeline style)
# ═══════════════════════════════════════════════════════════════════════════════

def _angular_momentum_ez(mass, pos, vel, r_max=None):
    """Unit vector along the total angular momentum within r_max (if given)."""
    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        if idx.sum() == 0:
            raise ValueError("No particles within r_max")
        pos, vel, mass = pos[idx], vel[idx], mass[idx]
    J   = np.sum(mass[:, None] * np.cross(pos, vel), axis=0)
    Jn  = np.linalg.norm(J)
    if Jn == 0:
        raise ValueError("Zero angular momentum")
    return J / Jn


def kappa_rotation(mass, pos, vel, r_max=None):
    """
    κ_rot = E_rot / E_kin — fraction of kinetic energy in ordered rotation.

    Computed within r_max if given.  Following the reference code:
      E_rot = 0.5 * Σ m_i (j_z,i / r_i)²       (rotational KE)
      E_kin = 0.5 * Σ m_i |v_i|²                (total KE)
    where j_z,i = (r_i × v_i) · ê_z is the specific z-angular momentum
    and r_i is the cylindrical radius.

    κ_rot > 0.5 is a standard disc criterion (Sales et al. 2012).
    We use > 0.5 as the minimum to call a galaxy a disc.
    """
    ez = _angular_momentum_ez(mass, pos, vel, r_max=r_max)

    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        pos, vel, mass = pos[idx], vel[idx], mass[idx]

    ji   = np.cross(pos, vel)         # specific angular momentum vectors (N,3)
    jz   = ji @ ez                    # projection onto disc normal

    # cylindrical radius
    pos_z = (pos @ ez)[:, None] * ez  # component along ez
    r_cyl = np.linalg.norm(pos - pos_z, axis=1)
    r_cyl = np.maximum(r_cyl, 1e-9)   # avoid division by zero at centre

    E_rot = 0.5 * np.sum(mass * (jz / r_cyl) ** 2)
    E_kin = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))

    if E_kin == 0:
        return 0.0
    return float(E_rot / E_kin)


def vrot_over_sigma(mass, pos, vel, r_max=None):
    """
    V_rot / σ — mass-weighted mean rotation speed over 3D velocity dispersion,
    computed within r_max.  Following find_v_rot_v_sigma() in the reference.

    V_rot: mass-weighted mean of the azimuthal velocity component
    σ:     sqrt of mass-weighted mean squared residual velocity (3D dispersion
           after subtracting the ordered rotation)

    Returns (v_rot_over_sigma, v_rot_kms, sigma_kms).
    V_rot/σ > 1 is required; > 2 is a clean rotating disc.
    """
    ez = _angular_momentum_ez(mass, pos, vel, r_max=r_max)

    if r_max is not None:
        idx = np.linalg.norm(pos, axis=1) < r_max
        if idx.sum() < 10:
            raise ValueError("Too few particles within r_max")
        pos, vel, mass = pos[idx], vel[idx], mass[idx]

    # Azimuthal unit vectors at each particle
    pos_z  = (pos @ ez)[:, None] * ez
    e_phi  = np.cross(ez, pos - pos_z)
    norm   = np.linalg.norm(e_phi, axis=1)
    good   = norm > 1e-8
    e_phi[good]  /= norm[good, None]
    e_phi[~good]  = 0.0

    v_phi  = np.einsum("ij,ij->i", vel, e_phi)   # azimuthal speed per particle

    M_tot  = mass.sum()
    v_rot  = float(np.sum(mass * v_phi) / M_tot)  # mass-weighted mean

    # 3D residual after subtracting ordered rotation
    v_mean  = v_rot * e_phi                        # ordered component (N,3)
    v_resid = vel - v_mean
    sigma   = float(np.sqrt(np.sum(mass * np.sum(v_resid**2, axis=1)) / M_tot))

    if sigma == 0:
        return np.inf, v_rot, sigma
    return float(v_rot / sigma), float(v_rot), float(sigma)


def disc_quality_checks(star_coords, star_vels, star_mass, r_half,
                        gas_coords=None, gas_vels=None, sfr_weight=None,
                        kappa_min=0.4, vrot_sigma_min=1.0,
                        min_particles=100, redshift=1.0):
    """
    Run disc quality checks on face-on-aligned particles.

    Aperture: 1×r_half for both κ_rot and V/σ.
    Using 2×r_half was wrong — it includes too much stellar halo, which
    inflates σ and suppresses V/σ even for clean rotating discs.

    V/σ is computed from SFR-weighted gas particles if available (preferred:
    gas is disc-confined, so V/σ is naturally clean without aperture issues).
    Falls back to stellar particles within 1×r_half if no gas is provided.

    κ_rot threshold scales with redshift: discs at z=1 are more turbulent
    (κ_rot ~ 0.4 is a good disc at z=1 vs 0.5 at z=0).
    """
    info = {"n_star": len(star_mass), "kappa_rot": np.nan,
            "vrot_sigma": np.nan, "vrot_kms": np.nan, "sigma_kms": np.nan,
            "vrot_source": "stars"}

    # Redshift-dependent κ_rot threshold: looser at high-z
    # Linear interpolation: 0.4 at z≥1, 0.5 at z=0
    kappa_min_z = float(np.clip(kappa_min + 0.1 * (1.0 - min(redshift, 1.0)),
                                kappa_min, 0.5))
    info["kappa_min_used"] = round(kappa_min_z, 3)

    # 1. Particle count
    if len(star_mass) < min_particles:
        info["fail_reason"] = f"n_star={len(star_mass)} < {min_particles}"
        return False, info

    # Use 1×r_half — tighter aperture excludes stellar halo
    r_cut = 1.0 * r_half

    # 2. κ_rot (stellar, within 1×r_half)
    try:
        kappa = kappa_rotation(star_mass, star_coords, star_vels, r_max=r_cut)
        info["kappa_rot"] = round(kappa, 4)
        if kappa < kappa_min_z:
            info["fail_reason"] = (f"kappa_rot={kappa:.3f} < "
                                   f"{kappa_min_z:.3f} (z={redshift:.2f})")
            return False, info
    except ValueError as e:
        info["fail_reason"] = f"kappa_rot error: {e}"
        return False, info

    # 3. V/σ — prefer gas (disc-confined, no aperture contamination from halo)
    if gas_coords is not None and sfr_weight is not None:
        try:
            vs, vr, sg = vrot_over_sigma(sfr_weight, gas_coords, gas_vels,
                                         r_max=None)  # gas already disc-confined
            info["vrot_source"]  = "gas_SFR"
            info["vrot_sigma"]   = round(vs, 3)
            info["vrot_kms"]     = round(vr, 2)
            info["sigma_kms"]    = round(sg, 2)
            if vs < vrot_sigma_min:
                info["fail_reason"] = f"vrot/sigma(gas)={vs:.3f} < {vrot_sigma_min}"
                return False, info
        except Exception:
            # Fall through to stellar fallback
            pass

    if np.isnan(info["vrot_sigma"]):
        # Stellar fallback within 1×r_half
        try:
            vs, vr, sg = vrot_over_sigma(star_mass, star_coords, star_vels,
                                          r_max=r_cut)
            info["vrot_source"] = "stars_1rhalf"
            info["vrot_sigma"]  = round(vs, 3)
            info["vrot_kms"]    = round(vr, 2)
            info["sigma_kms"]   = round(sg, 2)
            if vs < vrot_sigma_min:
                info["fail_reason"] = f"vrot/sigma(stars)={vs:.3f} < {vrot_sigma_min}"
                return False, info
        except ValueError as e:
            info["fail_reason"] = f"vrot/sigma error: {e}"
            return False, info

    info["fail_reason"] = ""
    return True, info


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


def fetch_galaxy_data(subhalo_id, sim, snap, scale_factor, boxsize_kpc, headers):
    """
    Download and prepare particle data for one galaxy.
    Returns a dict of face-on-aligned arrays ready for projection, or None on failure.
    This is called ONCE per galaxy regardless of how many draws are requested.
    """
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
        return None

    sft = star_data["GFM_StellarFormationTime"]
    ok  = sft > 0
    if ok.sum() < 50:
        print(f"  [{subhalo_id}] SKIP – too few real star particles ({ok.sum()})")
        return None
    for k in list(star_data.keys()):
        star_data[k] = star_data[k][ok]

    star_coords = code_to_physical(star_data["Coordinates"].astype(np.float64), scale_factor)
    star_vels   = vel_to_physical(star_data["Velocities"].astype(np.float64), scale_factor)
    star_mass   = mass_to_solar(star_data["Masses"].astype(np.float64))

    phot  = star_data["GFM_StellarPhotometrics"]
    r_col = 3 if phot.ndim == 2 and phot.shape[1] >= 8 else 0
    r_lum = 10.0 ** (-0.4 * phot[:, r_col].astype(np.float64))

    # 3. Gas cutout — SFR-weighted (Hα proxy)
    print(f"  [{subhalo_id}] Downloading gas cutout …", flush=True)
    gas_data = download_cutout(sim, snap, subhalo_id, "gas",
                               ["Coordinates", "Velocities", "Masses",
                                "StarFormationRate"], headers)

    have_gas   = (gas_data is not None and
                  len(gas_data.get("Coordinates", [])) > 0)
    sfr_weight = None

    if have_gas:
        gas_coords_raw = code_to_physical(
            gas_data["Coordinates"].astype(np.float64), scale_factor)
        gas_vels_raw   = vel_to_physical(
            gas_data["Velocities"].astype(np.float64), scale_factor)
        sfr_all        = gas_data["StarFormationRate"].astype(np.float64)
        sf_mask        = sfr_all > 0
        if sf_mask.sum() < 10:
            print(f"  [{subhalo_id}] No star-forming gas — velocity map will be NaN")
            have_gas = False
        else:
            gas_coords = gas_coords_raw[sf_mask]
            gas_vels   = gas_vels_raw[sf_mask]
            sfr_weight = sfr_all[sf_mask]
            print(f"  [{subhalo_id}] {sf_mask.sum()}/{len(sfr_all)} "
                  f"gas particles are star-forming", flush=True)
    else:
        print(f"  [{subhalo_id}] No gas particles.")

    # 4. Centre + subtract bulk velocity (global mass-weighted mean)
    star_coords = center_particles(star_coords, pos_kpc, boxsize_kpc)
    v_bulk      = np.average(star_vels, weights=star_mass, axis=0)
    star_vels  -= v_bulk
    if have_gas:
        gas_coords = center_particles(gas_coords, pos_kpc, boxsize_kpc)
        gas_vels  -= v_bulk

    # 4b. Face-on alignment from stellar Lz
    r_half           = float(meta["halfmassrad_stars"]) * scale_factor / h
    rot_face, L_star = face_on_rotation(star_coords, star_vels, star_mass,
                                        r_max=2.0 * r_half)

    # Measure gas-stellar misalignment for the FITS header (informational only)
    misalign = -1.0
    if have_gas:
        try:
            L_gas_orig = angular_momentum_direction(gas_coords, gas_vels,
                                                    sfr_weight,
                                                    r_max=2.0 * r_half)
            misalign   = misalignment_deg(L_star, L_gas_orig)
        except ValueError:
            pass

    # Apply stellar face-on rotation to both components
    star_coords = rot_face.apply(star_coords)
    star_vels   = rot_face.apply(star_vels)
    if have_gas:
        gas_coords = rot_face.apply(gas_coords)
        gas_vels   = rot_face.apply(gas_vels)

    if have_gas and misalign >= 0:
        print(f"  [{subhalo_id}] star-gas misalignment = {misalign:.1f}°",
              flush=True)

    return {
        "star_coords": star_coords,
        "star_vels":   star_vels,
        "star_mass":   star_mass,
        "r_lum":       r_lum,
        "have_gas":    have_gas,
        "gas_coords":  gas_coords  if have_gas else None,
        "gas_vels":    gas_vels    if have_gas else None,
        "sfr_weight":  sfr_weight  if have_gas else None,
        "r_half":      r_half,
        "misalign":    misalign,
        "rot_face":    rot_face,
        "L_star":      L_star,
    }


def render_draw(gd, subhalo_id, sim, snap, scale_factor,
                g1, g2, inclination, theta_int,
                npix, fov_kpc, gal_dir):
    """
    Project and render one (inclination, theta_int, g1, g2) draw from
    pre-fetched, face-on-aligned particle data `gd`.
    Writes 4 FITS files to gal_dir.  Returns True on success.
    """
    gal_dir.mkdir(parents=True, exist_ok=True)

    # Verify the composed rotation reproduces the requested angles
    inc_real, pa_real = realized_orientation(
        gd["rot_face"], gd["L_star"], inclination, theta_int)
    assert abs(inc_real - inclination) < 1e-6 and abs(pa_real - theta_int) < 1e-6

    # 5. Project to image plane for this draw's (inclination, theta_int)
    sx, sy, _ = rotate_to_los(gd["star_coords"], gd["star_vels"],
                              inclination, theta_int)
    if gd["have_gas"]:
        gx, gy, gv_los = rotate_to_los(gd["gas_coords"], gd["gas_vels"],
                                        inclination, theta_int)

    # 6. Render original stellar image
    img_orig = particles_to_image(sx, sy, gd["r_lum"], npix, fov_kpc)

    # 7. Render original SFR-weighted velocity map
    if gd["have_gas"]:
        vmap_orig = particles_to_velmap(gx, gy, gv_los,
                                        gd["sfr_weight"], npix, fov_kpc)
    else:
        vmap_orig = np.full((npix, npix), np.nan)

    # 8. Apply shear for this draw's (g1, g2)
    img_sheared = shear_image_remap(img_orig, g1, g2)
    if gd["have_gas"]:
        gx_obs, gy_obs = apply_shear_to_coords(gx, gy, g1, g2)
        vmap_sheared   = particles_to_velmap(gx_obs, gy_obs, gv_los,
                                             gd["sfr_weight"], npix, fov_kpc)
    else:
        vmap_sheared = np.full((npix, npix), np.nan)

    # 9. Save FITS
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
    hdr["RHALF"]    = gd["r_half"]
    hdr["MISALIGN"] = gd["misalign"]
    hdr["VELWGT"]   = "SFR"

    def save_fits(data, fname, bunit):
        h2 = hdr.copy(); h2["BUNIT"] = bunit
        fits.PrimaryHDU(data=data.astype(np.float32), header=h2).writeto(
            fname, overwrite=True)

    sid = subhalo_id
    save_fits(img_orig,    str(gal_dir / f"galaxy_{sid}_image_original.fits"),  "rel_lum")
    save_fits(img_sheared, str(gal_dir / f"galaxy_{sid}_image_sheared.fits"),   "rel_lum")
    save_fits(vmap_orig,   str(gal_dir / f"galaxy_{sid}_velmap_original.fits"), "km/s")
    save_fits(vmap_sheared,str(gal_dir / f"galaxy_{sid}_velmap_sheared.fits"),  "km/s")
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

    # Group rows by (subhalo_id) so we fetch particle data once per galaxy
    # and render all draws (different inclination/PA/shear) in a tight loop.
    galaxy_groups = list(df.groupby("subhalo_id", sort=False))
    n_ok = n_skip = n_fail = 0

    for gal_idx, (sid, gal_rows) in enumerate(
            tqdm(galaxy_groups, desc="Galaxies")):
        sid = int(sid)

        # Check if ALL draws for this galaxy already exist (skip the fetch)
        draw_indices = (gal_rows["draw_idx"].tolist()
                        if "draw_idx" in gal_rows.columns
                        else [0] * len(gal_rows))
        if args.skip_existing:
            all_done = all(
                all((out_dir / f"galaxy_{sid}_draw{k:04d}" /
                     f"galaxy_{sid}_{s}.fits").exists()
                    for s in ("image_original", "image_sheared",
                              "velmap_original"))
                for k in draw_indices
            )
            if all_done:
                n_skip += len(gal_rows)
                continue

        print(f"\n[{gal_idx+1}/{len(galaxy_groups)}] "
              f"Subhalo {sid}  ({len(gal_rows)} draws) …", flush=True)

        # ── Fetch particle data ONCE for this galaxy ──────────────────────
        try:
            gd = fetch_galaxy_data(sid, args.sim, args.snap,
                                   scale_factor, boxsize_kpc, headers)
        except Exception as exc:
            print(f"  [ERROR] fetch failed for {sid}: {exc}", flush=True)
            n_fail += len(gal_rows)
            time.sleep(1.0)
            continue

        if gd is None:
            n_fail += len(gal_rows)
            continue

        # ── Render each draw from the cached face-on particles ────────────
        for row_idx, (_, row) in enumerate(gal_rows.iterrows()):
            draw_idx  = int(row["draw_idx"]) if "draw_idx" in row.index else 0
            g1        = float(row["g1"])
            g2        = float(row["g2"])
            incl      = float(row["inclination"])
            theta     = float(row["theta_int"])
            gal_dir   = out_dir / f"galaxy_{sid}_draw{draw_idx:04d}"

            if args.skip_existing and gal_dir.exists():
                expected = [gal_dir / f"galaxy_{sid}_{s}.fits"
                            for s in ("image_original", "image_sheared",
                                      "velmap_original")]
                if all(p.exists() for p in expected):
                    n_skip += 1
                    continue

            try:
                render_draw(gd, sid, args.sim, args.snap, scale_factor,
                            g1, g2, incl, theta,
                            args.npix, args.fov_kpc, gal_dir)
                print(f"  draw {draw_idx:04d}  "
                      f"i={incl:.2f}  θ={theta:.2f}  "
                      f"g=({g1:+.3f},{g2:+.3f})  → {gal_dir.name}",
                      flush=True)
                n_ok += 1
            except Exception as exc:
                print(f"  [ERROR] render draw {draw_idx} for {sid}: {exc}",
                      flush=True)
                n_fail += 1

        # Polite delay between galaxies (not between draws — only 1 API call)
        time.sleep(0.5)

    print(f"\nDone.  Success: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}",
          flush=True)


if __name__ == "__main__":
    main()