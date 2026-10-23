#!/usr/bin/env python3
"""
convert_manga_pkl.py
====================
Convert data_info-<mangaid>.pkl files (the dicts returned by
Manga.get_data_info) into compact FITS files that predict_manga.py reads
with --kl_dir.

The pkl arrays carry no WCS of their own:
  * the image is a cutout of an SDSS frame, but par_meta['ap_wcs'] is the
    WCS of the whole frame, so the cutout origin is recomputed the same way
    Manga.get_image made it (int(objpos - box/2), 1-based pixels);
  * the velocity map was np.flip-ped, together with its SPX_SKYCOO offset
    grids (par_meta RA_grid / Dec_grid).  Those are sky-right offsets in
    arcsec from the galaxy centre, +RA toward the East, so a TAN WCS is
    fitted to them directly -- this is correct whatever flips were applied.

Output <outdir>/<mangaid>_kl.fits
  PRIMARY      header: MANGAID, PLATEIFU, RA, DEC, REDSHIFT, IMGPSF, VELPSF
               (FWHM, arcsec; VELPSF interpolated to observed H-alpha),
               LOGMSTAR, SRCFILE
  IMAGE        SDSS r-band cutout [electrons], with the cutout's WCS
  IMAGE_CONTAM 1 = pixel inside a contaminant box (variance set to 1e14)
  VEL          H-alpha gas velocity [km/s]
  VEL_VAR      velocity variance [km^2/s^2] (includes the 5 km/s floor)
  MASK_COVER   1 = spaxel has a velocity measurement (finite variance)
  MASK_DAP     1 = DAP velocity quality mask clean   ('default_mask')
  MASK_SNR     1 = H-alpha A/N above the pkl's cut   ('Halpha_snr_mask')
  MASK_STRICT  1 = largest coherent region of both   ('mask')
plus manifest.csv with one row per input file.

With --grid_dir (or save_grid_files() from a notebook) it also writes
<mangaid>_grid_image.fits and <mangaid>_grid_velmap.fits: image and
velocity aligned on one North-up grid spanning the model's field of view
(30 kpc before the crop, npix pixels).  See save_grid().

Unpickling needs the packages the pkl files were written with (galsim for
the GalSim WCS objects): pip install galsim

Usage
-----
  python convert_manga_pkl.py --pkl_dir "../drive/MyDrive/Manga data" \
      --outdir /path/to/KL_repo/data/manga \
      --grid_dir /path/to/KL_repo/sdss-manga/fits --checkpoint best_model.pt

  # notebook
  from convert_manga_pkl import save_grid_files
  save_grid_files(glob.glob("../drive/MyDrive/Manga data/data_info-*.pkl"),
                  "/content/Kinematic_Lensing_ML/sdss-manga/fits",
                  checkpoint="kl_dataset/model_output/best_model.pt")
"""

import argparse
import glob
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck15
from astropy.io import fits
from astropy.wcs import WCS

HALPHA_REST = 6564.6   # Angstrom (vacuum)


def cutout_wcs(ap_wcs, shape, ra, dec):
    """
    WCS of the cutout made in Manga.get_image: the frame WCS with its
    reference pixel shifted by the cutout origin.  Returns the WCS and the
    distance [px] between the galaxy and the cutout centre (should be <~1).
    """
    w = ap_wcs.celestial.deepcopy()
    x, y = w.all_world2pix([[ra, dec]], 1)[0]          # 1-based, like GalSim
    ny, nx = shape
    xmin, ymin = int(x - (nx - 1) / 2), int(y - (ny - 1) / 2)
    w.wcs.crpix = w.wcs.crpix - np.array([xmin - 1, ymin - 1])
    w.pixel_shape = (nx, ny)
    cx, cy = w.all_world2pix([[ra, dec]], 0)[0]
    return w, float(np.hypot(cx - (nx - 1) / 2, cy - (ny - 1) / 2))


def wcs_from_offsets(dra, ddec, ra, dec):
    """
    TAN WCS for a map whose spaxels have sky-right offsets (dRA toward East,
    dDec toward North, arcsec) from (ra, dec).  Returns the WCS and the
    largest residual of the linear fit [arcsec] (should be ~0).
    """
    yy, xx = np.indices(dra.shape)
    ok = np.isfinite(dra) & np.isfinite(ddec)
    A  = np.column_stack([np.ones(ok.sum()), xx[ok], yy[ok]])
    cx = np.linalg.lstsq(A, dra[ok],  rcond=None)[0]
    cy = np.linalg.lstsq(A, ddec[ok], rcond=None)[0]
    M  = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])          # arcsec / px
    p0 = np.linalg.solve(M, -np.array([cx[0], cy[0]]))      # 0-based centre

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = p0 + 1
    w.wcs.cd    = M / 3600.0
    w.pixel_shape = dra.shape[::-1]
    resid = max(np.abs(A @ cx - dra[ok]).max(), np.abs(A @ cy - ddec[ok]).max())
    return w, float(resid)


def read_pkl(path, strict_only=False):
    """
    Load one data_info pkl and attach WCS to its image and velocity map.

    strict_only: restrict every velocity mask to the strict mask ('mask').
    Use it for pkl files sheared by manga_shear.py: that script shears the
    velocity only inside 'mask' and leaves the spaxels outside it with their
    ORIGINAL, unsheared values (and 'default_mask' untouched), so the
    coverage / DAP masks would mix sheared and unsheared velocities.
    """
    d   = joblib.load(path)
    g   = d["galaxy"]
    im  = d["image"]
    gas = d["vmap"]["gas"]
    ra, dec, z = float(g["RA"]), float(g["Dec"]), float(g["redshift"])

    img = np.asarray(im["data"], dtype=np.float32)
    iwcs, img_off = cutout_wcs(im["par_meta"]["ap_wcs"], img.shape, ra, dec)

    pm = gas["par_meta"]
    vwcs, resid = wcs_from_offsets(np.asarray(pm["RA_grid"], float),
                                   np.asarray(pm["Dec_grid"], float), ra, dec)
    var   = np.asarray(gas["var"], dtype=np.float64)
    cover = np.isfinite(var) & (var < 1e10)
    masks = {
        "MASK_COVER":  cover,
        "MASK_DAP":    (np.asarray(gas["default_mask"]) > 0.5) & cover,
        "MASK_SNR":    np.asarray(gas["Halpha_snr_mask"]) > 0.5,
        "MASK_STRICT": np.asarray(gas["mask"]).astype(bool) & cover,
    }
    if strict_only:
        masks["MASK_COVER"] = masks["MASK_DAP"] = masks["MASK_STRICT"]
    meta = {
        "mangaid": str(g["mangaid"]).strip(), "plateifu": str(g["plateifu"]).strip(),
        "ra": ra, "dec": dec, "z": z, "src": os.path.basename(path),
        # PSF FWHM at the observed H-alpha wavelength (between r and i)
        "vel_psf": float(np.interp(HALPHA_REST * (1 + z), pm["psf_wavel"], pm["psf_fwhm"])),
        "img_psf": float(im["par_meta"].get("psfFWHM", 1.32)),
        "logmstar": (float(np.asarray(g["log10_Mstar"]).ravel()[0])
                     if "log10_Mstar" in g else None),
        "img_off": img_off, "vel_resid": resid, "strict_only": bool(strict_only),
    }
    return {"meta": meta, "img": img, "img_wcs": iwcs,
            "contam": np.asarray(im["var"]) >= 1e13,
            "vel": np.asarray(gas["data"], dtype=np.float32), "vel_var": var,
            "vel_wcs": vwcs, "masks": masks}


def _base_header(m):
    h = fits.Header()
    h["MANGAID"]  = m["mangaid"]
    h["PLATEIFU"] = m["plateifu"]
    h["RA"]       = (m["ra"],  "deg")
    h["DEC"]      = (m["dec"], "deg")
    h["REDSHIFT"] = m["z"]
    h["IMGPSF"]   = (m["img_psf"], "image PSF FWHM [arcsec]")
    h["VELPSF"]   = (m["vel_psf"], "MaNGA PSF FWHM at obs. H-alpha [arcsec]")
    if m["logmstar"] is not None:
        h["LOGMSTAR"] = m["logmstar"]
    h["SRCFILE"]  = m["src"][:68]
    if m.get("strict_only"):
        h["VELMASKS"] = ("strict only", "VEL_DAP/VEL_ALL = strict (sheared input)")
    return h


def _record(a, file, status):
    m = a["meta"]
    return {"mangaid": m["mangaid"], "plateifu": m["plateifu"], "src": m["src"],
            "file": file, "status": status, "redshift": m["z"],
            "img_shape": f"{a['img'].shape[0]}x{a['img'].shape[1]}",
            "img_centre_offset_px": m["img_off"], "vel_wcs_resid_arcsec": m["vel_resid"],
            "n_cover": int(a["masks"]["MASK_COVER"].sum()),
            "n_dap": int(a["masks"]["MASK_DAP"].sum()),
            "n_strict": int(a["masks"]["MASK_STRICT"].sum()),
            "img_psf": m["img_psf"], "vel_psf": m["vel_psf"]}


def convert(path, outdir, overwrite=False):
    """pkl -> <mangaid>_kl.fits (native resolution, with WCS and all masks)."""
    a   = read_pkl(path)
    m   = a["meta"]
    out = outdir / f"{m['mangaid']}_kl.fits"
    if out.exists() and not overwrite:
        return {**_record(a, out.name, "exists")}

    ih = a["img_wcs"].to_header(relax=True); ih["BUNIT"] = "electron"
    vh = a["vel_wcs"].to_header()
    hdus = [fits.PrimaryHDU(header=_base_header(m)),
            fits.ImageHDU(a["img"], ih, name="IMAGE"),
            fits.ImageHDU(a["contam"].astype(np.uint8), ih, name="IMAGE_CONTAM"),
            fits.ImageHDU(a["vel"], vh, name="VEL"),
            fits.ImageHDU(a["vel_var"].astype(np.float32), vh, name="VEL_VAR")]
    hdus += [fits.ImageHDU(k_m.astype(np.uint8), vh, name=k)
             for k, k_m in a["masks"].items()]
    fits.HDUList(hdus).writeto(out, overwrite=True)
    return _record(a, out.name, "ok")


# ═══════════════════════════════════════════════════════════════════════════════
# Aligned files on the model grid
# ═══════════════════════════════════════════════════════════════════════════════

def model_grid_wcs(ra, dec, z, fov_kpc, npix):
    """
    North-up / East-left TAN grid of npix pixels spanning fov_kpc (Planck15),
    centred on (ra, dec).  Identical to predict_manga.grid_wcs, so
    predict_manga uses files on this grid without resampling them again.
    """
    kpc_as = Planck15.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
    pix_as = fov_kpc / npix / kpc_as
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(npix + 1) / 2.0] * 2
    w.wcs.cdelt = [-pix_as / 3600.0, pix_as / 3600.0]
    return w, pix_as


def save_grid(path, outdir, fov_kpc=30.0, npix=128, overwrite=False, strict_only=False):
    """
    pkl -> <mangaid>_grid_image.fits + <mangaid>_grid_velmap.fits, both on
    the same npix x npix grid spanning fov_kpc: the field of the TNG training
    images before the central crop.  Nothing else is done to the data (no
    sky subtraction, neighbour removal or smoothing): predict_manga.py
    applies those, exactly as for the other input formats.

    _grid_image.fits   PRIMARY: image [electrons], NaN outside the cutout
    _grid_velmap.fits  PRIMARY: H-alpha velocity [km/s], strict mask (NaN = masked)
                       VEL_DAP: DAP quality mask only
                       VEL_ALL: every spaxel with a measurement (no quality cuts)
    strict_only=True (sheared pkl from manga_shear.py): VEL_DAP and VEL_ALL
    are the strict map too, see read_pkl.
    """
    from reproject import reproject_exact, reproject_interp

    a = read_pkl(path, strict_only)
    m = a["meta"]
    out_img = outdir / f"{m['mangaid']}_grid_image.fits"
    out_vel = outdir / f"{m['mangaid']}_grid_velmap.fits"
    if out_img.exists() and out_vel.exists() and not overwrite:
        # already made: report its grid statistics from the header, so a
        # rerun still writes a complete grid_manifest.csv
        h = fits.getheader(out_vel, 0)
        return {**_record(a, out_vel.name, "exists"), "arcsec_per_px": h.get("ARCS_PX"),
                "img_coverage": h.get("IMGCOV"), "fill_strict": h.get("FILL_STR"),
                "fill_dap": h.get("FILL_DAP"), "fill_all": h.get("FILL_ALL")}

    grid, pix_as = model_grid_wcs(m["ra"], m["dec"], m["z"], fov_kpc, npix)
    shape = (npix, npix)

    img, cov = reproject_exact((a["img"].astype(np.float64), a["img_wcs"]),
                               grid, shape_out=shape)
    img[~(cov > 0)] = np.nan

    vel = {}
    for name, mask in (("STRICT", "MASK_STRICT"), ("DAP", "MASK_DAP"),
                       ("ALL", "MASK_COVER")):
        v = np.where(a["masks"][mask], a["vel"], np.nan).astype(np.float64)
        vel[name], _ = reproject_interp((v, a["vel_wcs"]), grid,
                                        shape_out=shape, order="bilinear")

    h = _base_header(m)
    h.update(grid.to_header())
    h["FOV_KPC"]  = (fov_kpc, "grid width [kpc]")
    h["NPIX"]     = npix
    h["KPC_PX"]   = (fov_kpc / npix, "kpc per pixel")
    h["ARCS_PX"]  = (pix_as, "arcsec per pixel")
    h["IMGCOV"]   = (float(np.isfinite(img).mean()), "fraction of grid covered by image")
    for k in ("STRICT", "DAP", "ALL"):
        h[f"FILL_{k[:3]}"] = (float(np.isfinite(vel[k]).mean()),
                              f"velmap fill fraction, {k.lower()} mask")

    hi = h.copy(); hi["BUNIT"] = "electron"
    fits.PrimaryHDU(img.astype(np.float32), hi).writeto(out_img, overwrite=True)
    hv = h.copy(); hv["BUNIT"] = "km/s"; hv["VELMASK"] = "strict"
    vh = grid.to_header()
    fits.HDUList([fits.PrimaryHDU(vel["STRICT"].astype(np.float32), hv),
                  fits.ImageHDU(vel["DAP"].astype(np.float32), vh, name="VEL_DAP"),
                  fits.ImageHDU(vel["ALL"].astype(np.float32), vh, name="VEL_ALL")]
                 ).writeto(out_vel, overwrite=True)

    return {**_record(a, out_vel.name, "ok"), "arcsec_per_px": pix_as,
            "img_coverage": h["IMGCOV"], "fill_strict": h["FILL_STR"],
            "fill_dap": h["FILL_DAP"], "fill_all": h["FILL_ALL"]}


def save_grid_files(paths, outdir, fov_kpc=30.0, npix=None, checkpoint=None,
                    overwrite=False, strict_only=False, verbose=True):
    """
    Write aligned image + velocity files on the model grid for many pkl files.

    npix: grid size; if None, read from `checkpoint` (train_kl_model.py
    best_model.pt), else 128.  fov_kpc: field before the crop (TNG: 30 kpc).
    Returns the manifest (also written to <outdir>/grid_manifest.csv).
    """
    if npix is None:
        npix = 128
        if checkpoint:
            import torch
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            npix = int(ck.get("args", {}).get("npix", 128))
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(paths)
    print(f"{len(paths)} pkl files -> {outdir}  (grid {npix}x{npix}, {fov_kpc} kpc"
          + (", strict mask only)" if strict_only else ")"))

    rows = []
    for i, f in enumerate(paths, 1):
        try:
            r = save_grid(f, outdir, fov_kpc, npix, overwrite, strict_only)
            if r["status"] == "ok":
                _warn(r)
        except Exception as exc:
            r = {"src": os.path.basename(f), "status": f"error: {exc}"}
            print(f"  [ERROR] {os.path.basename(f)}: {exc}")
        rows.append(r)
        if verbose and (i % 50 == 0 or i == len(paths)):
            print(f"  {i}/{len(paths)} done")
    return _write_manifest(rows, outdir / "grid_manifest.csv")


def _warn(r):
    if r["img_centre_offset_px"] > 2:
        print(f"  [WARN] {r['mangaid']}: galaxy {r['img_centre_offset_px']:.1f} px "
              f"from the cutout centre; check the image WCS")
    if r["vel_wcs_resid_arcsec"] > 0.05:
        print(f"  [WARN] {r['mangaid']}: velocity WCS fit residual "
              f"{r['vel_wcs_resid_arcsec']:.3f}\"")
    if r["n_strict"] == 0:
        print(f"  [WARN] {r['mangaid']}: strict mask is empty")


def _write_manifest(rows, path):
    man = pd.DataFrame(rows)
    man.to_csv(path, index=False)
    st = man["status"].astype(str)
    print(f"\nok: {(st == 'ok').sum()}  already there: {(st == 'exists').sum()}  "
          f"errors: {st.str.startswith('error').sum()}")
    print(f"Saved: {path}")
    return man


def main():
    p = argparse.ArgumentParser(description="Convert Manga data_info pkl files to FITS.")
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--outdir", help="write <mangaid>_kl.fits here (native resolution)")
    p.add_argument("--grid_dir", help="write aligned <mangaid>_grid_image/_grid_velmap.fits "
                                      "on the model grid here")
    p.add_argument("--fov_kpc", type=float, default=30.0)
    p.add_argument("--npix", type=int, default=None,
                   help="grid size (default: from --checkpoint, else 128)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--pattern", default="data_info-*.pkl")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict_only", action="store_true",
                   help="pkl files sheared by manga_shear.py: use only the strict "
                        "velocity mask (spaxels outside it are left unsheared)")
    a = p.parse_args()
    if not (a.outdir or a.grid_dir):
        p.error("give --outdir and/or --grid_dir")

    files = sorted(glob.glob(os.path.join(a.pkl_dir, a.pattern)))
    print(f"{len(files)} pkl files in {a.pkl_dir}")

    if a.outdir:
        outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, f in enumerate(files, 1):
            try:
                r = convert(f, outdir, a.overwrite)
                if r["status"] == "ok":
                    _warn(r)
            except Exception as exc:
                r = {"src": os.path.basename(f), "status": f"error: {exc}"}
                print(f"  [ERROR] {os.path.basename(f)}: {exc}")
            rows.append(r)
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)} done")
        _write_manifest(rows, outdir / "manifest.csv")

    if a.grid_dir:
        save_grid_files(files, a.grid_dir, a.fov_kpc, a.npix, a.checkpoint, a.overwrite,
                        a.strict_only)

if __name__ == "__main__":
    main()