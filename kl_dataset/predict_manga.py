#!/usr/bin/env python3
"""
predict_manga.py
================
Run a trained TwoStreamShearNet (train_kl_model.py) on real galaxies:
SDSS r-band image + MaNGA DAP H-alpha velocity field.

Two input formats (either or both):
  --data_dir  raw files: <mangaid>_image.fits (or <mangaid>.fits; SDSS frame)
              + <mangaid>_velmap*.fits (DAP MAPS), e.g. from fetch_manga.py;
              redshift and PSF from DRPall (--drpall).
              The same folder may hold <mangaid>_grid_image.fits +
              <mangaid>_grid_velmap.fits (convert_manga_pkl --grid_dir):
              already aligned on the model grid, used without resampling.
  --kl_dir    <mangaid>_kl.fits made by convert_manga_pkl.py from the
              data_info-*.pkl files; everything needed is in the file.
If a galaxy is present in both, both versions are run (column `source`),
which is a handy cross-check of the two routes.

For every galaxy and every velocity-mask scheme in --vel_masks it

  1. builds a square grid centred on the galaxy spanning --fov_kpc (physical,
     Planck15 = TNG's cosmology) with the checkpoint's npix -- the same field
     the TNG images had before the central crop;
  2. resamples the image (flux-conserving) and the masked H-alpha velocity
     field onto that grid;
  3. removes residual sky and neighbouring sources from the image, since the
     TNG images are noiseless and isolated (--clean_nsigma 0 disables);
  4. adds only as much Gaussian smoothing as needed so the total resolution
     matches the training smooth_sigma;
  5. writes the maps in the training directory layout and loads them with
     KLShearDataset itself, so normalisation, crop and the v_obs / v_asym /
     v_sym channels come from exactly the training code;
  6. predicts (g+, g×), also on the 8 rotations/flips of the input mapped back
     to the original frame (a self-consistency check), and makes SmoothGrad
     saliency maps for --n_saliency galaxies.

Velocity masks (velocity_mask / load_converted):
  none   : every spaxel with a measurement, no quality cuts
  dap    : DAP velocity mask clean (raw route: plus H-alpha flux S/N cut)
  strict : colleague's Manga.get_data_info mask: DAP clean, H-alpha A/N cut,
           largest coherent region

Shear frame: predictions are in the pixel frame of the grid, North up and
East LEFT (x increases toward the West).  In the convention with x toward
the East, g× changes sign.

Usage
-----
  python predict_manga.py --checkpoint kl_dataset/model_output/best_model.pt \
      --kl_dir /path/to/KL_repo/data/manga
  python predict_manga.py --checkpoint ... --data_dir /content \
      --drpall /content/drpall-v3_1_1.fits          # raw FITS route
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from reproject import reproject_exact, reproject_interp
from scipy.ndimage import binary_dilation, gaussian_filter, label

import train_kl_model
from train_kl_model import KLShearDataset, TwoStreamShearNet

# The training data on disk use galaxy_<id>/ (no draw suffix) while
# KLShearDataset looks in galaxy_<id>_draw<NNNN>/.  Accept both layouts.
_orig_gal_dir = train_kl_model._gal_dir


def _gal_dir_any(images_root, snap, sid, draw_idx):
    d = _orig_gal_dir(images_root, snap, sid, draw_idx)
    plain = Path(images_root) / f"snap{snap}" / f"galaxy_{sid}"
    return d if d.exists() or not plain.exists() else plain


train_kl_model._gal_dir = _gal_dir_any

FWHM_TO_SIGMA = 1.0 / 2.3548
CH_NAMES = ["v_obs", "v_asym", "v_sym"]


# ═══════════════════════════════════════════════════════════════════════════════
# Building model inputs from real data
# ═══════════════════════════════════════════════════════════════════════════════

def line_channel(hdr, line):
    """Index of an emission line in a DAP multi-channel extension (C01, C02 …)."""
    for k, v in hdr.items():
        if k.startswith("C") and k[1:].isdigit() and isinstance(v, str) \
                and v.strip() == line:
            return int(k[1:]) - 1
    raise KeyError(f"{line!r} not found in {hdr.get('EXTNAME')} channel list")


def grid_wcs(ra, dec, npix, pix_arcsec):
    """TAN grid centred on (ra, dec), North up, East left.
    Centre at 0-based pixel (npix-1)/2, same as apply_shear_numpy."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(npix + 1) / 2.0] * 2
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    return w


def clean_image(img, nsigma, grow_px=2):
    """Subtract residual sky; keep only the source touching the centre."""
    good = np.isfinite(img)
    _, med, _ = sigma_clipped_stats(img[good], sigma=3)
    img = np.where(good, img - med, 0.0)
    if nsigma <= 0:
        return img
    sm = gaussian_filter(img, 1.0)
    _, _, std = sigma_clipped_stats(sm[good], sigma=3)
    seg, _ = label(sm > nsigma * std)
    c = img.shape[0] // 2
    ids = np.unique(seg[c - 2:c + 3, c - 2:c + 3])
    ids = ids[ids > 0]
    if ids.size == 0:
        print("    [WARN] no source detected at the centre; image left uncleaned")
        return img
    keep = binary_dilation(np.isin(seg, ids), iterations=grow_px)
    return np.where(keep, img, 0.0)


def velocity_mask(maps, ch, mode, cfg):
    """
    Spaxels to keep (True) in the DAP H-alpha velocity map.

    none   : every spaxel with IFU coverage (DAP bit 0, NOCOV, unset); no
             quality cuts at all.
    dap    : DAP velocity mask == 0 and H-alpha flux S/N >= --min_ha_snr.
    strict : as in Manga.get_data_info (colleague's class): DAP velocity
             mask == 0, H-alpha amplitude-to-noise (EMLINE_GANR) >=
             --ha_anr_cut, then only the largest 8-connected region.
    """
    vmask   = maps["EMLINE_GVEL_MASK"].data[ch]
    covered = (vmask & 1) == 0
    if mode == "none":
        return covered
    good = covered & (vmask == 0)
    if mode == "dap":
        snr = maps["EMLINE_GFLUX"].data[ch] * np.sqrt(maps["EMLINE_GFLUX_IVAR"].data[ch])
        return good & (snr >= cfg["min_ha_snr"])
    good &= maps["EMLINE_GANR"].data[ch] >= cfg["ha_anr_cut"]
    seg, n = label(good, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        print("    [WARN] strict mask left no spaxels")
        return good
    counts = np.bincount(seg.ravel()); counts[0] = 0
    return seg == counts.argmax()


def load_dap_pair(img_path, maps_path, drp, cfg):
    """Raw route: SDSS frame + DAP MAPS file (+ DRPall for z and PSF)."""
    with fits.open(maps_path) as maps:
        h0   = maps[0].header
        pifu = str(h0["PLATEIFU"]).strip()
        row  = drp.get(pifu)
        if row is None:
            raise KeyError(f"{pifu} not in DRPall")
        z = next((float(row[c]) for c in ("nsa_z", "z")
                  if c in row.colnames and row[c] > 0), None)
        if z is None:
            raise ValueError(f"{pifu}: no valid redshift in DRPall")
        gv   = maps["EMLINE_GVEL"]
        ch   = line_channel(gv.header, cfg["line"])
        vel  = gv.data[ch].astype(np.float64)
        keep = velocity_mask(maps, ch, cfg["vel_mask"], cfg)
        ifu  = maps["SPX_MFLUX"].data > 0
        vwcs = WCS(gv.header).celestial
        ra, dec = float(h0["OBJRA"]), float(h0["OBJDEC"])

    with fits.open(img_path) as hd:
        img, ihdr = hd[0].data.astype(np.float64), hd[0].header
    iwcs = WCS(ihdr)
    if not iwcs.has_celestial:
        raise ValueError(f"{img_path}: header has no celestial WCS")

    vel_fwhm = (float(row["rfwhm"]) if "rfwhm" in row.colnames and row["rfwhm"] > 0
                else cfg["manga_fwhm"])
    return dict(plateifu=pifu, ra=ra, dec=dec, z=z, img=img, img_wcs=iwcs,
                vel=vel, vel_wcs=vwcs, keep=keep, ifu=ifu,
                img_fwhm=cfg["sdss_fwhm"], vel_fwhm=vel_fwhm)


CONVERTED_MASK = {"none": "MASK_COVER", "dap": "MASK_DAP", "strict": "MASK_STRICT"}


def load_converted(path, cfg):
    """pkl route: one <mangaid>_kl.fits from convert_manga_pkl.py.
    The A/N cut of the strict mask was fixed when the pkl was made, so
    --ha_anr_cut / --min_ha_snr do not apply here."""
    with fits.open(path) as hd:
        h0    = hd[0].header
        img   = hd["IMAGE"].data.astype(np.float64)
        iwcs  = WCS(hd["IMAGE"].header)
        vel   = hd["VEL"].data.astype(np.float64)
        vwcs  = WCS(hd["VEL"].header)
        cover = hd["MASK_COVER"].data.astype(bool)
        keep  = hd[CONVERTED_MASK[cfg["vel_mask"]]].data.astype(bool) & cover
    return dict(plateifu=str(h0["PLATEIFU"]), ra=float(h0["RA"]),
                dec=float(h0["DEC"]), z=float(h0["REDSHIFT"]),
                img=img, img_wcs=iwcs, vel=vel, vel_wcs=vwcs, keep=keep,
                ifu=cover, img_fwhm=float(h0.get("IMGPSF", cfg["sdss_fwhm"])),
                vel_fwhm=float(h0.get("VELPSF", cfg["manga_fwhm"])))


GRID_EXT = {"none": "VEL_ALL", "dap": "VEL_DAP", "strict": 0}


def load_grid(img_path, vel_path, cfg):
    """
    Aligned route: <mangaid>_grid_image.fits + <mangaid>_grid_velmap.fits
    from convert_manga_pkl.save_grid.  Already on the model grid, so
    build_galaxy uses them as they are when fov/npix match the checkpoint.
    """
    with fits.open(img_path) as hi, fits.open(vel_path) as hv:
        h   = hi[0].header
        img = hi[0].data.astype(np.float64)
        iwcs = WCS(h).celestial
        vel = hv[GRID_EXT[cfg["vel_mask"]]].data.astype(np.float64)
        vwcs = WCS(hv[0].header).celestial
        ifu = np.isfinite(hv["VEL_ALL"].data)
    return dict(plateifu=str(h["PLATEIFU"]), ra=float(h["RA"]), dec=float(h["DEC"]),
                z=float(h["REDSHIFT"]), img=img, img_wcs=iwcs, vel=vel,
                vel_wcs=vwcs, keep=np.isfinite(vel), ifu=ifu,
                img_fwhm=float(h.get("IMGPSF", cfg["sdss_fwhm"])),
                vel_fwhm=float(h.get("VELPSF", cfg["manga_fwhm"])))


def same_grid(w1, shape1, w2, shape2):
    """True if two celestial WCS + shapes describe the same pixel grid."""
    return (tuple(shape1) == tuple(shape2)
            and np.allclose(w1.wcs.crval, w2.wcs.crval, rtol=0, atol=1e-9)
            and np.allclose(w1.wcs.crpix, w2.wcs.crpix, rtol=0, atol=1e-6)
            and np.allclose(w1.pixel_scale_matrix, w2.pixel_scale_matrix,
                            rtol=1e-6, atol=0))


def build_galaxy(src, cfg):
    """Return (photo, vel, wcs, info) on the pre-crop training grid."""
    npix, fov = cfg["npix"], cfg["fov_kpc"]
    kpc_as = Planck15.kpc_proper_per_arcmin(src["z"]).to(u.kpc / u.arcsec).value
    pix_as = fov / npix / kpc_as
    wcs    = grid_wcs(src["ra"], src["dec"], npix, pix_as)
    shape  = (npix, npix)

    # ── Velocity ────────────────────────────────────────────────────────────
    v = src["vel"].copy()
    v[~src["keep"]] = np.nan
    if same_grid(src["vel_wcs"], v.shape, wcs, shape):
        vel = v                                    # already on the model grid
    else:
        vel, _ = reproject_interp((v, src["vel_wcs"]), wcs, shape_out=shape,
                                  order="bilinear")
    if cfg["circular_ifu"]:
        # Largest circle about the centre that lies inside the IFU footprint.
        # Removes the hexagon's fixed on-sky orientation, which TNG maps lack.
        ifu, _ = reproject_interp((src["ifu"].astype(np.float64), src["vel_wcs"]),
                                  wcs, shape_out=shape, order="nearest-neighbor")
        yy, xx  = np.indices(shape)
        r       = np.hypot(xx - (npix - 1) / 2, yy - (npix - 1) / 2)
        outside = ~(np.nan_to_num(ifu) > 0.5)
        r_max   = r[outside].min() if outside.any() else r.max()
        vel[r >= r_max] = np.nan
    vel_fill = float(np.isfinite(vel).mean())

    # ── Image ───────────────────────────────────────────────────────────────
    if same_grid(src["img_wcs"], src["img"].shape, wcs, shape):
        photo = src["img"].copy()                  # already on the model grid
        cov   = np.isfinite(photo).astype(float)
    else:
        centre = SkyCoord(src["ra"] * u.deg, src["dec"] * u.deg)
        cut = Cutout2D(src["img"], centre, size=1.5 * fov / kpc_as * u.arcsec,
                       wcs=src["img_wcs"], mode="partial", fill_value=np.nan)
        photo, cov = reproject_exact((cut.data, cut.wcs), wcs, shape_out=shape)
    img_cov = float(np.nanmean(np.where(np.isfinite(photo), cov, 0)))
    photo = clean_image(photo, cfg["clean_nsigma"])

    # ── Resolution matching ─────────────────────────────────────────────────
    # Training smoothing acts after the crop, in pixels of fov*crop/npix kpc.
    kpc_px_model = fov * cfg["crop"] / npix
    kpc_px_grid  = fov / npix
    psf_fwhm = {"photo": src["img_fwhm"], "vel": src["vel_fwhm"]}

    info = {"plateifu": src["plateifu"], "z": src["z"],
            "vel_mask": cfg["vel_mask"], "n_spaxels": int(src["keep"].sum()),
            "arcsec_per_px": pix_as, "kpc_per_px_model": kpc_px_model,
            "vel_fill_frac": vel_fill, "img_coverage": img_cov,
            "img_psf_fwhm_arcsec": src["img_fwhm"],
            "vel_psf_fwhm_arcsec": src["vel_fwhm"]}
    extra = {}
    for s in ("photo", "vel"):
        real_kpc  = psf_fwhm[s] * FWHM_TO_SIGMA * kpc_as
        smoothed  = cfg["smooth_target"] in ("both", s)
        train_kpc = cfg["smooth_sigma"] * kpc_px_model if smoothed else 0.0
        add_kpc   = (np.sqrt(max(train_kpc ** 2 - real_kpc ** 2, 0.0))
                     if cfg["psf_mode"] == "match" else 0.0)
        extra[s]  = add_kpc / kpc_px_grid
        info[f"psf_sigma_real_px_{s}"]  = real_kpc / kpc_px_model
        info[f"psf_sigma_train_px_{s}"] = train_kpc / kpc_px_model

    if extra["photo"] > 0:
        photo = gaussian_filter(photo, extra["photo"], mode="nearest")
    if extra["vel"] > 0:
        # Training fills empty velmap pixels with 0 before smoothing; same here.
        vel = gaussian_filter(np.nan_to_num(vel, nan=0.0), extra["vel"],
                              mode="nearest")
    return photo, vel, wcs, info


def write_inputs(root, sid, photo, vel, wcs, info):
    """Save in the layout KLShearDataset expects (snap0, draw 0)."""
    d = root / "snap0" / f"galaxy_{sid}_draw0000"
    d.mkdir(parents=True, exist_ok=True)
    hdr = wcs.to_header()
    hdr["MANGAID"]  = info["mangaid"]
    hdr["PLATEIFU"] = info["plateifu"]
    hdr["REDSHIFT"] = info["z"]
    for name, arr in [("image_sheared", photo), ("image_original", photo),
                      ("velmap_original", vel)]:
        fits.writeto(d / f"galaxy_{sid}_{name}.fits",
                     arr.astype(np.float32), hdr, overwrite=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Model: prediction with rotation/flip consistency, and saliency
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_dihedral(model, photo, vel, device):
    """
    Predict on the 8 rotations/flips of the input and map each prediction
    back to the original frame.  Shear is spin-2: a 90° rotation flips the
    sign of both components, a mirror flips the sign of g×.
    Row 0 is the untransformed input.
    """
    out = []
    for k in range(4):
        for flip in (False, True):
            p = torch.rot90(photo, k, dims=(-2, -1))
            v = torch.rot90(vel,   k, dims=(-2, -1))
            if flip:
                p, v = torch.flip(p, dims=(-1,)), torch.flip(v, dims=(-1,))
            g1, g2 = model(p[None].contiguous().to(device),
                           v[None].contiguous().to(device))
            s = (-1) ** k
            out.append([s * g1.item(), s * (-1 if flip else 1) * g2.item()])
    return np.array(out)


def smoothgrad(model, photo, vel, device, n, noise):
    """SmoothGrad |∂g/∂input| for g+ and g×."""
    photo = photo[None].contiguous().to(device)
    vel   = vel[None].contiguous().to(device)
    sal = {t: [torch.zeros_like(photo), torch.zeros_like(vel)] for t in ("g1", "g2")}
    n = max(1, n)
    for _ in range(n):
        p_in = (photo + noise * torch.randn_like(photo)).requires_grad_(True)
        v_in = (vel   + noise * torch.randn_like(vel)).requires_grad_(True)
        g1, g2 = model(p_in, v_in)
        for t, o in (("g1", g1), ("g2", g2)):
            gp, gv = torch.autograd.grad(o.sum(), [p_in, v_in],
                                         retain_graph=(t == "g1"))
            sal[t][0] += gp.abs()
            sal[t][1] += gv.abs()
    return {f"photo_{t}": (sal[t][0][0].sum(0) / n).cpu().numpy() for t in sal} | \
           {f"vel_{t}":   (sal[t][1][0] / n).cpu().numpy()        for t in sal}


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _show(ax, img, cmap, title, sal=False):
    if sal:
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0,
                  vmax=np.percentile(img, 99.5) or 1.0)
    else:
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_inputs(rows, out_path):
    """Model inputs exactly as the network sees them: real vs. training."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.6 * len(rows)),
                             squeeze=False)
    for ax, (tag, photo, vel) in zip(axes, rows):
        _show(ax[0], photo, "gray", f"{tag}\nphoto")
        for c in range(3):
            _show(ax[1 + c], vel[c], "RdBu_r", CH_NAMES[c])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_saliency(results, out_path):
    fig, axes = plt.subplots(len(results), 8, figsize=(20, 2.8 * len(results)),
                             squeeze=False)
    for ax, r in zip(axes, results):
        _show(ax[0], r["photo"], "gray",
              f"{r['mangaid']}  g+={r['g1']:+.3f}  g×={r['g2']:+.3f}\nphoto")
        for c in range(3):
            _show(ax[1 + c], r["vel"][c], "RdBu_r", CH_NAMES[c])
        _show(ax[4], r["sal"]["photo_g1"],       "magma", "∂g+/∂photo", sal=True)
        _show(ax[5], r["sal"]["vel_g1"].sum(0),  "magma", "∂g+/∂vel",   sal=True)
        _show(ax[6], r["sal"]["photo_g2"],       "magma", "∂g×/∂photo", sal=True)
        _show(ax[7], r["sal"]["vel_g2"].sum(0),  "magma", "∂g×/∂vel",   sal=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def training_examples(ckpt_dir, ta, n, seed):
    """A few training-set inputs (as the model sees them) for comparison."""
    csv, root = ckpt_dir / "split_test.csv", Path(ta.get("images_root", ""))
    if n <= 0 or not csv.exists() or not root.exists():
        return []
    df = pd.read_csv(csv).drop_duplicates(["subhalo_id", "snap"])
    df = df.sample(min(5 * n, len(df)), random_state=seed).reset_index(drop=True)
    ds = KLShearDataset(df, root, ta.get("npix", 128),
                        ta.get("use_original_image", False),
                        smooth_sigma=ta.get("smooth_sigma", 0.0),
                        smooth_target=ta.get("smooth_target", "both"),
                        crop_frac=ta.get("crop_frac", 1.0))
    out, n_fail = [], 0
    for i in range(len(ds)):
        if len(out) >= n:
            break
        try:
            p, v, _ = ds[i]
            out.append((f"TNG {int(df.loc[i, 'subhalo_id'])} (training)",
                        p[0].numpy(), v.numpy()))
        except Exception as exc:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [WARN] training example {i} skipped: {exc}")
    if n_fail:
        msg = (f"  [WARN] {n_fail} training examples could not be read; "
               f"the comparison figure shows {len(out)}.")
        lfs = [f for f in sorted(root.glob("snap*/galaxy_*/*.fits"))[:20]
               if _is_lfs_pointer(f)]
        if lfs:
            msg += (f"\n         {lfs[0]} is a Git LFS pointer, not a FITS file:"
                    f"\n         run `git lfs install && git lfs pull` in the repo.")
        print(msg)
    return out


def _is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Predict shear for MaNGA galaxies.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default=None,
                   help="Raw route: folder with <mangaid>_image.fits + <mangaid>_velmap*.fits")
    p.add_argument("--kl_dir", default=None,
                   help="pkl route: folder with <mangaid>_kl.fits from convert_manga_pkl.py")
    p.add_argument("--max_galaxies", type=int, default=0,
                   help="Only use the first N galaxies (0 = all), for quick tests")
    p.add_argument("--drpall", default="/content/drpall-v3_1_1.fits")
    p.add_argument("--outdir", default="/content/manga_predictions")
    p.add_argument("--fov_kpc", type=float, default=30.0,
                   help="Field of view before the crop (TNG renders: 30 kpc)")
    p.add_argument("--line", default="Ha-6564", help="DAP emission-line channel")
    p.add_argument("--min_ha_snr", type=float, default=3.0,
                   help="Mask spaxels with H-alpha flux S/N below this")
    p.add_argument("--vel_masks", nargs="+", default=["none", "strict"],
                   choices=["none", "dap", "strict"],
                   help="Velocity-map masking schemes to run (see velocity_mask)")
    p.add_argument("--ha_anr_cut", type=float, default=5.0,
                   help="H-alpha amplitude-to-noise cut for the strict mask")
    p.add_argument("--circular_ifu", action="store_true",
                   help="Trim the velocity map to the largest centred circle "
                        "inside the IFU hexagon")
    p.add_argument("--clean_nsigma", type=float, default=2.0,
                   help="Detection threshold for isolating the galaxy (0 = off)")
    p.add_argument("--sdss_fwhm", type=float, default=1.4,
                   help="SDSS seeing FWHM [arcsec]")
    p.add_argument("--manga_fwhm", type=float, default=2.5,
                   help="MaNGA PSF FWHM if DRPall has no rfwhm [arcsec]")
    p.add_argument("--psf_mode", choices=["match", "train", "none"], default="match",
                   help="match: add only the smoothing needed to reach the "
                        "training resolution; train: apply the training "
                        "smooth_sigma on top of the real PSF; none: no smoothing")
    p.add_argument("--n_saliency", type=int, default=12,
                   help="Galaxies per mask scheme with saliency maps and input "
                        "plots (random subset; -1 = all)")
    p.add_argument("--smoothgrad_samples", type=int, default=16)
    p.add_argument("--smoothgrad_noise", type=float, default=0.05)
    p.add_argument("--n_train_compare", type=int, default=3,
                   help="Training galaxies to show next to the real inputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true",
                   help="One line per galaxy (default: progress every 100)")
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    out_dir = Path(a.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Checkpoint and training settings ────────────────────────────────────
    ckpt_path = Path(a.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta   = ckpt.get("args", {})
    cfg = dict(npix=ta.get("npix", 128), crop=ta.get("crop_frac", 1.0),
               smooth_sigma=ta.get("smooth_sigma", 0.0),
               smooth_target=ta.get("smooth_target", "both"),
               fov_kpc=a.fov_kpc, line=a.line, min_ha_snr=a.min_ha_snr, ha_anr_cut=a.ha_anr_cut,
               clean_nsigma=a.clean_nsigma, circular_ifu=a.circular_ifu, sdss_fwhm=a.sdss_fwhm,
               manga_fwhm=a.manga_fwhm, psf_mode=a.psf_mode)
    min_fill = ta.get("min_vel_fill", 0.0)
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})")
    print(f"  npix={cfg['npix']}  crop_frac={cfg['crop']}  "
          f"smooth_sigma={cfg['smooth_sigma']} ({cfg['smooth_target']})  "
          f"min_vel_fill={min_fill}  fov={a.fov_kpc} kpc  psf_mode={a.psf_mode}")

    model = TwoStreamShearNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    items = find_inputs(a)
    if not items:
        raise SystemExit("No galaxies found: give --data_dir and/or --kl_dir.")
    drp = {}
    if any(it["source"] == "dap_fits" for it in items):
        t   = Table.read(a.drpall, hdu=1)
        drp = {str(p).strip(): r for p, r in zip(t["plateifu"], t)}
    print(f"\n{len(items)} galaxies: "
          + ", ".join(f"{k} {v}" for k, v in
                      pd.Series([it['source'] for it in items]).value_counts().items()))

    train_rows = training_examples(ckpt_path.parent, ta, a.n_train_compare, a.seed)
    all_df = []
    for mode in a.vel_masks:
        print(f"\n══ Velocity mask: {mode} " + "═" * 50)
        df, results = run_mode(mode, items, drp, cfg, ta, min_fill, model,
                               device, out_dir, a)
        if df is None:
            continue
        all_df.append(df)
        if results:
            real = [(f"{r['mangaid']} ({r['source']}, {mode})", r["photo"], r["vel"])
                    for r in results]
            plot_inputs(real + train_rows,
                        out_dir / f"inputs_real_vs_training_{mode}.png")
            plot_saliency(results, out_dir / f"saliency_manga_{mode}.png")

    if not all_df:
        raise SystemExit("No galaxies could be built.")
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(out_dir / "predictions_manga.csv", index=False)
    print(f"\n  Saved: {out_dir / 'predictions_manga.csv'}")
    summarize(out, a.vel_masks, min_fill, out_dir)


def find_inputs(a):
    """Galaxies from the raw route (--data_dir) and the pkl route (--kl_dir)."""
    items = []
    if a.data_dir:
        for vf in sorted(glob.glob(os.path.join(a.data_dir, "*_grid_velmap.fits"))):
            mangaid = os.path.basename(vf)[:-len("_grid_velmap.fits")]
            img = os.path.join(a.data_dir, f"{mangaid}_grid_image.fits")
            if os.path.exists(img):
                items.append({"mangaid": mangaid, "source": "grid",
                              "img": img, "maps": vf})
        for vf in sorted(glob.glob(os.path.join(a.data_dir, "*_velmap*.fits"))):
            if "_grid_velmap" in vf:
                continue
            mangaid = os.path.basename(vf).split("_velmap")[0]
            cands = [os.path.join(a.data_dir, f"{mangaid}{s}.fits")
                     for s in ("_image", "")]        # fetch_manga.py, old name
            img = next((c for c in cands if os.path.exists(c)), None)
            if img:
                items.append({"mangaid": mangaid, "source": "dap_fits",
                              "img": img, "maps": vf})
            else:
                print(f"{mangaid}: no image ({mangaid}_image.fits), skipping")
    if a.kl_dir:
        for f in sorted(glob.glob(os.path.join(a.kl_dir, "*_kl.fits"))):
            items.append({"mangaid": os.path.basename(f)[:-len("_kl.fits")],
                          "source": "pkl", "path": f})
    if a.max_galaxies > 0:
        items = items[:a.max_galaxies]
    return items


def run_mode(mode, items, drp, cfg, ta, min_fill, model, device, out_dir, a):
    """Build inputs with one velocity mask, then predict (+ saliency subset)."""
    cfg  = {**cfg, "vel_mask": mode}
    root = out_dir / f"inputs_{mode}"
    rows, n_low = [], 0
    for k, it in enumerate(items, 1):
        try:
            if it["source"] == "dap_fits":
                src = load_dap_pair(it["img"], it["maps"], drp, cfg)
            elif it["source"] == "grid":
                src = load_grid(it["img"], it["maps"], cfg)
            else:
                src = load_converted(it["path"], cfg)
            photo, vel, wcs, info = build_galaxy(src, cfg)
        except Exception as exc:
            print(f"  [ERROR] {it['mangaid']} ({it['source']}): {exc}")
            continue
        info.update(mangaid=it["mangaid"], source=it["source"])
        sid = len(rows)
        write_inputs(root, sid, photo, vel, wcs, info)
        n_low += info["vel_fill_frac"] < min_fill
        if a.verbose:
            print(f"  {it['mangaid']:<10} {it['source']:<8} {info['plateifu']:<11} "
                  f"z={info['z']:.4f}  spaxels={info['n_spaxels']:4d}  "
                  f"fill={info['vel_fill_frac']:.2f}  img cov={info['img_coverage']:.2f}  "
                  f"PSF σ px photo {info['psf_sigma_real_px_photo']:.1f}/"
                  f"{info['psf_sigma_train_px_photo']:.1f} vel "
                  f"{info['psf_sigma_real_px_vel']:.1f}/{info['psf_sigma_train_px_vel']:.1f}"
                  " (real/train)")
        elif k % 100 == 0 or k == len(items):
            print(f"  built {k}/{len(items)}")
        rows.append({"subhalo_id": sid, "snap": 0, "draw_idx": 0,
                     "g1": 0.0, "g2": 0.0, **info})
    if not rows:
        return None, []
    df = pd.DataFrame(rows)
    if n_low:
        print(f"  [WARN] {n_low}/{len(df)} galaxies have velmap fill below the "
              f"training cut ({min_fill}); see column vel_fill_frac.")
    psf = df[["psf_sigma_real_px_photo", "psf_sigma_real_px_vel"]].median()
    print(f"  median real PSF σ in model px: photo {psf.iloc[0]:.2f}, vel {psf.iloc[1]:.2f} "
          f"(training: {cfg['smooth_sigma']})")

    smooth_in_ds = cfg["smooth_sigma"] if a.psf_mode == "train" else 0.0
    ds = KLShearDataset(df, root, cfg["npix"], ta.get("use_original_image", False),
                        smooth_sigma=smooth_in_ds,
                        smooth_target=cfg["smooth_target"], crop_frac=cfg["crop"])

    rng = np.random.default_rng(a.seed)
    n_sal = len(df) if a.n_saliency < 0 else min(a.n_saliency, len(df))
    sal_idx = set(rng.choice(len(df), size=n_sal, replace=False).tolist())

    print(f"\n── Predictions [{mode}] (pixel frame: x = West, y = North) ──────")
    results = []
    for i in range(len(ds)):
        photo, vel, _ = ds[i]
        d8 = predict_dihedral(model, photo, vel, device)
        df.loc[i, "g1_pred"] = d8[0, 0]
        df.loc[i, "g2_pred"] = d8[0, 1]
        df.loc[i, "g1_dihedral_mean"] = d8[:, 0].mean()
        df.loc[i, "g2_dihedral_mean"] = d8[:, 1].mean()
        df.loc[i, "g1_dihedral_std"]  = d8[:, 0].std()
        df.loc[i, "g2_dihedral_std"]  = d8[:, 1].std()

        if i in sal_idx:
            sal = smoothgrad(model, photo, vel, device,
                             a.smoothgrad_samples, a.smoothgrad_noise)
            results.append({"mangaid": df.loc[i, "mangaid"],
                            "source": df.loc[i, "source"], "g1": d8[0, 0],
                            "g2": d8[0, 1], "photo": photo[0].numpy(),
                            "vel": vel.numpy(), "sal": sal})
            for t in ("g1", "g2"):
                share = sal[f"vel_{t}"].sum(axis=(1, 2))
                share = share / share.sum()
                for n, s_ in zip(CH_NAMES, share):
                    df.loc[i, f"sal_share_{n}_{t}"] = s_
                tot_p, tot_v = sal[f"photo_{t}"].sum(), sal[f"vel_{t}"].sum()
                df.loc[i, f"sal_photo_frac_{t}"] = tot_p / (tot_p + tot_v)

        if a.verbose:
            print(f"  {df.loc[i, 'mangaid']:<10}  g+ = {d8[0,0]:+.4f}  "
                  f"(8-fold {d8[:,0].mean():+.4f} ± {d8[:,0].std():.4f})   "
                  f"g× = {d8[0,1]:+.4f}  "
                  f"(8-fold {d8[:,1].mean():+.4f} ± {d8[:,1].std():.4f})")
        elif (i + 1) % 100 == 0 or i + 1 == len(ds):
            print(f"  predicted {i + 1}/{len(ds)}")

    df = df.drop(columns=["g1", "g2", "snap", "draw_idx"]).rename(
        columns={"subhalo_id": "index"})
    return df, results


def summarize(out, modes, min_fill, out_dir):
    """Sample means (expected ~0: real cosmic shear is ~0.01) and mask comparison."""
    modes = [m for m in modes if m in set(out["vel_mask"])]
    print("\n── Sample summary (mean ± standard error over galaxies) ─────────")
    for m in modes:
        for src, sub in out[out["vel_mask"] == m].groupby("source"):
            for tag, cut in (("all", sub),
                             (f"fill≥{min_fill}", sub[sub["vel_fill_frac"] >= min_fill])):
                if len(cut) < 2:
                    continue
                se = lambda c: cut[c].std(ddof=1) / np.sqrt(len(cut))
                print(f"  [{m:<6} {src:<8} {tag:<9} n={len(cut):4d}]  "
                      f"g+ {cut['g1_pred'].mean():+.4f} ± {se('g1_pred'):.4f}   "
                      f"g× {cut['g2_pred'].mean():+.4f} ± {se('g2_pred'):.4f}   "
                      f"(8-fold: g+ {cut['g1_dihedral_mean'].mean():+.4f}, "
                      f"g× {cut['g2_dihedral_mean'].mean():+.4f})")

    if len(modes) > 1:
        cols = ["g1_pred", "g2_pred", "g1_dihedral_mean", "g2_dihedral_mean",
                "vel_fill_frac", "n_spaxels"]
        wide = out.pivot_table(index=["mangaid", "source"], columns="vel_mask",
                               values=cols, aggfunc="first")
        wide.columns = [f"{c}_{m}" for c, m in wide.columns]
        wide.to_csv(out_dir / "predictions_manga_mask_comparison.csv")
        print(f"\n  Saved: {out_dir / 'predictions_manga_mask_comparison.csv'}")
        base = modes[0]
        for m in modes[1:]:
            for c, sym in (("g1_pred", "g+"), ("g2_pred", "g×")):
                x, y = wide[f"{c}_{base}"], wide[f"{c}_{m}"]
                ok = x.notna() & y.notna()
                if ok.sum() > 2:
                    r = np.corrcoef(x[ok], y[ok])[0, 1]
                    print(f"  {sym}: {base} vs {m}: correlation r = {r:+.2f}, "
                          f"rms difference = {np.sqrt(np.mean((x[ok] - y[ok])**2)):.4f}")

if __name__ == "__main__":
    main()