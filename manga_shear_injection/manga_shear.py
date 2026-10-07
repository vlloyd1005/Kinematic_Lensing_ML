"""Metacal-style shear injection for MaNGA r-band image + Halpha velocity maps (standalone).

Operation on a linear map f (image, Halpha flux M0, or flux-weighted velocity M1 = M0*v):
    f' = P_dil * Shear_g( P^-1 * f )
with P the (Gaussian) PSF and P_dil = P dilated by a factor dil (1.2 for the image, 1.15 for
Halpha by default).  The velocity map is v' = M1'/M0'.

The operation is applied (method='imcom', default) as a configuration-space IMCOM solve using
InterpMatrix from PyIMCOM's pyimcom.meta.ginterp by Christopher M. Hirata (MIT license),
copied unmodified in _vendor/ -- see _vendor/README.md for provenance and citations, and
imcom_shear.py for how it is used here.  method='galsim' is the original Fourier-space galsim
implementation, kept only to reproduce the first blind set; it rings (see README).  Anisotropic noise from the shear is symmetrised metacal-"fixnoise" style:
a fresh noise field with the data's statistics is rotated 90 deg, given the same operation,
rotated back and added.  g = 0 gives the no-shear control (same smoothing and noise as
every sheared set).

Conventions: world coords u = +East (dRA cos dec), v = +North, arcsec. g1 > 0 stretches
E-W, g2 > 0 stretches along PA = 45 deg (NE-SW).  Shear is about the galaxy centre.
Pixel-frame equivalents of (g1, g2) for each map's array axes (x = column, y = row) are
written to the truth table; see README.md.

Usage:
    python manga_shear.py noshear --src DIR --out DIR
    python manga_shear.py fixed   --src DIR --out DIR --g1 0.02 --g2 0
    python manga_shear.py random  --src DIR --out DIR --truth truth.csv [--seed N]
    defaults: --method imcom --inpaint gp; the original blind set is --method galsim --inpaint none
"""
import argparse, copy, glob, json, os, zlib
import numpy as np, galsim, joblib
import imcom_shear as ic          # our adapter around PyIMCOM's InterpMatrix (see _vendor/README.md)
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.linalg import cho_factor, cho_solve

G_MAX = 0.03                                  # per component
V_FLOOR = 25.0                                # (km/s)^2 systematic floor in on-disk gas var
HA = 6564.6                                   # Halpha vacuum wavelength, Angstrom
GSP = galsim.GSParams(maximum_fft_size=16384)
FWHM2SIG = 1 / (2 * np.sqrt(2 * np.log(2)))


def dilation(g_max=G_MAX):
    """PSF dilation that covers |g| = sqrt(2) * g_max (both components at their maximum).
    This was the (only) dilation of the original galsim method."""
    return 1 + 2 * np.hypot(g_max, g_max)


DIL = dilation()                  # 1.0849: original galsim method
DIL_IMG, DIL_VEL = 1.2, 1.15      # imcom method: net kernel resolved by the pixel grid (README)
RSEARCH_IMG, RSEARCH_VEL = 6.0, 8.0   # IMCOM search radius (input pixels) per map
_IMCOM_CACHE = {}


def _imcom(shape, J, sigma, g, c, dil, rsearch):
    """Cached IMCOM shear matrix (see imcom_shear.transfer)."""
    key = (tuple(shape), tuple(np.round(np.ravel(J), 12)), round(float(sigma), 12), tuple(np.round(g, 12)),
           tuple(np.round(c, 12)), round(float(dil), 12), float(rsearch))
    if key not in _IMCOM_CACHE:
        if len(_IMCOM_CACHE) > 16:
            _IMCOM_CACHE.clear()
        M, pad, _, _ = ic.transfer(shape, J, sigma, g, c, dil, rsearch)
        _IMCOM_CACHE[key] = (M, pad)
    return _IMCOM_CACHE[key]


# ---------------------------------------------------------------- geometry
def image_jacobian(d):
    """2x2 J: (dE, dN) arcsec per (dx = column, dy = row) image pixel, from the image WCS
    evaluated at the galaxy position."""
    g = d['galaxy']; w = d['image']['par_meta']['wcs']
    c0 = galsim.CelestialCoord(g['RA'] * galsim.degrees, g['Dec'] * galsim.degrees)
    p0 = w.toImage(c0); J = np.zeros((2, 2))
    for k, (dx, dy) in enumerate([(1, 0), (0, 1)]):
        c = w.toWorld(galsim.PositionD(p0.x + dx, p0.y + dy))
        dE = (c.ra - c0.ra).wrap().rad * np.cos(c0.dec.rad) * 206264.806
        dN = (c.dec - c0.dec).rad * 206264.806
        J[:, k] = [dE, dN]
    return J


def vmap_jacobian(gm):
    """2x2 J: (dE, dN) arcsec per (column, row) spaxel, from the RA_grid/Dec_grid offsets."""
    E, N = gm['RA_grid'].astype(float), gm['Dec_grid'].astype(float)
    return np.array([[E[0, 1] - E[0, 0], E[1, 0] - E[0, 0]],
                     [N[0, 1] - N[0, 0], N[1, 0] - N[0, 0]]])


def shear_to_pixel(g1, g2, J):
    """Express a sky-frame shear (u=+E, v=+N) in the pixel frame (x=column, y=row) of a
    map whose pixel->sky Jacobian is J.  Exact for any rotation/reflection J."""
    A = np.linalg.inv(J / np.sqrt(abs(np.linalg.det(J))))       # sky -> pixel, unit scale
    S = galsim.Shear(g1=g1, g2=g2).getMatrix()
    return _shear_from_matrix(A @ S @ np.linalg.inv(A))


def _shear_from_matrix(M):
    """Inverse of galsim.Shear.getMatrix() for a symmetric unit-determinant matrix."""
    M = (M + M.T) / 2 / np.sqrt(np.linalg.det(M))
    # M = [[1+g1, g2],[g2, 1-g1]] / sqrt(1-|g|^2)
    s = (M[0, 0] + M[1, 1]) / 2                                  # = 1/sqrt(1-|g|^2)
    return galsim.Shear(g1=(M[0, 0] - M[1, 1]) / 2 / s, g2=M[0, 1] / s)


# ---------------------------------------------------------------- core operation
def op(arr, wcs, psf, g=(0., 0.), c=(0., 0.), pad_sigma=None, seed=None, dil=DIL, method='imcom',
       rsearch=RSEARCH_VEL):
    """Apply P_dil * Shear(P^-1 * f) about world position c (rel. to array true centre).
    wcs: galsim JacobianWCS (pixel -> world); psf: galsim.Gaussian.
    pad_sigma: pad the array with white noise of this rms (else zeros) so that pixels near the
    array edge are computed from a full neighbourhood; then crop back.
    method: 'imcom' (IMCOM weights, see imcom_shear) or 'galsim' (original, rings)."""
    if method == 'imcom':
        M, pad = _imcom(arr.shape, wcs.jacobian().getMatrix(), psf.sigma, g, c, dil, rsearch)
        return ic.apply(M, arr, pad, pad_sigma, seed)
    if method != 'galsim':
        raise ValueError(f'unknown method {method!r}')
    ny, nx = arr.shape; py, px = ny // 2, nx // 2
    a = np.asarray(arr, dtype=float)
    if pad_sigma:
        big = np.random.default_rng(seed).normal(0, pad_sigma, (ny + 2 * py, nx + 2 * px))
        big[py:py + ny, px:px + nx] = a; a = big
    ii = galsim.InterpolatedImage(galsim.Image(np.ascontiguousarray(a), wcs=wcs),
                                  x_interpolant='lanczos15', gsparams=GSP)
    obj = galsim.Convolve(ii, galsim.Deconvolve(psf))
    if g[0] or g[1]:
        obj = obj.shift(-c[0], -c[1]).shear(g1=g[0], g2=g[1]).shift(c[0], c[1])
    obj = galsim.Convolve(obj, psf.dilate(dil))
    out = obj.drawImage(nx=a.shape[1], ny=a.shape[0], wcs=wcs, method='no_pixel').array
    return out[py:py + ny, px:px + nx] if pad_sigma else out


def transfer_matrix(shape, J, psf, g, c=(0., 0.), dil=DIL, rotated=False, pad=0, R=10, os=8):
    """(method='galsim' only.)  Sparse linear map from input-pixel noise to output pixels for op() (rotated=False) or for
    the fixnoise term rot90^-1(op(rot90(.))) (rotated=True, shear about the array centre).
    The shear moves the content of input pixel s to T(s) = c + S (x_s - c) (sub-pixel to ~1
    pixel), so the weight from s to output pixel p is K(p - T(s)), with K the kernel of
    P_dil * Shear(P^-1) evaluated at that fractional offset (drawn os-times oversampled).
    Sources cover the grid extended by `pad` pixels (rows/cols of np.pad(arr, pad)).
    J: pixel (col, row) -> world (u, v) Jacobian.  Returns a CSR matrix (ny*nx, (ny+2pad)*(nx+2pad))."""
    import scipy.sparse as sp
    from scipy.ndimage import map_coordinates, spline_filter
    ny, nx = shape
    k = galsim.Convolve(galsim.Deconvolve(psf).shear(g1=g[0], g2=g[1]), psf.dilate(dil), gsparams=GSP)
    nf = (2 * R + 1) * os + 1; cf = (nf - 1) / 2
    Kf = k.drawImage(nx=nf, ny=nf, wcs=galsim.JacobianWCS(*(np.asarray(J) / os).ravel()), method='no_pixel').array * os**2
    Kc = spline_filter(Kf, order=3)                # cubic-spline coefficients, computed once
    S = galsim.Shear(g1=g[0], g2=g[1]).getMatrix(); Ji = np.linalg.inv(J)
    pr, pc = (ny - 1) / 2, (nx - 1) / 2
    def T(r, cc, cw):                          # pixel (row, col) -> sheared pixel (row, col)
        x = np.asarray(J) @ np.vstack([cc - pc, r - pr]) - np.asarray(cw)[:, None]
        q = Ji @ (S @ x + np.asarray(cw)[:, None])
        return q[1] + pr, q[0] + pc
    sr, sc = np.mgrid[-pad:ny + pad, -pad:nx + pad]; sr = sr.ravel().astype(float); sc = sc.ravel().astype(float)
    src = np.arange(sr.size)
    if rotated:                                # rot90: (r, c) -> (n-1-c, r); inverse: (i, j) -> (j, n-1-i)
        assert ny == nx; n = nx
        i, j = T(n - 1 - sc, sr, (0., 0.)); yr, yc = j, n - 1 - i
    else:
        yr, yc = T(sr, sc, c)
    br, bc = np.round(yr).astype(int), np.round(yc).astype(int)
    d = np.arange(-R, R + 1)
    orow = (br[:, None, None] + d[None, :, None]) + 0 * d[None, None, :]      # (src, di, dj)
    ocol = (bc[:, None, None] + d[None, None, :]) + 0 * d[None, :, None]
    ok = (orow >= 0) & (orow < ny) & (ocol >= 0) & (ocol < nx)
    dr = (orow - yr[:, None, None])[ok]; dc = (ocol - yc[:, None, None])[ok]
    if rotated: dr, dc = -dc, dr               # kernel is applied in the rotated frame
    v = map_coordinates(Kc, [dr * os + cf, dc * os + cf], order=3, mode='constant', prefilter=False)
    cols = np.broadcast_to(src[:, None, None], ok.shape)[ok]
    return sp.csr_matrix((v, ((orow * nx + ocol)[ok], cols)), shape=(ny * nx, sr.size))


def noise_var(var, J, psf, g, c=(0., 0.), dil=DIL, pad_mode=None, fill=None, method='imcom', rsearch=RSEARCH_VEL):
    """Per-pixel variance of op(n) + fixnoise for independent input noise of variance `var`.
    pad_mode: None (no noise beyond the array; velocity maps) or 'edge' (op() pads with noise of
    the edge level; images).  fill: optional GPFill applied to the noise before op() (the noise
    then extends, correlated, outside fill.inside).  Exact for the imcom method (the operation
    is the explicit matrix M); for galsim it uses the kernel at the shear-displaced position."""
    if method == 'imcom':
        Md, pad = _imcom(var.shape, J, psf.sigma, g, c, dil, rsearch)
        Mr = ic.rotated(_imcom(var.shape, J, psf.sigma, g, (0., 0.), dil, rsearch)[0], var.shape, pad)
    else:
        R = 10; pad = R if pad_mode else 0
        Md = transfer_matrix(var.shape, J, psf, g, c, dil, rotated=False, pad=pad, R=R)
        Mr = transfer_matrix(var.shape, J, psf, g, (0., 0.), dil, rotated=True, pad=pad, R=R)
    if fill is None:
        v = np.pad(var, pad, mode=pad_mode or 'constant') if pad else var
        out = (Md.power(2) + Mr.power(2)) @ v.ravel()
        return np.clip(out.reshape(var.shape), 0, None)
    # fill: noise at grid pixels = Wfull @ n_inside (identity inside, GP weights on fill.tgt),
    # laid out on the input grid padded by `pad` (zeros in the padding)
    inside = fill.inside; idx = np.full(var.shape, -1); idx[inside] = np.arange(inside.sum())
    (tr, tc), W = fill.weights(fill.tgt)
    npx = var.shape[1] + 2 * pad
    flat = lambda r, cc: (np.asarray(r) + pad) * npx + (np.asarray(cc) + pad)
    Wfull = np.zeros(((var.shape[0] + 2 * pad) * npx, inside.sum()))
    ir, icol = np.nonzero(inside)
    Wfull[flat(ir, icol), idx[inside]] = 1.
    Wfull[flat(tr, tc)] = W
    rows = np.flatnonzero(inside)                  # output only needed inside the mask
    vin = var[inside]
    out = np.zeros(var.size)
    for M in (Md, Mr):
        A = M[rows] @ Wfull
        out[rows] += (A**2) @ vin
    return out.reshape(var.shape)


def fixnoise(var, wcs, psf, g, rng, pad_sigma=None, dil=DIL, method='imcom', rsearch=RSEARCH_VEL):
    """Noise realisation with variance map `var`, rotated 90 deg, operated on, rotated back."""
    n = np.sqrt(np.rot90(var)) * rng.standard_normal(var.shape)
    return np.rot90(op(n, wcs, psf, g, pad_sigma=pad_sigma, seed=int(rng.integers(1 << 30)), dil=dil,
                       method=method, rsearch=rsearch), -1)


class GPFill:
    """Linear fill of a map outside the boolean mask `inside`: the conditional mean of a
    zero-mean Gaussian process given the values inside.  Kernel: squared exponential with
    length `ell` (pixels).  For a PSF-convolved map, ell = sqrt(2) * sigma_psf is the map's own
    autocorrelation length, so the fill is as smooth as the PSF allows and the later
    deconvolution sees no step at the mask edge.
    noise_var: per-pixel noise variance of the inside values (units of arr^2); jitter: extra
    diagonal relative to the signal variance `amp`.  It sets how closely the fill follows
    pixel-scale structure: with jitter ~1e-6 the squared-exponential GP treats white noise as
    signal and extrapolates it ~100x amplified; jitter = 1e-2 (the default) keeps the noise gain
    of the fill ~1 while matching the data at the edge to ~1% (see README).  Pixels further than
    reach*ell from any inside pixel stay 0 (the conditional mean has decayed to the prior).
    The fill is linear in the inside values (fixed weights), so it can be applied to noise
    fields and its weights used for exact variance propagation."""
    def __init__(self, inside, ell, amp, noise_var=None, jitter=1e-2, reach=5.):
        self.inside = inside; self.ell = ell
        yy, xx = np.indices(inside.shape)
        self.pin = np.c_[yy[inside], xx[inside]].astype(float)
        self.tgt = ~inside & (distance_transform_edt(~inside) <= reach * ell)
        self.pout = np.c_[yy[self.tgt], xx[self.tgt]].astype(float)
        self.amp = amp if amp > 0 else 1.
        K = self._k(self.pin, self.pin)
        K[np.diag_indices_from(K)] += (0. if noise_var is None else np.asarray(noise_var, float)) + jitter * self.amp
        self.cf = cho_factor(K, lower=True)
        self.cross = self._k(self.pout, self.pin)

    def _k(self, a, b):
        d2 = (a[:, None, 0] - b[None, :, 0])**2 + (a[:, None, 1] - b[None, :, 1])**2
        return self.amp * np.exp(-d2 / (2 * self.ell**2))

    def __call__(self, arr):
        out = np.where(self.inside, arr, 0.).astype(float)
        out[self.tgt] = self.cross @ cho_solve(self.cf, arr[self.inside].astype(float))
        return out

    def weights(self, sel):
        """Fill weights for the target pixels flagged in the full-size boolean `sel`:
        returns (pixel indices (rows, cols), W) with filled values = W @ arr[inside]."""
        s = sel[self.tgt]
        return np.nonzero(self.tgt & sel), cho_solve(self.cf, self.cross[s].T).T


SPIKE_K = 10.0          # flag Halpha flux > SPIKE_K x the median 3-4.5 spaxels away (see halpha_spikes)


def halpha_spikes(flux, m, K=SPIKE_K, r_in=3.0, r_out=4.5, nmin=6):
    """Boolean map of Halpha flux spikes: gas-mask spaxels brighter than K times the median flux
    of the in-mask spaxels on an annulus r_in..r_out spaxels away (at least nmin of them).
    Where fewer than nmin annulus spaxels are in the mask, the galaxy's median flux is used instead.
    The intensity map is PSF-convolved (sigma ~2-2.7 spaxels), so even a point source is at most
    ~4-5x brighter than that annulus; anything above K = 10 is not resolved structure (a bad
    spaxel, or an unresolved source the data cannot represent).  Such spikes otherwise dominate
    the flux-weighted velocity around them and the GP fill's amplitude (README)."""
    import warnings
    R = int(np.ceil(r_out)); yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(yy, xx)
    offs = np.argwhere((rr >= r_in) & (rr <= r_out)) - R
    Fp = np.pad(np.where(m, flux, np.nan).astype(float), R, constant_values=np.nan)
    ny, nx = flux.shape
    stack = np.stack([Fp[R + dy:R + dy + ny, R + dx:R + dx + nx] for dy, dx in offs])   # annulus values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        L = np.nanmedian(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    # where the annulus is too sparse to judge, compare with the galaxy's median flux instead
    ref = np.where(n >= nmin, L, np.median(flux[m]) if m.any() else np.inf)
    with np.errstate(invalid='ignore'):
        return m & (flux > K * ref)


def make_fill(arr, inside, ell, noise_var=None, **kw):
    """GPFill with the signal variance estimated from the inside values of `arr`, minus the
    noise variance (so the fill's regularisation does not change with the noise level)."""
    if inside.sum() < 2 or inside.all():
        return None
    y2 = arr[inside].astype(float)**2
    amp = np.mean(y2) - (0. if noise_var is None else np.mean(noise_var))
    return GPFill(inside, ell, float(max(amp, 0.1 * np.mean(y2))), noise_var=noise_var, **kw)


def process(d, g=(0., 0.), seed=0, method='imcom', dil_img=None, dil_vel=None, inpaint='gp', ell_scale=1.0,
            erode=1, compute_var=True, rsearch_img=RSEARCH_IMG, rsearch_vel=RSEARCH_VEL, dil=None, spikes=True):
    """Return (deep-copied data dict with shear g injected, info dict).  g=0: no-shear control.
    method: 'imcom' (default) or 'galsim' (original; rings -- only to reproduce the first set).
    dil_img, dil_vel: PSF dilation per map (defaults DIL_IMG, DIL_VEL for imcom; DIL for galsim);
    dil sets both (legacy).  rsearch_*: IMCOM search radius in input pixels.
    inpaint: 'gp' (fill M0 and M1 outside the gas mask with a GP, kernel length
    ell_scale * sqrt(2) * sigma_psf; default) or None (zeros outside the mask: biased near edges).
    spikes: drop Halpha flux spikes (halpha_spikes) from the gas mask before processing, i.e. treat
    them as missing data; info['gas']['spike'] is the map of dropped spaxels.  Default True;
    False reproduces the first blind set.
    erode: spaxels removed from the edge of the gas mask in the output.
    compute_var=False skips the (slow, ~1 s) variance propagation and leaves the var maps as
    they came in; for repeated noise realisations where only the data maps are needed."""
    d = copy.deepcopy(d); rng = np.random.default_rng(seed); info = {}
    dflt = (DIL_IMG, DIL_VEL) if method == 'imcom' else (DIL, DIL)
    dil_img = dil if dil is not None else (dil_img or dflt[0])
    dil_vel = dil if dil is not None else (dil_vel or dflt[1])
    ki = dict(method=method, rsearch=rsearch_img); kv = dict(method=method, rsearch=rsearch_vel)

    # ---------------- image ----------------
    I = d['image']; pm = I['par_meta']; im = I['data'].astype(float)
    assert im.shape[0] == im.shape[1]
    J = image_jacobian(d); wcs = galsim.JacobianWCS(J[0, 0], J[0, 1], J[1, 0], J[1, 1])
    psf = galsim.Gaussian(fwhm=pm['psfFWHM'])
    edge = np.r_[im[:4].ravel(), im[-4:].ravel(), im[:, :4].ravel(), im[:, -4:].ravel()]
    sky = np.median(edge); sky_sd = 1.4826 * np.median(np.abs(edge - sky))
    # true per-pixel noise: sky (measured) + source Poisson in the same units (sky_sd^2/bkg per count)
    var_true = sky_sd**2 * (1 + np.clip(im - sky, 0, None) / pm['mean_bkg'])
    out = op(im - sky, wcs, psf, g, pad_sigma=sky_sd, seed=int(rng.integers(1 << 30)), dil=dil_img, **ki)
    out += fixnoise(var_true, wcs, psf, g, rng, pad_sigma=sky_sd, dil=dil_img, **ki)
    I['data'] = (out + sky).astype(I['data'].dtype)
    var_out = noise_var(var_true, J, psf, g, dil=dil_img, pad_mode='edge', **ki) if compute_var else var_true
    I['var'] = (I['var'] * var_out / var_true).astype(I['var'].dtype)   # keep on-disk var convention
    pm['psfFWHM'] = float(pm['psfFWHM'] * dil_img)
    info['img'] = dict(sky=sky, sky_sd=sky_sd, noise_ratio_med=float(np.median(var_out / var_true)), J=J)

    # ---------------- Halpha velocity map ----------------
    G = d['vmap']['gas']; gm = G['par_meta']
    E, N = gm['RA_grid'].astype(float), gm['Dec_grid'].astype(float)
    Jv = vmap_jacobian(gm)
    vwcs = galsim.JacobianWCS(Jv[0, 0], Jv[0, 1], Jv[1, 0], Jv[1, 1])
    ny, nx = E.shape; cpix = ((nx - 1) / 2, (ny - 1) / 2)
    E0 = np.interp(cpix[0], np.arange(nx), E[0]); N0 = np.interp(cpix[1], np.arange(ny), N[:, 0])
    c = (-E0, -N0)                                       # galaxy centre (0,0) rel. to array centre
    lam = HA * (1 + d['galaxy']['redshift'])
    vfwhm = float(np.interp(lam, gm['psf_wavel'], gm['psf_fwhm']))
    vpsf = galsim.Gaussian(fwhm=vfwhm)
    m = G['mask'].astype(bool) & np.isfinite(G['data']) & np.isfinite(G['var'])
    flux = np.nan_to_num(gm['intensity_map'].astype(float))
    spike = halpha_spikes(flux, m) if spikes else np.zeros_like(m)
    m = m & ~spike
    M0 = np.where(m, flux, 0.); M1 = M0 * np.where(m, G['data'], 0.)
    vr = np.where(m, np.clip(G['var'] - V_FLOOR, 1e-2, None), 0.)          # random part of var
    nv = M0**2 * vr                                       # noise variance of M1 (inside the mask)
    f1 = None
    if inpaint == 'gp':
        ell = ell_scale * np.sqrt(2) * vfwhm * FWHM2SIG / np.sqrt(abs(np.linalg.det(Jv)))   # pixels
        f1 = make_fill(M1, m, ell, noise_var=nv[m]); f0 = make_fill(M0, m, ell)
        if f1 is not None: M1 = f1(M1); M0 = f0(M0)
    elif inpaint is not None:
        raise ValueError(f'unknown inpaint mode {inpaint!r}')
    if not np.any(M1): M1[m] = 1e-12                      # galsim rejects zero-flux images
    M0p = op(M0, vwcs, vpsf, g, c, dil=dil_vel, **kv); M1p = op(M1, vwcs, vpsf, g, c, dil=dil_vel, **kv)
    if f1 is None:
        M1p += fixnoise(nv, vwcs, vpsf, g, rng, dil=dil_vel, **kv)
        vM1 = noise_var(nv, Jv, vpsf, g, c, dil=dil_vel, **kv) if compute_var else None
    else:
        # the filled M1 carries (correlated) noise outside the mask too: give the fresh noise
        # field the same fill before rotating, and propagate variance through fill + kernel
        n = f1(np.sqrt(nv) * rng.standard_normal(nv.shape))
        M1p += np.rot90(op(np.rot90(n), vwcs, vpsf, g, dil=dil_vel, **kv), -1)
        vM1 = noise_var(nv, Jv, vpsf, g, c, dil=dil_vel, fill=f1, **kv) if compute_var else None
    # without inpainting, zeroed spaxels outside the mask bias v' up to ~8 spaxels inside the
    # edge (see README); erode only trims the outermost ring(s)
    mout = (binary_erosion(m, iterations=erode) if erode else m) & (M0p > 0)
    vout = G['data'].copy(); vvar = G['var'].copy()
    vout[mout] = (M1p / np.where(mout, M0p, 1))[mout]
    if compute_var:
        vvar[mout] = (vM1 / np.where(mout, M0p, 1)**2)[mout] + V_FLOOR
    with np.errstate(invalid='ignore', divide='ignore'):
        vratio = float(np.median(((vvar - V_FLOOR) / np.clip(G['var'] - V_FLOOR, 1e-2, None))[mout])) if mout.any() else np.nan
    G['data'] = vout.astype(G['data'].dtype); G['var'] = vvar.astype(G['var'].dtype)
    G['mask'] = mout.astype(G['mask'].dtype)
    gm['intensity_map'] = op(flux, vwcs, vpsf, g, c, dil=dil_vel, **kv).astype(gm['intensity_map'].dtype)  # noiseless
    gm['psf_fwhm'] = [float(f * dil_vel) for f in gm['psf_fwhm']]
    info['run'] = dict(method=method, dil_img=dil_img, dil_vel=dil_vel, rsearch_img=rsearch_img,
                       rsearch_vel=rsearch_vel, inpaint=inpaint)
    info['gas'] = dict(spike=spike, n_spike=int(spike.sum()), inpaint=inpaint, psf_fwhm_ha=vfwhm, n_in=int(m.sum()), n_out=int(mout.sum()), vnoise_ratio_med=vratio, J=Jv)
    return d, info


# ---------------------------------------------------------------- batch driver
def noise_seed(mangaid, salt=0):
    """Per-galaxy noise seed; identical across the no-shear and sheared sets of one run.
    salt=0 gives crc32(mangaid), which anyone can recompute.  For a blind set use a secret salt
    (and keep it with the private truth table): otherwise the deterministic pipeline lets anyone
    with the original data regenerate the sheared maps for trial shears and match them exactly."""
    base = zlib.crc32(mangaid.encode())
    if not salt:
        return base
    return int(np.random.SeedSequence([base, int(salt)]).generate_state(1, np.uint64)[0])


def _one(f, g, out_dir, pkw, salt=0):
    d = joblib.load(f); mid = d['galaxy']['mangaid']; seed = noise_seed(mid, salt)
    o, info = process(d, tuple(g), seed=seed, **pkw)
    joblib.dump(o, os.path.join(out_dir, os.path.basename(f)))
    gi = shear_to_pixel(g[0], g[1], info['img']['J']); gv = shear_to_pixel(g[0], g[1], info['gas']['J'])
    spike = info['gas']['spike']
    return dict(file=os.path.basename(f), mangaid=mid, plateifu=d['galaxy']['plateifu'], _spike=spike,
                gas_n_spike=int(spike.sum()),
                g1=g[0], g2=g[1], g1_img_pix=gi.g1, g2_img_pix=gi.g2, g1_vel_pix=gv.g1, g2_vel_pix=gv.g2,
                noise_seed=seed, img_sky_sd=info['img']['sky_sd'], img_noise_var_ratio=info['img']['noise_ratio_med'],
                ha_psf_fwhm_in=info['gas']['psf_fwhm_ha'], gas_n_in=info['gas']['n_in'],
                gas_n_out=info['gas']['n_out'], gas_noise_var_ratio=info['gas']['vnoise_ratio_med'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['noshear', 'fixed', 'random'])
    ap.add_argument('--src', required=True, help='directory of input data_info-*.pkl files')
    ap.add_argument('--out', required=True, help='output directory (same filenames as input)')
    ap.add_argument('--truth', help='CSV of injected shears and run info (required for random; '
                    'keep it OUT of the output directory for a blind test)')
    ap.add_argument('--g1', type=float, default=0.); ap.add_argument('--g2', type=float, default=0.)
    ap.add_argument('--seed', type=int, default=20261001, help='shear seed for random mode')
    ap.add_argument('--gmax', type=float, default=G_MAX,
                    help='max |g| per component (random draws; sets the dilation for --method galsim)')
    ap.add_argument('--method', choices=['imcom', 'galsim'], default='imcom',
                    help='imcom (default; PyIMCOM weights) or galsim (original; rings -- reproduces the first blind set)')
    ap.add_argument('--dil-img', type=float, help=f'image PSF dilation (default {DIL_IMG} for imcom)')
    ap.add_argument('--dil-vel', type=float, help=f'Halpha PSF dilation (default {DIL_VEL} for imcom)')
    ap.add_argument('--only', nargs='*', help='process only these filenames (random draws still use the full list)')
    ap.add_argument('--inpaint', choices=['none', 'gp'], default='gp',
                    help='fill the gas maps outside the mask before shearing (default gp; none = original)')
    ap.add_argument('--ell-scale', type=float, default=1.0, help='GP kernel length in units of sqrt(2)*sigma_psf')
    ap.add_argument('--erode', type=int, default=1, help='spaxels trimmed from the gas-mask edge in the output')
    ap.add_argument('--no-spikes', action='store_true',
                    help='keep Halpha flux spikes in the gas mask (only to reproduce the first blind set)')
    ap.add_argument('--noise-salt', type=int, default=0,
                    help='secret integer mixed into the per-galaxy noise seeds; use one for blind sets and keep it '
                         'private (with --truth).  Use the same salt for a blind set and its no-shear control')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.src, 'data_info-*.pkl')))
    if not files: ap.error(f'no data_info-*.pkl files in {a.src}')
    if a.mode == 'noshear':
        G = np.zeros((len(files), 2))
    elif a.mode == 'fixed':
        if max(abs(a.g1), abs(a.g2)) > a.gmax: ap.error('|g| exceeds --gmax; raise --gmax')
        G = np.tile([a.g1, a.g2], (len(files), 1))
    else:
        if not a.truth: ap.error('random mode needs --truth')
        G = np.random.default_rng(a.seed).uniform(-a.gmax, a.gmax, (len(files), 2))
    if a.truth and os.path.abspath(os.path.dirname(a.truth) or '.') == os.path.abspath(a.out):
        print('WARNING: truth table is inside the output directory; do not ship it with a blind set')
    sel = [(f, g) for f, g in zip(files, G) if not a.only or os.path.basename(f) in a.only]
    os.makedirs(a.out, exist_ok=True)
    if a.method == 'galsim':
        dil_img = a.dil_img or dilation(a.gmax); dil_vel = a.dil_vel or dilation(a.gmax)
    else:
        dil_img = a.dil_img or DIL_IMG; dil_vel = a.dil_vel or DIL_VEL
    for dl in (dil_img, dil_vel):
        ic.check_stable(dl, (a.gmax, a.gmax) if a.mode == 'random' else (a.g1, a.g2))
    pkw = dict(method=a.method, dil_img=dil_img, dil_vel=dil_vel, inpaint=None if a.inpaint == 'none' else a.inpaint,
               ell_scale=a.ell_scale, erode=a.erode, spikes=not a.no_spikes)
    rows = joblib.Parallel(n_jobs=a.jobs)(joblib.delayed(_one)(f, g, a.out, pkw, a.noise_salt) for f, g in sel)
    # Halpha spike flags: depend only on the input data (not on the shear), so they can be shipped
    # with a blind set.  The mask is True on spaxels dropped from the gas mask as flux spikes.
    import pandas as pd
    spikes = {r['file']: r.pop('_spike') for r in rows}
    if not a.no_spikes:
        np.savez_compressed(os.path.join(a.out, 'halpha_spike_mask.npz'),
                            **{k.replace('.pkl', ''): v for k, v in spikes.items() if v.any()})
        pd.DataFrame([dict(file=r['file'], mangaid=r['mangaid'], plateifu=r['plateifu'], n_spike_spaxels=r['gas_n_spike'])
                      for r in rows if r['gas_n_spike']]).to_csv(os.path.join(a.out, 'halpha_spike_flags.csv'), index=False)
    if a.truth:
        pd.DataFrame(rows).to_csv(a.truth, index=False, float_format='%.6f')
        json.dump(dict(mode=a.mode, shear_dist=f'uniform per component in [-{a.gmax}, {a.gmax}]' if a.mode == 'random' else None,
                       shear_seed=a.seed if a.mode == 'random' else None, method=a.method,
                       psf_dilation_image=dil_img, psf_dilation_halpha=dil_vel,
                       imcom_rsearch=dict(image=RSEARCH_IMG, halpha=RSEARCH_VEL) if a.method == 'imcom' else None,
                       imcom_code='InterpMatrix from PyIMCOM pyimcom.meta.ginterp (C. M. Hirata; MIT), commit '
                                  '9ca33dd24aa87159ac9689faef49a1b7b49fcb22, unmodified copy in _vendor/' if a.method == 'imcom' else None,
                       noise_seed='crc32(mangaid)' if not a.noise_salt else 'SeedSequence([crc32(mangaid), noise_salt])',
                       noise_salt=a.noise_salt, halpha_spikes='kept' if a.no_spikes else f'dropped (K={SPIKE_K})',
                       n_files=len(rows), inpaint=a.inpaint, ell_scale=a.ell_scale,
                       erode=a.erode,
                       convention='u=+East (dRA cos dec), v=+North; g1>0 stretches E-W; g2>0 stretches along PA=45 deg '
                                  '(NE-SW); shear about galaxy centre. *_img_pix / *_vel_pix: same shear in each map\'s '
                                  'array axes (x=column, y=row).'),
                  open(os.path.splitext(a.truth)[0] + '_provenance.json', 'w'), indent=1)
    print(len(rows), 'written to', a.out)


if __name__ == '__main__':
    main()
