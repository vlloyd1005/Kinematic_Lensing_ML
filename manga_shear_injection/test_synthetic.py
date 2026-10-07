"""Validate manga_shear.py on synthetic maps against closed-form forward models.

The synthetic galaxy is a sum of elliptical Gaussians (flux F) and of Hermite-Gaussian terms
(flux-weighted velocity F*v), so every step the pipeline approximates numerically has an
exact answer:
    shear       G(y; C)          -> G(y; S C S^T)                 (det S = 1)
                (w.y) G(y; C)    -> (S^-T w . y) G(y; S C S^T)
    PSF         G(y; C) * G(.; P)       = G(y; C + P)
                (w.y) G(y; C) * G(.; P) = (w . C (C+P)^-1 y) G(y; C + P)
with S = [[1+g1, g2], [g2, 1-g1]] / sqrt(1-|g|^2), i.e. g1 > 0 stretches along +East.
None of the expected values use galsim.

All tests use the production defaults (method='imcom', dilation DIL_IMG / DIL_VEL, GP
inpainting) unless marked legacy.
Tests
  1  op() on the image, noiseless, several pixel orientations and shears
  1c op() on the velocity path (M1'/M0', off-centre galaxy) with untruncated maps
  2  process() end to end on a synthetic data dict with a celestial WCS (image + velocity)
  3  sign convention: g1>0 stretches E-W and g2>0 along NE-SW on the sky; pixel-frame
     truth (shear_to_pixel) matches the elongation measured in array axes
  4  noise: Monte Carlo per-pixel variance vs the propagated var map, isotropy of the
     symmetrised noise, unbiasedness of the noisy outputs, sheared-minus-control noise
  5  gas-mask edges: response accuracy near the mask edge on smooth and patchy masks, and the
     noise checks of 4 repeated on the patchy mask
  6  ringing / shear "tell": impulse response of the operation vs the ideal Gaussian kernel,
     its shear-dependent part, and finite support
  7  integrity of the copied PyIMCOM file (SHA-256 of the unmodified part)
  8  Halpha flux spikes: no false positives on a bright PSF-convolved nucleus, exact detection of
     injected spikes, and outputs away from a spike unaffected once it is dropped
  L  legacy (method='galsim', zero-fill, spikes kept; the first blind set): edge bias, ringing
     and spike damage, reported as known limitations
Run:  python test_synthetic.py [--quick] [--plots DIR]
"""
import argparse, os, sys, time
import numpy as np, galsim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manga_shear as ms

FWHM2SIG = 1 / (2 * np.sqrt(2 * np.log(2)))
RESULTS = []


def check(name, value, limit, fmt='{:.2e}', note='', known=False):
    """known=True: a failure is an expected, documented limitation (reported as KNOWN)."""
    ok = bool(np.all(np.abs(value) <= limit))
    RESULTS.append((name, fmt.format(np.max(np.abs(value))), fmt.format(limit), 'PASS' if ok else ('KNOWN' if known else 'FAIL'), note))
    return ok


# ---------------------------------------------------------------- analytic model
def smat(g1, g2):
    return np.array([[1 + g1, g2], [g2, 1 - g1]]) / np.sqrt(1 - g1**2 - g2**2)


def rot(deg):
    t = np.radians(deg); return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def gauss(y, C):
    Ci = np.linalg.inv(C)
    q = np.einsum('...i,ij,...j->...', y, Ci, y)
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(np.linalg.det(C)))


class Galaxy:
    """F(y) = sum a_k G(y; C_k);  F v(y) = v_sys F(y) + sum b_k (w_k . y) G(y; D_k),  y = x - centre."""
    def __init__(self, flux_terms, vel_terms=(), v_sys=0.):
        self.F, self.V, self.v_sys = flux_terms, vel_terms, v_sys

    def observe(self, y, psf_sigma, g=(0., 0.)):
        """(F, F*v) after shear g then convolution with a round Gaussian PSF, at world offsets y."""
        S = smat(*g); Si_T = np.linalg.inv(S).T; P = psf_sigma**2 * np.eye(2)
        F = sum(a * gauss(y, S @ C @ S.T + P) for a, C in self.F)
        Fv = self.v_sys * F
        for b, w, D in self.V:
            Dp = S @ D @ S.T; wp = Si_T @ w
            Fv = Fv + b * np.einsum('i,ij,...j->...', wp, Dp @ np.linalg.inv(Dp + P), y) * gauss(y, Dp + P)
        return F, Fv


def disk_cov(sig_major, q, pa_deg):
    """Covariance of an elliptical Gaussian, major axis at PA (deg E of N) on the (E, N) sky."""
    phi = np.radians(90 - pa_deg)                  # angle from +E toward +N
    R = rot(np.degrees(phi)); return R @ np.diag([sig_major**2, (q * sig_major)**2]) @ R.T


def make_galaxy(pa=30., q=0.5, vmax=200., v_sys=15.):
    Cd = disk_cov(3.0, q, pa); Cb = disk_cov(0.8, 0.9, pa + 40)
    w = rot(90 - pa) @ np.array([1., 0.])          # unit vector along the major axis
    # F v ~ (w.y) G(narrower) gives a rising-then-turning-over rotation curve
    gal = Galaxy([(1.0, Cd), (0.3, Cb)], [(1.0, w, 0.7 * Cd)], v_sys=v_sys)
    y = np.stack(np.meshgrid(np.linspace(-15, 15, 301), np.linspace(-15, 15, 301)), -1)
    F, Fv = gal.observe(y, 0.)
    vpk = np.max(np.abs(Fv - v_sys * F) / np.maximum(F, 1e-300) * (F > 1e-3 * F.max()))
    gal.V = [(vmax / vpk, w, 0.7 * Cd)]
    return gal


def pixel_offsets(shape, J, centre_off=(0., 0.)):
    """World (E, N) offset of each pixel centre from the galaxy, for array-centre offset centre_off."""
    ny, nx = shape; c, r = np.meshgrid(np.arange(nx) - (nx - 1) / 2, np.arange(ny) - (ny - 1) / 2)
    return np.einsum('ij,...j->...i', J, np.stack([c, r], -1)) - np.asarray(centre_off)


def jwcs(J):
    return galsim.JacobianWCS(J[0, 0], J[0, 1], J[1, 0], J[1, 1])


# ---------------------------------------------------------------- 1. op() on images
def test_op_image(quick=False):
    gal = make_galaxy(); fwhm = 1.32; s = fwhm * FWHM2SIG; dil = ms.DIL_IMG
    Js = {'SDSS-like (transposed)': np.array([[0., 0.396], [0.396, 0.]]),
          'rotated 37 deg + flip': 0.396 * rot(37.) @ np.diag([1., -1.]),
          'E-W/N-S aligned': np.array([[0.396, 0.], [0., 0.396]])}
    shears = [(0., 0.), (0.03, 0.), (0., 0.03), (-0.021, 0.027)]
    if quick: shears = shears[1:3]
    worst = []
    for name, J in Js.items():
        y = pixel_offsets((65, 65), J); F0, _ = gal.observe(y, s); peak = F0.max()
        for g in shears:
            for pad in (None, 1e-9):
                out = ms.op(F0, jwcs(J), galsim.Gaussian(fwhm=fwhm), g, pad_sigma=pad, seed=1, dil=dil,
                            rsearch=ms.RSEARCH_IMG)
                Fexp, _ = gal.observe(y, s * dil, g)
                Fctl, _ = gal.observe(y, s * dil)
                resid = (out - Fexp) / peak
                signal = np.abs(Fexp - Fctl).max() / peak if any(g) else np.nan
                worst.append((name, g, pad, np.abs(resid).max(), np.sqrt(np.mean(resid**2)), signal))
    mx = max(w[3] for w in worst); rel = max(w[3] / w[5] for w in worst if np.isfinite(w[5]))
    check('1a op(image) max |resid| / peak', mx, 1e-5, note=f'{len(worst)} cases: 3 pixel orientations x shears x padding')
    check('1b op(image) max |resid| / max |shear signal|', rel, 1e-3, note='error relative to the change the shear makes')
    return worst


# ---------------------------------------------------------------- synthetic data dict
def make_dict(gal, img_J_affine, dec0=30., sky=100., img_noise=None, vel_noise=None, rng=None,
              v_off=(0.7, -0.45), vpsf=2.5, ipsf=1.32, mean_bkg=700., img_scale=2e4, holes=0.):
    """Synthetic MaNGA-like data dict.  img_J_affine: 2x2 (u,v) arcsec per frame pixel for a
    galsim TanWCS; the expected image is evaluated from each pixel's true RA/Dec."""
    ra0, dec0 = 150., dec0
    c0 = galsim.CelestialCoord(ra0 * galsim.degrees, dec0 * galsim.degrees)
    aff = galsim.AffineTransform(*img_J_affine.ravel(), origin=galsim.PositionD(1000., 700.))
    wcs = galsim.TanWCS(aff, world_origin=c0, units=galsim.arcsec)
    n = 65; pc = (n - 1) / 2; p0 = wcs.toImage(c0)
    E = np.zeros((n, n)); N = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            c = wcs.toWorld(galsim.PositionD(p0.x + i - pc, p0.y + j - pc))
            E[j, i] = (c.ra - c0.ra).wrap().rad * np.cos(c0.dec.rad) * 206264.806
            N[j, i] = (c.dec - c0.dec).rad * 206264.806
    yi = np.stack([E, N], -1)
    F, _ = gal.observe(yi, ipsf * FWHM2SIG); img = img_scale * F / F.max()
    var_true = (np.full_like(img, 1e-12) if img_noise is None else img_noise**2 * (1 + img / mean_bkg))
    data = sky + img + (rng.standard_normal(img.shape) * np.sqrt(var_true) if img_noise is not None else 0)
    image = dict(data=data.astype(np.float32), var=var_true.astype(np.float32),
                 par_meta=dict(psfFWHM=ipsf, wcs=wcs, mean_bkg=mean_bkg))

    nv = 72; Jv = np.array([[0.5, 0.], [0., -0.5]])
    yv = pixel_offsets((nv, nv), Jv, centre_off=(-v_off[0], -v_off[1]))   # galaxy at E,N = 0
    F0, Fv = gal.observe(yv, vpsf * FWHM2SIG); vel = Fv / F0
    flux = 100 * F0 / F0.max()
    mask = (np.hypot(*np.moveaxis(yv, -1, 0)) < 13.) & (flux > 3.)
    if holes:                                      # patchy mask: knock out smooth random blobs
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(np.random.default_rng(1).standard_normal(mask.shape), 1.5)
        mask &= f < np.quantile(f[mask], 1 - holes)
    vvar = np.full(vel.shape, ms.V_FLOOR + (0 if vel_noise is None else vel_noise**2))
    vdata = vel + (rng.standard_normal(vel.shape) * vel_noise if vel_noise is not None else 0)
    gas = dict(data=np.where(mask, vdata, 0.).astype(np.float32), var=np.where(mask, vvar, 0.).astype(np.float32),
               mask=mask.astype(int),
               par_meta=dict(RA_grid=yv[..., 0].astype(np.float32), Dec_grid=yv[..., 1].astype(np.float32),
                             psf_wavel=[4770, 6231, 7625, 9134], psf_fwhm=[vpsf] * 4,
                             intensity_map=flux))
    d = dict(galaxy=dict(mangaid='synthetic', plateifu='0-0', RA=ra0, Dec=dec0, redshift=0.03),
             image=image, vmap=dict(gas=gas))
    return d, dict(yi=yi, yv=yv, img_norm=img_scale / F.max(), flux_norm=100 / F0.max(), mask=mask)


def expected(gal, aux, g, ipsf=1.32, vpsf=2.5, sky=100., dil_img=ms.DIL_IMG, dil_vel=ms.DIL_VEL):
    Fi, _ = gal.observe(aux['yi'], ipsf * FWHM2SIG * dil_img, g)
    Fv0, Fv1 = gal.observe(aux['yv'], vpsf * FWHM2SIG * dil_vel, g)
    return dict(img=sky + aux['img_norm'] * Fi, vel=Fv1 / Fv0, flux=aux['flux_norm'] * Fv0)


def interior(mask, k):
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=k)


# ---------------------------------------------------------------- 2. process() end to end
AFFINES = {'SDSS-like': np.array([[0., -0.396], [0.396, 0.]]),
           'rotated 37 deg': 0.396 * rot(37.) @ np.diag([-1., 1.])}


def test_process_noiseless(pkw={}, legacy=False):
    """process() end to end.  pkw={} is production; legacy=True runs the first blind set's
    settings (galsim, zero-fill, DIL) and reports its edge bias as known limitations."""
    from scipy.ndimage import distance_transform_edt
    gal = make_galaxy(pa=30.); rows = []
    di, dv = (ms.DIL, ms.DIL) if legacy else (ms.DIL_IMG, ms.DIL_VEL)
    for aname, A in AFFINES.items():
        d, aux = make_dict(gal, A, img_noise=1e-3, rng=np.random.default_rng(0))   # ~zero noise
        oc, _ = ms.process(d, (0., 0.), seed=3, compute_var=False, **pkw)
        ctl = expected(gal, aux, (0., 0.), dil_img=di, dil_vel=dv)
        dist = np.round(distance_transform_edt(aux['mask'])).astype(int)          # spaxels from input-mask edge
        for g in [(0.03, 0.), (0., -0.03), (-0.021, 0.027)]:
            o, info = ms.process(d, g, seed=3, compute_var=False, **pkw); ex = expected(gal, aux, g, dil_img=di, dil_vel=dv)
            m_out = o['vmap']['gas']['mask'].astype(bool)
            ri = (o['image']['data'] - ex['img']) / (ex['img'].max() - 100)
            n_ = ri.shape[0]; yy_, xx_ = np.mgrid[:n_, :n_]
            bdist = np.minimum.reduce([yy_, xx_, n_ - 1 - yy_, n_ - 1 - xx_])     # pixels from the cutout edge
            rf = (o['vmap']['gas']['par_meta']['intensity_map'] - ex['flux']) / ex['flux'].max()
            # velocity response (sheared - control, same noise seed) vs the analytic response
            resp = o['vmap']['gas']['data'] - oc['vmap']['gas']['data']; resp_ex = ex['vel'] - ctl['vel']
            err = np.abs(resp - resp_ex); sig = np.abs(resp_ex)[m_out].max()
            rows.append(dict(case=f'{aname} g={g}', img=np.abs(ri[bdist >= 6]).max(), img_border=np.abs(ri[bdist < 6]).max(),
                             flux=np.abs(rf).max(), sig=sig,
                             err_by_d={k: err[m_out & (dist == k)].max() / sig for k in range(2, 9) if (m_out & (dist == k)).any()},
                             err_deep=err[m_out & (dist >= 8)].max(), err_in=err[m_out & (dist >= 3)].max() / sig,
                             err_edge=err[m_out & (dist <= 2)].max() / sig,
                             psf=o['image']['par_meta']['psfFWHM'], psf_v=o['vmap']['gas']['par_meta']['psf_fwhm'][0],
                             dv_map=np.where(m_out, resp - resp_ex, np.nan), mask_in=aux['mask']))
    if legacy:
        for k in range(2, 9):
            v = max(r['err_by_d'].get(k, 0) for r in rows)
            RESULTS.append((f'L1 legacy (galsim, zero-fill): response error / signal, {k} spaxels from mask edge', f'{v:.3f}',
                            'report', 'KNOWN' if v > 0.01 else 'INFO',
                            'first blind set: zero-fill + galsim ringing at the mask edge' if k == 2 else ''))
        return rows
    check('2a process: image max |resid| / peak, >= 6 px from the cutout edge', max(r['img'] for r in rows), 1e-5,
          note='celestial TanWCS at Dec=30, sky=100 subtracted & restored')
    check('2a process: image max |resid| / peak, outer 6 px of the cutout', max(r['img_border'] for r in rows), 1e-4,
          note='galaxy light beyond the cutout is unknown (padding is zero-mean noise)')
    check('2b process: intensity_map max |resid| / peak', max(r['flux'] for r in rows), 1e-5)
    check('2c process: velocity response error / signal, >= 3 spaxels from mask edge', max(r['err_in'] for r in rows), 0.01, '{:.4f}',
          note=f"shear signal up to {max(r['sig'] for r in rows):.1f} km/s; includes the small noise process() adds")
    check('2d process: velocity response error / signal, output spaxels <= 2 from mask edge', max(r['err_edge'] for r in rows), 0.05, '{:.4f}',
          note='limited by how well the GP fill matches the gas just outside the mask (see 5a/5b)')
    check('2e process: PSF FWHM updated by DIL_IMG / DIL_VEL',
          max(abs(rows[0]['psf'] - 1.32 * ms.DIL_IMG), abs(rows[0]['psf_v'] - 2.5 * ms.DIL_VEL)), 1e-12)
    return rows


def test_op_velocity():
    """Velocity path (M1'/M0', shear about an off-centre galaxy) on UNtruncated maps."""
    gal = make_galaxy(pa=-50.); fw = 2.5; s = fw * FWHM2SIG; Jv = np.array([[0.5, 0.], [0., -0.5]]); dil = ms.DIL_VEL
    c = (0.7, -0.45); yv = pixel_offsets((72, 72), Jv, centre_off=c)
    F0, Fv = gal.observe(yv, s); ctl = gal.observe(yv, s * dil); inner = F0 > 0.03 * F0.max()
    worst_abs = worst_resp = 0.; sig = 0.
    for g in [(0.03, 0.), (0., 0.03), (-0.021, -0.027)]:
        a, b = gal.observe(yv, s * dil, g); psf = galsim.Gaussian(fwhm=fw)
        v = ms.op(Fv, jwcs(Jv), psf, g, c, dil=dil) / ms.op(F0, jwcs(Jv), psf, g, c, dil=dil)
        v0 = ms.op(Fv, jwcs(Jv), psf, dil=dil) / ms.op(F0, jwcs(Jv), psf, dil=dil)
        worst_abs = max(worst_abs, np.abs(v - b / a)[inner].max())
        worst_resp = max(worst_resp, np.abs((v - v0) - (b / a - ctl[1] / ctl[0]))[inner].max())
        sig = max(sig, np.abs(b / a - ctl[1] / ctl[0])[inner].max())
    check('1c op(velocity) max |v err| (km/s), untruncated maps', worst_abs, 0.01, '{:.1e}', note='off-centre galaxy (0.7, -0.45)"')
    check('1d op(velocity) max |response err| / max |shear signal|, untruncated', worst_resp / sig, 1e-4, '{:.1e}',
          note=f'signal up to {sig:.1f} km/s')


# ---------------------------------------------------------------- 5. GP inpainting
def test_inpaint():
    """Velocity response error vs distance from the mask edge with GP inpainting (ell = sqrt2 sigma_psf),
    on the ellipse mask and on a patchy mask with holes."""
    from scipy.ndimage import distance_transform_edt
    # vmax x10 so the small noise process() always adds (var floor 1e-2 km^2/s^2) is negligible
    # and the systematic error of the fill is what is measured
    gal = make_galaxy(pa=30., vmax=2000., v_sys=150.); out = {}
    for mname, holes in [('ellipse', 0.), ('patchy', 0.12)]:
        d, aux = make_dict(gal, AFFINES['SDSS-like'], img_noise=1e-3, rng=np.random.default_rng(0), holes=holes)
        dist = np.round(distance_transform_edt(aux['mask'])).astype(int); ctl = expected(gal, aux, (0., 0.))
        oc, _ = ms.process(d, (0., 0.), seed=3, erode=0, compute_var=False)
        e1 = e2 = 0.
        for g in [(0.03, 0.), (0., -0.03), (-0.021, 0.027)]:
            o, _ = ms.process(d, g, seed=3, erode=0, compute_var=False)
            ex = expected(gal, aux, g)['vel'] - ctl['vel']; sig = np.abs(ex)[aux['mask']].max()
            err = np.abs(o['vmap']['gas']['data'] - oc['vmap']['gas']['data'] - ex) / sig
            mo = o['vmap']['gas']['mask'].astype(bool)
            e1 = max(e1, err[mo & (dist == 1)].max()); e2 = max(e2, err[mo & (dist >= 2)].max())
            out[(mname, g)] = np.where(mo, err, np.nan)
        check(f'5a {mname} mask: response err / signal, >=2 spaxels from edge', e2, 0.02, '{:.4f}',
              note='default erode=1 keeps these' + (' (12% of mask knocked out as holes)' if holes else ''))
        check(f'5b {mname} mask: response err / signal, edge ring (erode=0)', e1, 0.10, '{:.4f}')
    return out


# ---------------------------------------------------------------- 3. conventions
def moments(img, y, w_sig=4.):
    w = img * np.exp(-np.sum(y**2, -1) / (2 * w_sig**2))
    Q = np.einsum('ab,abi,abj->ij', w, y, y) / w.sum()
    e = complex(Q[0, 0] - Q[1, 1], 2 * Q[0, 1]) / (Q[0, 0] + Q[1, 1])
    return e


def test_conventions():
    rnd = Galaxy([(1.0, np.eye(2) * 2.5**2)])
    out = {}
    for aname, A in AFFINES.items():
        for g in [(0.03, 0.), (0., 0.03), (-0.02, 0.02)]:
            d, aux = make_dict(rnd, A, img_noise=1e-3, rng=np.random.default_rng(0))
            o, info = ms.process(d, g, seed=0, compute_var=False)
            im = o['image']['data'] - 100.
            e_sky = moments(im, aux['yi'])                     # in true (E, N)
            n = im.shape[0]; c, r = np.meshgrid(np.arange(n) - (n - 1) / 2, np.arange(n) - (n - 1) / 2)
            e_pix = moments(im, np.stack([c, r], -1) * 0.396)  # in array axes (x=col, y=row)
            gp = ms.shear_to_pixel(g[0], g[1], info['img']['J'])
            out[(aname, g)] = (e_sky, e_pix, complex(*g), complex(gp.g1, gp.g2))
            # velocity grid: moments of the sheared intensity map in sky and in array axes
            fl = o['vmap']['gas']['par_meta']['intensity_map']
            ev = moments(fl, aux['yv'], 5.)
            yv_pix = np.einsum('ij,...j->...i', np.linalg.inv(info['gas']['J']), aux['yv']) * 0.5
            ev_pix = moments(fl, yv_pix, 5.)
            gv = ms.shear_to_pixel(g[0], g[1], info['gas']['J'])
            out[(aname, g, 'vel')] = (ev, complex(gv.g1, gv.g2), ev_pix)
    dang_sky = [np.degrees(np.angle(v[0] / v[2])) / 2 for k, v in out.items() if len(k) == 2]
    dang_pix = [np.degrees(np.angle(v[1] / v[3])) / 2 for k, v in out.items() if len(k) == 2]
    dang_vel = [np.degrees(np.angle(v[0] / complex(*k[1]))) / 2 for k, v in out.items() if len(k) == 3]
    dang_velp = [np.degrees(np.angle(v[2] / v[1])) / 2 for k, v in out.items() if len(k) == 3]
    check('3a sky frame: measured elongation angle - injected (deg)', np.array(dang_sky), 0.5, '{:.3f}',
          note='round source; g1>0 = E-W, g2>0 = NE-SW in true RA/Dec')
    check('3b image pixel frame: measured - shear_to_pixel (deg)', np.array(dang_pix), 0.5, '{:.3f}',
          note='moments in raw array axes (x=column, y=row)')
    check('3c velocity grid, sky frame: measured - injected (deg)', np.array(dang_vel), 0.5, '{:.3f}')
    check('3d velocity grid, pixel frame: measured - shear_to_pixel (deg)', np.array(dang_velp), 0.5, '{:.3f}')
    return out


# ---------------------------------------------------------------- 4. noise
def test_noise(n_mc=300, plots=None, pkw={}, tag='4', holes=0., label=''):
    from scipy.ndimage import distance_transform_edt
    gal = make_galaxy(); rng = np.random.default_rng(11)
    g = (0.025, -0.02); A = AFFINES['SDSS-like']; sky_sd = 30.; v_sd = 12.
    d0, aux = make_dict(gal, A, img_noise=1e-3, rng=np.random.default_rng(0), holes=holes)
    dist = np.round(distance_transform_edt(aux['mask'])).astype(int)
    ex = expected(gal, aux, g); exc = expected(gal, aux, (0., 0.))
    stack = dict(img=[], vel=[], dimg=[], dvel=[], pimg=None, pvel=None)
    for k in range(n_mc):
        d, _ = make_dict(gal, A, img_noise=sky_sd, vel_noise=v_sd, rng=rng, holes=holes)
        o, info = ms.process(d, g, seed=k, compute_var=(k == 0), **pkw)
        oc, _ = ms.process(d, (0., 0.), seed=k, compute_var=False, **pkw)   # control: same input, same noise seed
        stack['img'].append(o['image']['data']); stack['vel'].append(o['vmap']['gas']['data'])
        stack['dimg'].append(o['image']['data'] - oc['image']['data'])
        stack['dvel'].append(o['vmap']['gas']['data'] - oc['vmap']['gas']['data'])
        if k == 0:
            stack['pimg'] = o['image']['var']; stack['pvel'] = o['vmap']['gas']['var'] - ms.V_FLOOR
            m_out = o['vmap']['gas']['mask'].astype(bool)
    S = {k: np.array(v, dtype=float) for k, v in stack.items() if isinstance(v, list)}
    vi_emp = S['img'].var(0); vv_emp = S['vel'].var(0)
    # predicted image var: on-disk var is scaled by var_out/var_true, and we supplied var = var_true
    r_img = vi_emp / stack['pimg']; r_vel = vv_emp[m_out] / stack['pvel'][m_out]
    se = np.sqrt(2 / (n_mc - 1))
    # all spaxels share the same realisations, so a median over spaxels still has sampling error
    # ~1/sqrt(n_mc): widen the velocity limits for small runs (--quick); unchanged for n_mc >= 200
    widen = max(1., np.sqrt(200 / n_mc))
    check(f'{tag}a{label} image: median(empirical var / predicted var) - 1', np.median(r_img) - 1, 4 * se / np.sqrt(r_img.size) + 0.03, '{:.4f}',
          note=f'{n_mc} realisations; per-pixel scatter of ratio {np.std(r_img):.3f} (expect {se:.3f})')
    check(f'{tag}b{label} velocity: median(empirical var / predicted var) - 1', np.median(r_vel) - 1, 0.05 * widen, '{:.4f}',
          note=f'per-spaxel scatter {np.std(r_vel):.3f} (expect {se:.3f})')
    near = (dist[m_out] <= 2)
    check(f'{tag}b{label} velocity: median ratio - 1, <=2 spaxels from mask edge', np.median(r_vel[near]) - 1, 0.1 * widen, '{:.4f}',
          note=f'{near.sum()} spaxels')
    edge = ~interior(np.ones_like(r_img, bool), 3)
    check(f'{tag}c{label} image: |median ratio - 1| in outer 3-pixel border', np.median(r_img[edge]) - 1, 0.1, '{:.4f}',
          note='edge padding: noise pad vs mode="edge" in noise_var')
    # unbiasedness: mean of noisy outputs vs analytic expectation
    zi = (S['img'].mean(0) - ex['img']) / np.sqrt(vi_emp / n_mc)
    zv = ((S['vel'].mean(0) - ex['vel']) / np.sqrt(vv_emp / n_mc))[interior(m_out, 1)]
    one_sided = 'one-sided: bias raises rms z; correlated noise often puts it below 1'
    check(f'{tag}d{label} image: mean over realisations vs expected, rms z', max(np.sqrt(np.mean(zi**2)), 0), 1.15, '{:.3f}',
          note='z = (mean - analytic) / sem; ' + one_sided)
    check(f'{tag}e{label} velocity: mean over realisations vs expected, rms z', np.sqrt(np.mean(zv**2)), 1.15, '{:.3f}')
    dv = S['dvel'][:, interior(m_out, 1)]; rex = (ex['vel'] - exc['vel'])[interior(m_out, 1)]
    zr = (dv.mean(0) - rex) / np.sqrt(dv.var(0) / n_mc)
    check(f'{tag}e{label} velocity response (sheared - control): mean vs expected, rms z', np.sqrt(np.mean(zr**2)), 1.25, '{:.3f}',
          note='the shear signal itself; z uses the paired-difference scatter')
    # noise isotropy of the symmetrised output noise (residual from the analytic mean)
    res = S['img'] - ex['img']
    def xi(dx, dy):
        a = res[:, 8:-8, 8:-8]; b = np.roll(np.roll(res, -dy, 1), -dx, 2)[:, 8:-8, 8:-8]
        return np.mean(a * b, axis=(1, 2)) / np.mean(a * a, axis=(1, 2))
    an1 = xi(1, 0) - xi(0, 1); an2 = xi(1, 1) - xi(1, -1)
    # same statistic without the fixnoise term, for reference
    J = ms.image_jacobian(d0); w = jwcs(J); psf = galsim.Gaussian(fwhm=1.32)
    raw = np.array([ms.op(rng.standard_normal((65, 65)) * sky_sd, w, psf, g, pad_sigma=sky_sd, seed=k,
                          dil=ms.DIL_IMG, rsearch=ms.RSEARCH_IMG) for k in range(min(n_mc, 150))])
    def xi_raw(dx, dy):
        a = raw[:, 8:-8, 8:-8]; b = np.roll(np.roll(raw, -dy, 1), -dx, 2)[:, 8:-8, 8:-8]
        return np.mean(a * b, axis=(1, 2)) / np.mean(a * a, axis=(1, 2))
    ra1 = xi_raw(1, 0) - xi_raw(0, 1); ra2 = xi_raw(1, 1) - xi_raw(1, -1)
    z1 = an1.mean() / (an1.std() / np.sqrt(len(an1))); z2 = an2.mean() / (an2.std() / np.sqrt(len(an2)))
    zr = max(abs(ra1.mean() / (ra1.std() / np.sqrt(len(ra1)))), abs(ra2.mean() / (ra2.std() / np.sqrt(len(ra2)))))
    check(f'{tag}f{label} symmetrised noise anisotropy, |z| of xi(1,0)-xi(0,1) and xi(1,1)-xi(1,-1)', np.array([z1, z2]), 4., '{:.2f}',
          note=f'without fixnoise the same statistic gives |z| = {zr:.0f}')
    # sheared minus control (same input, same noise seed): how much noise survives
    di = S['dimg'] - (ex['img'] - exc['img']); dvv = (S['dvel'] - (ex['vel'] - exc['vel']))[:, interior(m_out, 1)]
    fi = np.sqrt(np.mean(di**2) / np.mean(vi_emp)); fv = np.sqrt(np.mean(dvv**2) / np.mean(vv_emp[interior(m_out, 1)]))
    RESULTS.append((f'{tag}g{label} sheared - control: residual noise rms / output noise rms', f'img {fi:.3f}, vel {fv:.3f}', 'report', 'INFO',
                    'metacal-style paired difference; smaller = less noise in the response'))
    if plots:
        return dict(r_img=r_img, r_vel=np.where(m_out, vv_emp / np.where(m_out, stack['pvel'], 1), np.nan),
                    zi=zi, an=(an1, an2, ra1, ra2))


# ---------------------------------------------------------------- 6. ringing / shear tell
def ideal_kernel(J, s, g, dil, xx, yy):
    """Continuous net kernel P_dil * Shear(P^-1) (anisotropic Gaussian) sampled at pixel offsets."""
    S = smat(*g); Ji = np.linalg.inv(J); Sig = Ji @ (s**2 * (dil**2 * np.eye(2) - S @ S.T)) @ Ji.T
    Si = np.linalg.inv(Sig); q = Si[0, 0] * xx**2 + 2 * Si[0, 1] * xx * yy + Si[1, 1] * yy**2
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(np.linalg.det(Sig)))


def test_kernel():
    """Impulse response of the operation vs the ideal kernel beyond 3 pixels.  The shear-dependent
    part K(g) - K(0) is what a network could learn as an unphysical 'tell'."""
    n = 61; a = np.zeros((n, n)); a[30, 30] = 1; yy, xx = np.mgrid[:n, :n] - 30.; r = np.hypot(yy, xx)
    for name, pix, fw, dil, rs in [('Halpha', 0.5, 2.5, ms.DIL_VEL, ms.RSEARCH_VEL), ('image', 0.396, 1.32, ms.DIL_IMG, ms.RSEARCH_IMG)]:
        J = np.array([[pix, 0.], [0., -pix]]); s = fw * FWHM2SIG; psf = galsim.Gaussian(fwhm=fw)
        for method, d_, tag in [('imcom', dil, '6'), ('galsim', ms.DIL, 'L2')]:
            e0 = ed = far = 0.
            for g in [(0.03, 0.03), (0.03, 0.), (-0.03, 0.03)]:
                K0 = ms.op(a, jwcs(J), psf, dil=d_, method=method, rsearch=rs)
                K = ms.op(a, jwcs(J), psf, g, dil=d_, method=method, rsearch=rs)
                I0, I = ideal_kernel(J, s, (0., 0.), d_, xx, yy), ideal_kernel(J, s, g, d_, xx, yy)
                e0 = max(e0, np.abs(K0 - I0)[r > 3].max() / I0.max()); ed = max(ed, np.abs((K - K0) - (I - I0))[r > 3].max() / I0.max())
                far = max(far, np.abs(K[r > rs + 3]).max())
            if method == 'imcom':
                check(f'6a {name}: |K - ideal| / peak beyond 3 px', e0, 2e-3, '{:.1e}', note=f'DIL {d_}, Rsearch {rs}')
                check(f'6b {name}: shear-dependent part |dK - dK_ideal| / peak beyond 3 px', ed, 3e-3, '{:.1e}',
                      note='the shear "tell"; physical shear signal is ~1e-2 of a pixel value')
                check(f'6c {name}: max |K| beyond Rsearch + 3 px (finite support)', far, 0., '{:.1e}')
            else:
                check(f'L2 legacy galsim {name}: shear-dependent part beyond 3 px', ed, 3e-3, '{:.1e}',
                      note=f'DIL {d_}: ringing of the original method', known=True)


# ---------------------------------------------------------------- 8. Halpha flux spikes
def test_spikes():
    from scipy.ndimage import binary_dilation
    gal = make_galaxy(pa=30.)
    # a very bright compact nucleus (1e4 x the disc), PSF-convolved by make_dict: must NOT be flagged
    nuc = Galaxy(gal.F + [(3e4, np.eye(2) * 0.05**2)], gal.V, gal.v_sys)
    d0, _ = make_dict(gal, AFFINES['SDSS-like'], img_noise=1e-3, rng=np.random.default_rng(0))
    d, aux = make_dict(nuc, AFFINES['SDSS-like'], img_noise=1e-3, rng=np.random.default_rng(0))
    m = d0['vmap']['gas']['mask'].astype(bool)               # the disc's gas mask
    F = d['vmap']['gas']['par_meta']['intensity_map']         # disc + bright nucleus, PSF-convolved
    n_fp = int(ms.halpha_spikes(F, m).sum())
    check('8a spikes: false positives on a 1e4-bright PSF-convolved nucleus', n_fp, 0, '{:d}',
          note=f'peak / disc median {F[m].max() / np.median(F[m]):.0f}')
    # inject single-spaxel spikes with wild velocities into the smooth galaxy
    d, aux = make_dict(gal, AFFINES['SDSS-like'], img_noise=1e-3, rng=np.random.default_rng(0))
    d_sp = __import__('copy').deepcopy(d); Gs = d_sp['vmap']['gas']; m = Gs['mask'].astype(bool)
    F0 = Gs['par_meta']['intensity_map']; cand = np.argwhere(m & (F0 > 8) & (F0 < 15))
    p2 = tuple(cand[np.argmax(np.hypot(*(cand - np.array([36, 30])).T))])   # faint disc spaxel far from p1
    pos = [(36, 30), p2]; amp = [1e4, 30.]
    for (r, c), a_ in zip(pos, amp):
        Gs['par_meta']['intensity_map'][r, c] *= a_; Gs['data'][r, c] = -1000.
    sp = ms.halpha_spikes(Gs['par_meta']['intensity_map'], m)
    truth = np.zeros_like(m); [truth.__setitem__(p_, True) for p_ in pos]
    check('8b spikes: injected spikes found exactly (missed + spurious)', int((sp ^ truth).sum()), 0, '{:d}',
          note='amplitudes 1e4 and 30 x the local flux')
    # with the spikes dropped, the velocity elsewhere equals that of the same galaxy with the
    # spike spaxels simply masked out (i.e. the spikes have no influence)
    g = (0.025, -0.02)
    o_sp, info = ms.process(d_sp, g, seed=2, compute_var=False)
    d_ref = __import__('copy').deepcopy(d); d_ref['vmap']['gas']['mask'] = (m & ~truth).astype(int)
    o_ref, _ = ms.process(d_ref, g, seed=2, compute_var=False)
    mo = o_sp['vmap']['gas']['mask'].astype(bool) & o_ref['vmap']['gas']['mask'].astype(bool)
    dv = np.abs(o_sp['vmap']['gas']['data'] - o_ref['vmap']['gas']['data'])[mo]
    check('8c spikes dropped: max |v - v(spike spaxels masked)| (km/s)', dv.max(), 1e-6, '{:.1e}',
          note=f"{info['gas']['n_spike']} spaxels dropped; output identical to masking them by hand")
    o_keep, _ = ms.process(d_sp, g, seed=2, compute_var=False, spikes=False)
    mk = mo & o_keep['vmap']['gas']['mask'].astype(bool) & ~binary_dilation(truth, iterations=2)
    dk = np.abs(o_keep['vmap']['gas']['data'] - o_ref['vmap']['gas']['data'])[mk]
    check('L3 legacy (spikes kept): max |v - v(spike masked)| > 2 spaxels from a spike (km/s)', dk.max(), 1.0, '{:.1f}',
          note='what an unflagged spike does to its surroundings', known=True)


# ---------------------------------------------------------------- 7. provenance of copied code
VENDOR_SHA256 = '387c37f3a99689e4bc1df9ee8c0ba12081dfbc7734e6aa24f62dc3df9096ad67'   # PyIMCOM 9ca33dd ginterp.py


def test_vendor():
    import hashlib
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_vendor', 'pyimcom_ginterp.py')
    txt = open(p, 'rb').read(); marker = b'original file follows (unmodified)'
    body = txt[txt.index(b'\n', txt.index(marker)) + 1:]
    ok = hashlib.sha256(body).hexdigest() == VENDOR_SHA256
    RESULTS.append(('7  _vendor/pyimcom_ginterp.py unmodified (SHA-256 of body)', 'match' if ok else 'MISMATCH', 'match',
                    'PASS' if ok else 'FAIL', 'PyIMCOM commit 9ca33dd, src/pyimcom/meta/ginterp.py'))


# ---------------------------------------------------------------- plots
def make_plots(dirn, worst, rows, noise):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(dirn, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    r = [x for x in rows if 'SDSS' in x['case'] and '-0.021' in x['case']][0]
    im = ax[0].imshow(r['dv_map'], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1); plt.colorbar(im, ax=ax[0], label='km/s')
    ax[0].contour(r['mask_in'], [0.5], colors='k', linewidths=0.6)
    ax[0].set_title(f"velocity response error (km/s)\n{r['case']}; black = input mask", fontsize=9)
    if noise is not None:
        im = ax[1].imshow(noise['r_img'], origin='lower', cmap='RdBu_r', vmin=0.7, vmax=1.3); plt.colorbar(im, ax=ax[1])
        ax[1].set_title('image: empirical / predicted variance', fontsize=9)
        im = ax[2].imshow(noise['r_vel'], origin='lower', cmap='RdBu_r', vmin=0.7, vmax=1.3); plt.colorbar(im, ax=ax[2])
        ax[2].set_title('velocity: empirical / predicted variance', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(dirn, 'synthetic_validation.png'), dpi=130)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--plots', default=None, help='directory for diagnostic plots')
    a = ap.parse_args(); t0 = time.time()
    test_vendor()
    worst = test_op_image(a.quick)
    test_op_velocity()
    rows = test_process_noiseless()
    test_conventions()
    noise = test_noise(60 if a.quick else 300, plots=a.plots)
    test_inpaint()
    test_noise(40 if a.quick else 200, tag='5', holes=0.12, label=' [patchy]')
    test_kernel()
    test_spikes()
    test_process_noiseless(pkw=dict(method='galsim', inpaint=None, dil=ms.DIL), legacy=True)
    w = max(len(r[0]) for r in RESULTS)
    print(f"{'test':<{w}}  {'value':>18}  {'limit':>9}  result")
    for name, val, lim, res, note in RESULTS:
        print(f'{name:<{w}}  {val:>18}  {lim:>9}  {res}' + (f'   ({note})' if note else ''))
    if a.plots: make_plots(a.plots, worst, rows, noise)
    nfail = sum(r[3] == 'FAIL' for r in RESULTS); nk = sum(r[3] == 'KNOWN' for r in RESULTS)
    print(f'\n{nfail} failed, {nk} known-limitation rows (see README)  ({time.time() - t0:.0f} s)')
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
