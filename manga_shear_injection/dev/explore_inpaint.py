"""Compare inpainting settings: velocity response error vs distance from the gas-mask edge
(synthetic galaxy; run from the package directory: python dev/explore_inpaint.py)."""
import sys, time, numpy as np
sys.path.insert(0, '.')
import manga_shear as ms, test_synthetic as T
from scipy.ndimage import distance_transform_edt, gaussian_filter

def patchy(mask, seed, frac=0.12):
    r = np.random.default_rng(seed); f = gaussian_filter(r.standard_normal(mask.shape), 1.5)
    return mask & (f < np.quantile(f[mask], 1 - frac))

def run(d, aux, g, mask, **kw):
    d = {**d, 'vmap': {'gas': {**d['vmap']['gas'], 'mask': mask.astype(int)}}}
    o, _ = ms.process(d, g, seed=3, erode=0, **kw); oc, _ = ms.process(d, (0., 0.), seed=3, erode=0, **kw)
    return o['vmap']['gas']['data'] - oc['vmap']['gas']['data'], o['vmap']['gas']['mask'].astype(bool)

gal = T.make_galaxy(pa=30.)
d, aux = T.make_dict(gal, T.AFFINES['SDSS-like'], img_noise=1e-3, rng=np.random.default_rng(0))
cfgs = [('zero-fill (original)', dict(inpaint=None)), ('gp ell x0.7', dict(inpaint='gp', ell_scale=0.7)),
        ('gp ell x1.0', dict(inpaint='gp', ell_scale=1.0)), ('gp ell x1.4', dict(inpaint='gp', ell_scale=1.4))]
for mname, mask in [('ellipse mask', aux['mask']), ('patchy mask', patchy(aux['mask'], 1))]:
    dist = np.round(distance_transform_edt(mask)).astype(int)
    print(f'\n{mname}: {mask.sum()} spaxels;  share within d<=1,2,3: ' +
          ', '.join(f'{100*np.mean(dist[mask]<=k):.0f}%' for k in (1, 2, 3)))
    print(f"{'':22}" + ''.join(f'  d={k:<5}' for k in range(1, 9)) + '   (max |response err| / max signal)')
    for name, kw in cfgs:
        t = time.time(); errs = np.zeros(8)
        for g in [(0.03, 0.), (0., -0.03), (-0.021, 0.027)]:
            resp, mo = run(d, aux, g, mask, **kw)
            ex = T.expected(gal, aux, g)['vel'] - T.expected(gal, aux, (0., 0.))['vel']
            e = np.abs(resp - ex); sig = np.abs(ex)[mask].max()
            for k in range(1, 9):
                s = mo & (dist == k) if k < 8 else mo & (dist >= 8)
                if s.any(): errs[k - 1] = max(errs[k - 1], e[s].max() / sig)
        print(f'{name:22}' + ''.join(f'  {x:7.4f}' for x in errs) + f'   {(time.time()-t)/6:.2f} s/process')
