"""Real velocity maps: injected response (sheared - control) from the pipeline's IMCOM operation,
the independent Fourier-space reference (dev/fourier_reference.py) and the legacy galsim
operation, on GP-filled maps; pairwise disagreement relative to the response size.
Usage: python dev/compare_methods.py --src "Manga data" --truth truth.csv [--n 50] [--dil 1.15]"""
import argparse, os, sys, numpy as np, pandas as pd, joblib, galsim
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import manga_shear as ms
from fourier_reference import op_local
from scipy.ndimage import binary_erosion, distance_transform_edt


def one(src, f, g, dil):
    d = joblib.load(os.path.join(src, f)); G = d['vmap']['gas']; gm = G['par_meta']
    E, N = gm['RA_grid'].astype(float), gm['Dec_grid'].astype(float); Jv = ms.vmap_jacobian(gm); ny, nx = E.shape
    c = (-np.interp((nx - 1) / 2, np.arange(nx), E[0]), -np.interp((ny - 1) / 2, np.arange(ny), N[:, 0]))
    vfw = float(np.interp(ms.HA * (1 + d['galaxy']['redshift']), gm['psf_wavel'], gm['psf_fwhm'])); s = vfw * ms.FWHM2SIG
    F = np.nan_to_num(gm['intensity_map'].astype(float))
    m = G['mask'].astype(bool) & np.isfinite(G['data']) & np.isfinite(G['var'])
    m &= ~ms.halpha_spikes(F, m)
    if m.sum() < 50:
        return []
    M0 = np.where(m, F, 0.); M1 = M0 * np.where(m, G['data'], 0.)
    ell = np.sqrt(2) * s / np.sqrt(abs(np.linalg.det(Jv))); vr = np.where(m, np.clip(G['var'] - ms.V_FLOOR, 1e-2, None), 0.)
    f1 = ms.make_fill(M1, m, ell, noise_var=(M0**2 * vr)[m]); f0 = ms.make_fill(M0, m, ell); M1, M0 = f1(M1), f0(M0)
    w = galsim.JacobianWCS(*Jv.ravel()); psf = galsim.Gaussian(fwhm=vfw)
    ops = {'imcom': lambda X, gg: ms.op(X, w, psf, gg, c, dil=dil),
           'fourier': lambda X, gg: op_local(X, Jv, s, gg, c, dil=dil),
           'galsim': lambda X, gg: ms.op(X, w, psf, gg, c, dil=dil, method='galsim')}
    R = {}
    with np.errstate(all='ignore'):
        for k, op in ops.items():
            R[k] = op(M1, g) / op(M0, g) - op(M1, (0., 0.)) / op(M0, (0., 0.))
    mo = binary_erosion(m, iterations=1) & np.all([np.isfinite(r) for r in R.values()], axis=0)
    dist = np.round(distance_transform_edt(m)).astype(int)
    return [(min(dist.flat[i], 8), abs(R['imcom'].flat[i]), abs(R['fourier'].flat[i] - R['imcom'].flat[i]),
             abs(R['galsim'].flat[i] - R['imcom'].flat[i])) for i in np.flatnonzero(mo & (dist >= 2))]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--truth', required=True)
    ap.add_argument('--n', type=int, default=50); ap.add_argument('--dil', type=float, default=ms.DIL_VEL); a = ap.parse_args()
    t = pd.read_csv(a.truth).sample(a.n, random_state=1)
    out = joblib.Parallel(n_jobs=8)(joblib.delayed(one)(a.src, f, (g1, g2), a.dil) for f, g1, g2 in zip(t.file, t.g1, t.g2))
    df = pd.DataFrame([r for o in out for r in o], columns=['d', 'resp', 'fourier_vs_imcom', 'galsim_vs_imcom'])
    agg = df.groupby('d').median()
    for col in ['fourier_vs_imcom', 'galsim_vs_imcom']:
        agg[col] = agg[col] / agg['resp']
    print(f'DIL {a.dil}: median |difference| / median |response| (response from imcom, km/s); d = spaxels from mask edge (8 = 8+)')
    print(agg.round(4).to_string())
