"""Real galaxies: compare the injected velocity response (sheared - control) of the legacy
settings (galsim, zero-fill: the first blind set) with the current defaults (IMCOM, GP fill),
by distance from the gas-mask edge.
Usage: python dev/real_compare.py --src "Manga data" --truth truth.csv [--n 100]"""
import argparse, sys, glob, os, time, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import manga_shear as ms
from scipy.ndimage import distance_transform_edt

LEGACY = dict(method='galsim', inpaint=None, spikes=False, dil=ms.DIL)


def one(src, f, g):
    d = joblib.load(os.path.join(src, f)); seed = ms.noise_seed(d['galaxy']['mangaid']); G = d['vmap']['gas']
    m = G['mask'].astype(bool) & np.isfinite(G['data']) & np.isfinite(G['var'])
    dist = np.round(distance_transform_edt(m)).astype(int)
    r = {}
    for name, kw in [('zero', LEGACY), ('gp', {})]:
        o, _ = ms.process(d, g, seed=seed, compute_var=False, **kw)
        oc, _ = ms.process(d, (0., 0.), seed=seed, compute_var=False, **kw)
        r[name] = o['vmap']['gas']['data'].astype(float) - oc['vmap']['gas']['data'].astype(float)
        mo = o['vmap']['gas']['mask'].astype(bool) & oc['vmap']['gas']['mask'].astype(bool)
    rows = []
    for k in range(2, 9):
        s = mo & (dist == k) if k < 8 else mo & (dist >= 8)
        for a, b in zip(np.abs(r['zero'][s] - r['gp'][s]), np.abs(r['gp'][s])):
            rows.append(dict(file=f, d=k, diff=a, resp=b))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--truth', required=True)
    ap.add_argument('--n', type=int, default=100); a = ap.parse_args()
    t = pd.read_csv(a.truth).sample(a.n, random_state=0)
    t0 = time.time()
    out = joblib.Parallel(n_jobs=4)(joblib.delayed(one)(a.src, f, (g1, g2)) for f, g1, g2 in zip(t.file, t.g1, t.g2))
    df = pd.DataFrame([r for rs in out for r in rs])
    # medians: robust to the few spaxels with input outliers / discontinuities
    agg = df.groupby('d').agg(spaxels=('diff', 'size'), median_abs_response_current=('resp', 'median'),
                              median_abs_legacy_minus_current=('diff', 'median'))
    agg['ratio'] = agg.median_abs_legacy_minus_current / agg.median_abs_response_current
    agg['share_of_output_spaxels'] = agg.spaxels / agg.spaxels.sum()
    print(agg.round(3).to_string())
    print(f'{time.time() - t0:.0f} s')
