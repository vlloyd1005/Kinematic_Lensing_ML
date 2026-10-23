# MaNGA shear injection

Adds a known weak shear to MaNGA `data_info-*.pkl` files: the SDSS r-band image and the
Hα gas velocity map. It also makes the matching no-shear control set.

| file | what it is |
|---|---|
| `manga_shear.py` | the pipeline, usable as a library or from the command line |
| `imcom_shear.py` | our adapter that builds the shear operation from PyIMCOM's `InterpMatrix` |
| `_vendor/pyimcom_ginterp.py` | **third-party**: PyIMCOM's `ginterp.py` by Christopher M. Hirata, copied unmodified (MIT). See [Credits](#credits-and-citation) |
| `test_synthetic.py` | checks the pipeline against synthetic galaxies with exact closed-form answers |
| `requirements.txt` | the package versions it was validated with |
| `dev/` | cross-checks and exploration scripts (not needed to run): `fourier_reference.py` (an independent Fourier-space implementation of the operation), `compare_methods.py`, `real_compare.py`, `explore_inpaint.py` |

No data is included.

## Install

```
pip install -r requirements.txt      # galsim, numpy, scipy, pandas, joblib (+ matplotlib for plots)
```

## Run

```
# no-shear control: same smoothing and noise treatment as the sheared sets, g = 0
python manga_shear.py noshear --src "Manga data" --out no-shear --truth no-shear_info.csv

# the same shear for every galaxy
python manga_shear.py fixed --src "Manga data" --out g1p02 --g1 0.02 --g2 0 --truth g1p02_info.csv

# a different random shear per galaxy (uniform in [-gmax, gmax] per component)
python manga_shear.py random --src "Manga data" --out random-shear --truth truth/random_truth.csv --seed 12345
```

The defaults are the recommended settings:
- `--method imcom`;
- PSF dilation 1.2 for the image and 1.15 for Hα;
- `--inpaint gp`;
- `--erode 1`;
- Hα flux spikes dropped from the gas mask.

Each run also writes two files to `--out`. They depend only on the input data, not on the
shear, so they can be shipped with a blind set:
- `halpha_spike_flags.csv`: galaxies with flux spikes and the number of spaxels dropped;
- `halpha_spike_mask.npz`: a 72×72 boolean map per flagged galaxy (key = file name without
  `.pkl`), True on the dropped spaxels.

Options:
- `--method`: `imcom` (default) or `galsim`. `galsim` is the original implementation. It rings
  (see [Legacy method](#legacy-method-and-the-first-blind-set)), and is kept only to reproduce the
  first blind set.
- `--dil-img`, `--dil-vel`: PSF dilation per map (defaults 1.2 and 1.15).
- `--inpaint`: `gp` (default) fills the gas maps outside the mask before shearing; `none` uses
  zeros, which biases the result near mask edges.
- `--erode N`: trim N spaxels from the gas-mask edge in the output (default 1).
- `--ell-scale`: GP kernel length in units of √2·σ_PSF (default 1; leave it).
- `--jobs`: number of parallel workers (default 8).
- `--only f1.pkl f2.pkl ...`: process only these files. In random mode, the shears are still
  drawn for the full sorted file list, so each galaxy gets the same shear as in a full run.
- `--gmax`: the largest shear per component for random draws. The code checks that the
  dilations are large enough for it.
- `--no-spikes`: keep Hα flux spikes in the gas mask. Only for reproducing the first set.
- `--noise-salt N`: a secret integer mixed into the per-galaxy noise seeds. Use one for blind
  sets (see below).

To reproduce the first blind set's data maps exactly:
`--method galsim --inpaint none --no-spikes --seed 20261001`. Its `var` maps will differ slightly, because
the variance propagation has since been corrected.

**Speed:** about 3 s per galaxy per core, mostly for the variance maps; a full 1135-galaxy set
took about 10 minutes on 8 workers. In library use, `process(..., compute_var=False)`
skips the variance maps (about 0.9 s per galaxy).

**Blind tests.** The truth table goes to `--truth`, so keep it outside `--out`. Its
`*_provenance.json` records the shear seed and the noise salt, so keep it private too. For a
blind set:
- **Use a new `--seed`.** The default, 20261001, gives the first blind set's shears.
- **Use a secret `--noise-salt`.** The pipeline is deterministic. Without a salt the noise seeds
  are crc32(mangaid), so anyone with this package and the original data could regenerate the
  sheared maps for trial shears and match them pixel for pixel. Use the same salt for the
  blind set and its no-shear control.
- **Keep the no-shear control private.** Subtracting it from the sheared set cancels most of the
  noise and exposes the shear.

Output files have the same names and format as the input files (load them with `joblib.load`).
The truth CSV is written with a `*_provenance.json` file that records every setting, including
the method and the PyIMCOM commit used.

## What it does

The same operation is applied to each linear map f. Those are the image, the Hα flux
M0 (`intensity_map`) and the flux-weighted velocity M1 = M0·v:

    f' = P_dil ⊗ Shear_g( P⁻¹ ⊗ f )

Here P is a Gaussian PSF with the FWHM stored in the file, and P_dil is that PSF dilated by a
factor `dil`. The new velocity map is v' = M1'/M0'.

### How the operation is computed (IMCOM)

For Gaussian PSFs, the operation equals two steps: resample the map onto sheared coordinates,
then blur it with a small Gaussian whose covariance is σ²(dil²·I − S·Sᵀ), where S is the shear
matrix. We compute it as a configuration-space IMCOM solve (Rowe, Hirata & Rhodes 2011), using
`InterpMatrix` from PyIMCOM's `pyimcom.meta.ginterp`, written by Christopher M. Hirata:

- **Weights:** for each output pixel, `InterpMatrix` finds the weights over input pixels
  within a search radius that best reproduce the target PSF. The target is the input PSF
  convolved with the extra Gaussian smoothing, sheared and grown by `dil`.
- **Locality:** the weights are exactly zero beyond the search radius (6 pixels for the image,
  8 spaxels for Hα), so the operation can't ring across the map.
- **Matrix:** `imcom_shear.py` arranges the weights as a sparse matrix M over the input grid,
  padded so pixels near the array edge get a full neighbourhood.
- **Noise:** the output is M @ input. For independent input noise the output variance is
  exactly (M²) @ variance.

**Why the dilations are 1.2 and 1.15.** With the old dilation of 1.085, the net blurring kernel
was narrower than a pixel. The pixel grid can't represent a kernel that narrow, so any
implementation leaves pixel-scale artifacts that depend on the shear. At 1.15 for Hα (FWHM
2.5" → 2.9") and 1.2 for the image (1.32" → 1.58"), the kernel is resolved by the pixels.

### Noise symmetrisation and the control

The shear also makes the noise slightly direction-dependent. To cancel that, a second noise
field with the data's statistics is rotated by 90°, put through the same operation, rotated
back and added (the metacal "fixnoise" step). The PSF FWHMs stored in the file are multiplied
by the dilations, and the gas `mask` is eroded by `--erode` spaxels.

The no-shear control uses g = 0 and the same per-galaxy noise seed, crc32(mangaid). That means
sheared and control files get matching noise draws.

### Gas-map inpainting (`--inpaint gp`)

Outside the gas mask, M0 and M1 are unknown. Setting them to zero corrupts the result near
every mask edge and hole. With `--inpaint gp` (the default), both maps are filled outside the
mask with the conditional mean of a Gaussian process:

- **Kernel:** squared exponential with length √2·σ_PSF, which is the autocorrelation length of
  a PSF-convolved field. Longer or shorter kernels were worse.
- **Noise:** the per-spaxel noise variance of M1, plus a 1% "jitter" floor relative to the
  signal variance. With a smaller floor, the GP treats pixel noise as signal and extrapolates it
  about 100× amplified.
- **Signal variance:** estimated from the inside values minus the noise variance, so the fill
  does not depend on the noise level.

The fill is linear in the data. The fixnoise field gets the same fill, and the variance maps
include the noise that the fill carries outside the mask.

### Variance maps

The `var` maps are propagated exactly: through the IMCOM matrix, through the fixnoise term
(the matrix of the rotated operation), and through the GP fill. Monte Carlo tests confirm this
to the sampling limit, including at faint mask edges.

Effects on the noise, as medians over the 1135 galaxies:
- The image noise variance is about 0.18× the input, and the velocity noise variance about
  0.11×. That's lower than with the old dilation, because the maps are smoothed a bit more.
- The noise is now correlated between neighbouring pixels.
- The image `var` keeps the on-disk convention var = image + mean_bkg², scaled by the change in
  the true noise variance.

## Shear conventions

The shear is defined on the sky, about the galaxy centre:
- u = +East (ΔRA·cos Dec) and v = +North, in arcsec;
- **g1 > 0** stretches East–West;
- **g2 > 0** stretches along PA = 45° (NE–SW).

The two maps sit on differently oriented pixel grids:
- **Image:** the code uses the image WCS at the galaxy position to get the true East/North
  directions. The stored WCS describes the full SDSS frame, not the 65×65 cutout, so it gives
  the correct axis directions and pixel scale, but the galaxy is assumed to be at the centre of
  the cutout. The orientation differs from galaxy to galaxy.
- **Velocity map:** East/North come from `RA_grid`/`Dec_grid`, and the grid is the same for
  every galaxy: columns run East, rows run South.

The truth table gives the same shear in each map's own array axes (x = column, y = row):

| column | meaning |
|---|---|
| `g1`, `g2` | sky frame |
| `g1_img_pix`, `g2_img_pix` | r-band image array axes (rotation differs per galaxy) |
| `g1_vel_pix`, `g2_vel_pix` | velocity-map array axes; equal to (g1, −g2) |

If you measure ellipticity in raw array coordinates, compare against the `*_pix` columns for
that map. If your y axis points up on a displayed image (`origin='upper'`), flip the sign of g2.

## Validation (`python test_synthetic.py [--quick] [--plots DIR]`)

The synthetic galaxies are sums of elliptical Gaussians for the flux and Hermite–Gaussian
terms for flux × velocity. For these shapes, shearing and Gaussian PSF convolution have exact
formulas, so every output can be predicted without galsim. The patchy-mask cases knock 12% of
the mask out as holes. The full run takes about 15 minutes and the last output is in
`validation_plots/test_output.txt`.

Results with the default settings (all pass):

| check | result |
|---|---|
| image, noise-free, 3 pixel orientations × 4 shears | max error 3e-6 of peak; 0.03% of the shear-induced change |
| velocity path, no mask, off-centre galaxy | max error 7e-7 km/s; shear response correct to 2e-7 |
| full `process()` with a celestial WCS at Dec 30° | image within 6e-7 of peak (2e-5 in the outer 6 px, where light beyond the cutout is unknown) |
| velocity response, ≥ 3 spaxels from a mask edge | within 0.6% of the shear signal (including the small added noise) |
| velocity response, ≤ 2 spaxels from a mask edge | within 2% of the signal (4% for spaxels touching the edge diagonally); limited by the GP fill |
| sign conventions on the sky and in both pixel frames | measured elongation matches the injected angle to < 0.01° |
| predicted vs empirical noise variance (200–300 realisations) | image within 0.3%, velocity within 2.5% (median); per-pixel scatter at the sampling limit, including at mask edges |
| mean of noisy outputs vs exact answer | no bias detected, for the maps and for sheared − control |
| noise isotropy after fixnoise | no detectable anisotropy (\|z\| ≤ 1.2; without fixnoise, \|z\| = 33–35) |
| shear "tell": shear-dependent kernel error beyond 3 px | 7e-4 of the peak (Hα), 1.1e-3 (image); exactly zero beyond the search radius + 3 px |
| Hα spikes: bright PSF-convolved nucleus (1480× the disc median) | not flagged |
| Hα spikes: injected 10⁴× and 30× single-spaxel spikes | both found, nothing else flagged; output elsewhere identical to masking them by hand |
| `_vendor/pyimcom_ginterp.py` unmodified | SHA-256 of the copied code matches upstream |

**Real data:** on 50 real galaxies with GP-filled maps, the IMCOM response agrees with an
independent Fourier-space implementation (`dev/fourier_reference.py`; run
`dev/compare_methods.py`) to 0.1–0.2% of the response. The galsim method differs from both by
2–11%.

Across all 1135 galaxies there are no failures or non-finite values. Spaxels whose velocity
changes by more than 100 km/s fell from 0.09% (first blind set) to 0.012%. See
[Hα flux spikes](#hα-flux-spikes).

## Hα flux spikes

Some Hα intensity maps have single spaxels, or tight clumps, far brighter than anything the
PSF allows: up to 10⁷× the disc, from bad pixels or unresolved sources. Left in, a spike
dominates the flux-weighted velocity around it. It also inflates the GP fill's amplitude, so the
fill stops following the data at faint mask edges; in the first blind set this gave velocity
errors of hundreds to thousands of km/s, some of them far from the spike.

**Detection** (`halpha_spikes`): a gas spaxel is a spike if its flux is more than 10× the median
flux of the in-mask spaxels 3–4.5 spaxels away. Where fewer than 6 of those are in the mask,
the galaxy's median flux is used instead. The rule is grounded in the PSF: the intensity map is
PSF-convolved (σ ≈ 2–2.7 spaxels), so even a point source is at most about 4–5× brighter than
that ring. A bright nucleus 1480× the disc median, convolved with the PSF, is not flagged (test 8a).

**Treatment:** spike spaxels are dropped from the gas mask before filling and shearing, exactly
as if they had been masked, and they are listed in `halpha_spike_mask.npz` and
`halpha_spike_flags.csv`. Across the 1135 galaxies this flags 61 spaxels in 21 galaxies.

**Effect:** with spikes dropped, no flagged galaxy has a spaxel whose velocity changes by more than
100 km/s; with them kept (the first blind set), 144 such spaxels, up to 7,200 km/s. Over the
whole sample, 0.012% of output spaxels still change by more than 100 km/s, all in galaxies
without spikes. The cases inspected are bad input velocities or real velocity discontinuities,
which any PSF smoothing changes legitimately.

## Legacy method and the first blind set

The first blind set used `--method galsim --inpaint none --no-spikes` at dilation 1.085, and
has three problems: the two below, plus Hα flux spikes left in (see above).
- **Zero-fill edge bias:** the injected velocity change is wrong near mask edges. On real
  galaxies, zero-fill differs from the inpainted result by a median of 40%, 21% and 13% of the
  typical response at 2, 4 and 6 spaxels from an edge. Real gas masks are patchy: 18% of output
  spaxels lie within 2 spaxels of an edge, 44% within 4 and 62% within 6.
- **Ringing:** galsim removes the PSF in Fourier space and cuts the spectrum off at the PSF's
  own maximum frequency. With the net kernel narrower than a pixel, the cut makes it ring at
  1–4% of its peak out past 15 pixels. The rings change with the shear, so every sharp feature
  carries a shear-dependent ripple pattern that a CNN could learn instead of the real distortion.
  On real velocity maps this added 5–11% of the response, as stripes of several km/s at faint
  outskirts and edges. In images it shows as rings around bright stars.

All three are fixed by the defaults above. Tests L1–L3 in `test_synthetic.py` keep these
numbers as known-limitation rows.

## Credits and citation

The shear operation's numerical core is **`InterpMatrix` from PyIMCOM**
(<https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom>, file `src/pyimcom/meta/ginterp.py`):
- **Author:** Christopher M. Hirata, with formatting and packaging changes by Axel Guinot.
- **Version:** copied unmodified from commit `9ca33dd` into `_vendor/pyimcom_ginterp.py`.
- **License:** MIT, Copyright (c) 2023 Kaili Cao (the PyIMCOM copyright notice). The full
  license text is in `_vendor/LICENSE-pyimcom`.
- **More detail:** authorship, checksum and update instructions are in `_vendor/README.md`.

PyIMCOM's authors are Kaili Cao, Christopher Hirata and Katherine Laliotis, and it is
maintained by the Roman HLIS Cosmology Project Infrastructure Team. If you use this pipeline,
please cite:
- Rowe, Hirata & Rhodes (2011), ApJ 741:46, "Optimal linear image combination";
- Hirata et al. (2024), MNRAS 528:2533, "Simulating image coaddition with the Nancy Grace Roman Space Telescope I";
- Cao et al. (2025), ApJS 277:55, "Simulating image coaddition with the Nancy Grace Roman Space Telescope III";

and acknowledge PyIMCOM.
