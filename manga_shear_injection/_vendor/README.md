# Copied third-party code

## `pyimcom_ginterp.py`: from PyIMCOM

| | |
|---|---|
| Upstream project | PyIMCOM, <https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom> |
| Upstream file | `src/pyimcom/meta/ginterp.py` |
| Commit | `9ca33dd24aa87159ac9689faef49a1b7b49fcb22` (2026-09-09), retrieved 2026-10-02 |
| SHA-256 of the upstream file | `387c37f3a99689e4bc1df9ee8c0ba12081dfbc7734e6aa24f62dc3df9096ad67` |
| Author of the code | Christopher M. Hirata (Ohio State University); formatting and packaging changes by Axel Guinot |
| PyIMCOM authors | Kaili Cao, Christopher Hirata, Katherine Laliotis; maintained by the Roman HLIS Cosmology Project Infrastructure Team |
| License | MIT, Copyright (c) 2023 Kaili Cao; full text in [`LICENSE-pyimcom`](LICENSE-pyimcom) |

**What it is.** `InterpMatrix` builds the weights that take a gridded image with a Gaussian PSF
to a resampled image with extra Gaussian smoothing. That covers deconvolution, shear,
reconvolution and resampling in one linear step. It does this by solving the IMCOM problem in
configuration space: it minimises the leakage between the achieved and target output PSF, with
weights confined to a finite search radius. In PyIMCOM it drives the `pyimcom.meta`
("meta-ing") afterburner for coadded images; see `docs/meta_README.rst` upstream.

**What we changed.** Nothing. The file is the upstream file with a comment header added above
it. Everything below the header's marker line is byte-identical to the upstream file, and
`test_synthetic.py` checks the SHA-256 above on every run. Our adaptation is in
`../imcom_shear.py`, which calls `InterpMatrix` and arranges its output as a sparse matrix
over our pixel grids. It is a separate file in this package; it is not part of PyIMCOM.

**Please cite**, when using the IMCOM-based shear operation:
- Rowe, Hirata & Rhodes (2011), ApJ 741:46, "Optimal linear image combination"
- Hirata et al. (2024), MNRAS 528:2533, "Simulating image coaddition with the Nancy Grace Roman Space Telescope I"
- Cao et al. (2025), ApJS 277:55, "Simulating image coaddition with the Nancy Grace Roman Space Telescope III"

and acknowledge PyIMCOM (link above).

**To update**, replace everything below the marker line with the new upstream file. Then
update the commit and SHA-256 in the header, in this README and in `test_synthetic.py`, and
rerun the tests.
