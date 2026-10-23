"""Shear operation built on the IMCOM interpolation matrix from PyIMCOM.

The numerical core is `InterpMatrix` from PyIMCOM's `pyimcom.meta.ginterp`, written by
Christopher M. Hirata, copied unmodified into `_vendor/pyimcom_ginterp.py` (MIT license;
provenance, authorship and citations in `_vendor/README.md`).  For a gridded map with a
Gaussian PSF, it returns the weights over input pixels within a search radius that best
reproduce, at each output position, a target PSF equal to the input PSF convolved with an
extra Gaussian smoothing (IMCOM; Rowe, Hirata & Rhodes 2011).  It also reports the leakage U
and noise factor Sigma for each output point.

This module is ours, not part of PyIMCOM.  It
  1. sets the geometry for a shear about a world-frame centre c: output pixel p takes input
     position p_in = A p + origin, with A = J^-1 S^-1 J (J: pixel -> world Jacobian), and the
     target PSF is the input PSF grown by `dil` in the output frame, so the extra smoothing
     covariance in input pixels is sigma_pix^2 (dil^2 A A^T - I) (as in PyIMCOM's
     `MetaMosaic.shearimage`);
  2. calls InterpMatrix for every output pixel and arranges the weights as a sparse matrix M
     over the input grid padded by `pad` pixels.  The index bookkeeping (integer and fractional
     parts of p_in, offsets of the returned weights) follows PyIMCOM's `ginterp.MultiInterp`;
     MultiInterp itself is not used because it masks output pixels near array edges, whereas we
     pad the input instead and keep every output pixel.
Applying the shear is then out = M @ padded_input, and for independent input noise the output
variance is exactly (M**2) @ padded_variance.
"""
import numpy as np, galsim
import scipy.sparse as sp
from _vendor.pyimcom_ginterp import InterpMatrix          # PyIMCOM code, unmodified (see _vendor/)

FWHM2SIG = 1 / (2 * np.sqrt(2 * np.log(2)))


def geometry(J, g, c):
    """A (d p_in / d p_out), and the shear centre in pixel offsets, for a shear g about world
    offset c (from the array centre); J is the pixel (col, row) -> world (u, v) Jacobian."""
    J = np.asarray(J, float); Ji = np.linalg.inv(J)
    S = galsim.Shear(g1=g[0], g2=g[1]).getMatrix()
    return Ji @ np.linalg.inv(S) @ J, Ji @ np.asarray(c, float)


def check_stable(dil, g):
    """The extra smoothing must be positive definite: dil^2 >= (1+|g|)/(1-|g|)."""
    gm = float(np.hypot(*g))
    if dil**2 < (1 + gm) / (1 - gm):
        raise ValueError(f'dilation {dil} too small for |g| = {gm:.4f} (need >= {np.sqrt((1 + gm) / (1 - gm)):.4f})')


def pad_width(shape, rsearch, gmax=0.1):
    """Input padding that holds every pixel the weights can touch: the search radius plus the
    largest shear displacement (|g| times the array half-diagonal) plus a margin."""
    return int(np.ceil(rsearch + gmax * np.hypot(*shape) / 2)) + 3


def transfer(shape, J, psf_sigma, g=(0., 0.), c=(0., 0.), dil=1.15, rsearch=8.0, pad=None):
    """Sparse IMCOM shear matrix.

    Returns (M, pad, Umax, Smax): M has shape (ny*nx, (ny+2pad)*(nx+2pad)) and maps the input
    padded by `pad` pixels (np.pad layout) to the output on the original grid.  psf_sigma is
    the input PSF sigma in world units (same units as J).  Umax and Smax are the largest leakage
    and noise factor that InterpMatrix reports over the output pixels."""
    check_stable(dil, g)
    ny, nx = shape; pad = pad_width(shape, rsearch) if pad is None else pad
    A, cpx = geometry(J, g, c)
    p0 = np.array([(nx - 1) / 2, (ny - 1) / 2])
    sig_pix = psf_sigma / np.sqrt(abs(np.linalg.det(J)))
    Cv = sig_pix**2 * (dil**2 * A @ A.T - np.eye(2))
    yo, xo = np.mgrid[:ny, :nx]
    q = A @ np.vstack([xo.ravel() - p0[0] - cpx[0], yo.ravel() - p0[1] - cpx[1]]) + (p0 + cpx)[:, None]
    x_in, y_in = q[0] + pad, q[1] + pad                       # positions in the padded input
    xi, yi = np.floor(x_in).astype(np.int64), np.floor(y_in).astype(np.int64)
    offx, offy, T, U, S = InterpMatrix(rsearch, sig_pix / FWHM2SIG, x_in - xi, y_in - yi,
                                       [Cv[0, 0], Cv[0, 1], Cv[1, 1]])
    cols_x = xi[:, None] + offx[None, :]; cols_y = yi[:, None] + offy[None, :]
    npx, npy = nx + 2 * pad, ny + 2 * pad
    if cols_x.min() < 0 or cols_y.min() < 0 or cols_x.max() >= npx or cols_y.max() >= npy:
        raise ValueError('IMCOM weights reach outside the padded input; increase pad')
    rows = np.broadcast_to(np.arange(ny * nx)[:, None], T.shape)
    M = sp.csr_matrix((T.ravel(), (rows.ravel(), (cols_y * npx + cols_x).ravel())), shape=(ny * nx, npy * npx))
    return M, pad, float(U.max()), float(S.max())


def rotated(M, shape, pad):
    """The matrix of x -> rot90^-1( M_op( rot90(x) ) ) for a square grid, where M_op is the
    operation represented by M (input padded by `pad`).  Used for the fixnoise term."""
    ny, nx = shape; assert ny == nx
    n_pad = nx + 2 * pad
    rows_map = np.rot90(np.arange(ny * nx).reshape(ny, nx), -1).ravel()
    cols_map = np.rot90(np.arange(n_pad * n_pad).reshape(n_pad, n_pad)).ravel()
    return M[rows_map][:, np.argsort(cols_map)]


def apply(M, arr, pad, pad_sigma=None, seed=None):
    """out = M @ arr padded by `pad` pixels with zeros, or with white noise of rms pad_sigma."""
    a = np.asarray(arr, float)
    if pad_sigma:
        big = np.random.default_rng(seed).normal(0, pad_sigma, (a.shape[0] + 2 * pad, a.shape[1] + 2 * pad))
        big[pad:-pad, pad:-pad] = a
    else:
        big = np.pad(a, pad)
    return (M @ big.ravel()).reshape(a.shape)
