"""Independent Fourier-space reference for the shear operation (cross-check only; not used by
the pipeline).  Ringing-free reference for op(): f' = K * Shear_c(f), with Shear_c a cubic-spline resampling
of the map onto sheared coordinates and K = P_dil * Shear(P^-1) the analytic anisotropic
Gaussian, covariance sigma^2 (dil^2 I - S S^T), applied exactly in Fourier space on a 2x
zero-padded grid (no truncation below Nyquist).  Valid for Gaussian PSFs."""
import numpy as np, galsim
from scipy.ndimage import map_coordinates


def op_local(arr, J, psf_sigma, g=(0., 0.), c=(0., 0.), dil=1.0848528137423856):
    a = np.asarray(arr, float); ny, nx = a.shape; pr, pc = (ny - 1) / 2, (nx - 1) / 2
    J = np.asarray(J, float); Ji = np.linalg.inv(J); S = galsim.Shear(g1=g[0], g2=g[1]).getMatrix()
    c = np.asarray(c, float)
    # 1. resample: output pixel at world x takes the input value at S^-1 (x - c) + c
    r, cc = np.mgrid[:ny, :nx].astype(float)
    x = J @ np.vstack([(cc - pc).ravel(), (r - pr).ravel()])
    src = Ji @ (np.linalg.inv(S) @ (x - c[:, None]) + c[:, None])
    b = map_coordinates(a, [src[1] + pr, src[0] + pc], order=3, mode='constant', cval=0.).reshape(a.shape)
    # 2. convolve with the analytic Gaussian K: Sigma (world) -> pixel units
    Sig = psf_sigma**2 * (dil**2 * np.eye(2) - S @ S.T)
    Sp = Ji @ Sig @ Ji.T                                   # covariance in (col, row) pixel units
    P = 2 * max(ny, nx); ky = 2 * np.pi * np.fft.fftfreq(P); kx = 2 * np.pi * np.fft.rfftfreq(P)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    T = np.exp(-0.5 * (Sp[0, 0] * KX**2 + 2 * Sp[0, 1] * KX * KY + Sp[1, 1] * KY**2))
    out = np.fft.irfft2(np.fft.rfft2(b, (P, P)) * T, (P, P))
    return out[:ny, :nx]
