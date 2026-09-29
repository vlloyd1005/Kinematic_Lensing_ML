"""
kl_geometry.py
==============
Shared projection geometry for the KL pipeline.

Import this from BOTH generate_kl_tng50.py and test_kl_pipeline.py so that
the same (inclination, theta_int) always produces the same projection.

Pipeline order inside each script:
    centre particles  ->  subtract bulk velocity
    ->  face_on_rotation()   (galaxy's stellar angular momentum -> +z)
    ->  rotate_to_los()      (tilt by inclination, then turn by theta_int)

After these two steps:
    inclination = angle between the stellar disc normal and the line of sight
    theta_int   = position angle of the projected major axis, measured
                  counter-clockwise from the image +x axis
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_to_z(vec: np.ndarray) -> Rotation:
    """
    Shortest rotation taking the direction of `vec` onto +z.
    Deterministic (no arbitrary spin about the axis), unlike
    Rotation.align_vectors with a single vector pair.
    """
    v = vec / np.linalg.norm(vec)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)          # sin(angle)
    c = float(np.dot(v, z))           # cos(angle)
    if s < 1e-12:                     # already along +z or -z
        return Rotation.identity() if c > 0 else Rotation.from_euler("x", np.pi)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c))


def angular_momentum_direction(pos, vel, mass, r_max, min_particles=50):
    """
    Unit vector of the mass-weighted angular momentum of particles within
    r_max of the centre. Falls back to all particles if too few are inside.
    pos, vel must already be centred / bulk-velocity subtracted.
    """
    r = np.linalg.norm(pos, axis=1)
    sel = r < r_max
    if sel.sum() < min_particles:
        sel = np.ones(len(pos), dtype=bool)
    L = np.sum(mass[sel, None] * np.cross(pos[sel], vel[sel]), axis=0)
    norm = np.linalg.norm(L)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Angular momentum is zero or undefined")
    return L / norm


def face_on_rotation(pos, vel, mass, r_max, min_particles=50):
    """
    Rotation that makes the galaxy face-on (angular momentum along +z).
    Returns (rotation, L_hat) where L_hat is the original-frame direction.
    Compute it from the STARS and apply the same rotation to the gas.
    """
    L_hat = angular_momentum_direction(pos, vel, mass, r_max, min_particles)
    return rotation_to_z(L_hat), L_hat


def sky_rotation_matrix(inclination: float, theta_int: float) -> np.ndarray:
    """
    R = Rz(theta_int) @ Rx(inclination)  (Euler angles phi, i, psi=0).
    Identical to the matrix in the original generate_kl_tng50.py.
    """
    cp, sp = np.cos(theta_int), np.sin(theta_int)
    ci, si = np.cos(inclination), np.sin(inclination)
    return np.array([
        [cp, -ci * sp,  si * sp],
        [sp,  ci * cp, -si * cp],
        [0.0,      si,       ci],
    ])


def rotate_to_los(coords, vels, inclination, theta_int):
    """
    Project a FACE-ON galaxy to the sky.
    Returns x_im, y_im (kpc) and v_los (km/s, positive = receding).
    """
    R = sky_rotation_matrix(inclination, theta_int)
    r_rot = coords @ R.T
    v_rot = vels @ R.T
    return r_rot[:, 0], r_rot[:, 1], -v_rot[:, 2]


def realized_orientation(rot_face: Rotation, L_hat: np.ndarray,
                         inclination: float, theta_int: float):
    """
    Push the disc normal through both rotations and report the inclination
    and major-axis position angle actually achieved (radians).
    Should reproduce the requested values to numerical precision.
    """
    n = sky_rotation_matrix(inclination, theta_int) @ rot_face.apply(L_hat)
    inc = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    pa = float(np.arctan2(n[0], -n[1]))      # major axis is perpendicular
    return inc, pa                            # to the projected normal


def misalignment_deg(L_a: np.ndarray, L_b: np.ndarray) -> float:
    """Angle between two angular-momentum directions (degrees)."""
    return float(np.degrees(np.arccos(np.clip(np.dot(L_a, L_b), -1.0, 1.0))))