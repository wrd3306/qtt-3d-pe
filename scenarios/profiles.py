"""Refractivity M(z) profiles for the six propagation scenarios.

All profiles are callables (x, y, z) -> n, compatible with solver.PE3D's
n_profile argument.  The modified refractivity M is converted to n via the
single-source convention in solver/medium.py:
    n = 1 + [ M - (z / R_EARTH) * 1e6 ] * 1e-6.
"""
from __future__ import annotations
import numpy as np

from solver.medium import n_from_M


class StandardAtmosphereProfile:
    """M(z) = M0 + slope * z  (standard sub-refraction, reference)."""
    def __init__(self, M0=301.0, slope=0.118):
        self.M0 = float(M0); self.slope = float(slope)

    def M(self, z):
        return self.M0 + self.slope * np.asarray(z, dtype=float)

    def __call__(self, x, y, z):
        return n_from_M(self.M(z), z)


class EvaporationDuct:
    """Logarithmic evaporation duct over the sea surface.

    M(z) = M0 + 0.125 * ( z - delta * ln((z + z0) / z0) )
    delta ~ 15 m is the duct height; near the surface the M gradient is negative
    (trapping / sub-refraction), producing a surface-based waveguide.
    """
    def __init__(self, M0=301.0, delta=15.0, z0=0.1):
        self.M0 = float(M0); self.delta = float(delta); self.z0 = float(z0)

    def M(self, z):
        z = np.asarray(z, dtype=float)
        return self.M0 + 0.125 * (z - self.delta * np.log((z + self.z0) / self.z0))

    def __call__(self, x, y, z):
        return n_from_M(self.M(z), z)


class SurfaceDuct:
    """Tri-linear M profile with a surface-based trapping layer (50-100 m).

    M rises at the standard rate up to z=50 m, then a negative-M-gradient
    trapping layer from 50-100 m, then resumes the standard gradient above.
    """
    def __init__(self, M0=301.0):
        # break points (z, M)
        self.bz = np.array([0.0, 50.0, 100.0, 256.0])
        std = 0.118
        self.bM = np.array([M0, M0 + std * 50.0, M0 + std * 50.0 - 10.0,
                            M0 + std * 50.0 - 10.0 + std * (256.0 - 100.0)])

    def M(self, z):
        return np.interp(np.asarray(z, dtype=float), self.bz, self.bM)

    def __call__(self, x, y, z):
        return n_from_M(self.M(z), z)


class ElevatedDuct:
    """Tri-linear M profile with an elevated trapping layer (~100-120 m).

    M rises at the standard rate to z=80 m, then a negative-M-gradient trapping
    layer centered near 110 m, then resumes the standard gradient above.
    """
    def __init__(self, M0=301.0):
        std = 0.118
        self.bz = np.array([0.0, 80.0, 120.0, 256.0])
        self.bM = np.array([M0, M0 + std * 80.0, M0 + std * 80.0 - 12.0,
                            M0 + std * 80.0 - 12.0 + std * (256.0 - 120.0)])

    def M(self, z):
        return np.interp(np.asarray(z, dtype=float), self.bz, self.bM)

    def __call__(self, x, y, z):
        return n_from_M(self.M(z), z)


__all__ = ["StandardAtmosphereProfile", "EvaporationDuct", "SurfaceDuct",
           "ElevatedDuct"]
