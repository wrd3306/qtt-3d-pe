"""Refractivity medium: modified refractivity M profile -> index n(x,y,z).

Convention (sympy-verified):
    N = (n - 1) * 1e6
    M = N + (z / R_EARTH) * 1e6
=>  n = 1 + [ M - (z / R_EARTH) * 1e6 ] * 1e-6

Standard atmosphere: M(z) = M0 + 0.118 * z  (units: M-units / m).
Negative M gradient => trapping/ducting layer.
"""
from __future__ import annotations
import numpy as np
from .constants import R_EARTH


def n_from_M(M, z):
    """Convert modified refractivity M (scalar or array at height z) to refractive index n."""
    N = np.asarray(M) - (np.asarray(z, dtype=float) / R_EARTH) * 1.0e6
    return 1.0 + N * 1.0e-6


def M_from_n(n, z):
    """Inverse of n_from_M (diagnostic use)."""
    N = (np.asarray(n) - 1.0) * 1.0e6
    return N + (np.asarray(z, dtype=float) / R_EARTH) * 1.0e6


class StandardAtmosphere:
    """Standard horizontal-refractivity atmosphere: M(z) = M0 + slope * z.

    slope = +0.118 M-units/m (standard, sub-refraction relative to flat-Earth).
    A negative slope models a surface duct; a layered profile models elevated ducts.
    """

    def __init__(self, M0: float = 301.0, slope_M_per_m: float = 0.118):
        self.M0 = float(M0)
        self.slope = float(slope_M_per_m)

    def M(self, z):
        return self.M0 + self.slope * np.asarray(z, dtype=float)

    def n(self, z):
        return n_from_M(self.M(z), z)

    def __call__(self, x, y, z):
        """Return n(x, y, z); profile is horizontally homogeneous by default."""
        return self.n(z)


class LayeredDuct:
    """Piecewise-linear M(z) profile; used for evaporation / surface / elevated ducts.

    segments: list of (z_break, M_break) from bottom up; M is linearly interpolated
    between break points and held constant beyond the last one.
    """

    def __init__(self, breaks_z, breaks_M):
        z = np.asarray(breaks_z, dtype=float)
        M = np.asarray(breaks_M, dtype=float)
        order = np.argsort(z)
        self._z = z[order]
        self._M = M[order]

    def M(self, z):
        return np.interp(np.asarray(z, dtype=float), self._z, self._M)

    def n(self, z):
        return n_from_M(self.M(z), z)

    def __call__(self, x, y, z):
        return self.n(z)


def n_profile_uniform(n0: float = 1.0):
    """Fully uniform medium (free space default)."""
    def _n(x, y, z):
        return np.full(np.asarray(z).shape, n0, dtype=float) if np.ndim(z) else float(n0)
    return _n


__all__ = ["n_from_M", "M_from_n", "StandardAtmosphere", "LayeredDuct", "n_profile_uniform"]
