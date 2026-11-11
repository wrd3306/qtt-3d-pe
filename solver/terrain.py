"""Terrain-following coordinate mapping (NOT a staircase hard mask).

Physical coordinates (x, y, z); computational vertical coordinate
    zeta = z - h(x, y)          (height above terrain)
so the PEC ground (z = h(x,y)) always sits at zeta = 0.

Under this mapping the PE acquires a first-order tilt correction in the
refraction operator (Dockery-style effective refractivity):
    n_eff^2(zeta) = n^2(zeta) - 2 * zeta * dh/dx
which enters the refractivity half-step as exp[i k0/2 (n_eff^2 - 1) dx].
For flat terrain (dh/dx = 0) this reduces exactly to the flat-ground PE.
"""
from __future__ import annotations
import numpy as np


class Terrain:
    def __init__(self, height_func=None, dx_slope=1.0):
        """height_func(x, y) -> terrain height h(x,y) in meters; None = flat plane z=0."""
        self._h = height_func
        self._dx_slope = float(dx_slope)

    def height(self, x, y):
        if self._h is None:
            return 0.0
        return float(self._h(x, y))

    def slope_x(self, x, y):
        """Finite-difference dh/dx used in the terrain-following refractivity correction."""
        if self._h is None:
            return 0.0
        hx = (self._h(x + self._dx_slope, y) - self._h(x - self._dx_slope, y)) / (2.0 * self._dx_slope)
        return float(hx)

    def is_flat(self):
        return self._h is None


class GaussianHill(Terrain):
    """h(x, y) = H * exp(-((x-x0)^2/(2 sigma_x^2) + (y-y0)^2/(2 sigma_y^2)))."""

    def __init__(self, H=50.0, x0=5000.0, sigma_x=800.0, y0=0.0, sigma_y=np.inf):
        self.H = float(H); self.x0 = float(x0); self.sigma_x = float(sigma_x)
        self.y0 = float(y0); self.sigma_y = float(sigma_y)

        def _h(x, y):
            dy = 0.0 if np.isinf(self.sigma_y) else (y - self.y0) ** 2 / (2.0 * self.sigma_y ** 2)
            return self.H * np.exp(-(x - self.x0) ** 2 / (2.0 * self.sigma_x ** 2) - dy)
        super().__init__(_h, dx_slope=max(1.0, self.sigma_x / 20.0))


class SlopePlane(Terrain):
    """h(x, y) = s * x (gently tilted plane; sanity check for the tilt correction)."""

    def __init__(self, slope=0.01):
        self.slope = float(slope)
        super().__init__(lambda x, y: self.slope * x, dx_slope=1.0)


FLAT = Terrain()

__all__ = ["Terrain", "GaussianHill", "SlopePlane", "FLAT"]
