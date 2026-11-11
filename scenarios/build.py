"""Generalized trusted-field builder and the six scenario definitions.

Reuses the grid / aperture / sponge / valid-mask / metric machinery from
baselines/common.py, but allows each scenario to supply its own M profile
(n_profile callable) and terrain (Terrain object).  The PEC ground, absorbing
sponge, and exact-spectral reference operator are unchanged.
"""
from __future__ import annotations
import numpy as np

from solver import PE3D, gaussian_aperture
from solver.terrain import FLAT, GaussianHill, Terrain
from baselines.common import BASELINE_CONFIG as C

from .profiles import (StandardAtmosphereProfile, EvaporationDuct,
                       SurfaceDuct, ElevatedDuct)


# --------------------------------------------------------------------------- #
#  generalized trusted field builder
# --------------------------------------------------------------------------- #
def run_field(f_GHz, n_profile=None, terrain=None, operator=None):
    """Run the trusted PE3D one slice at x_end_m.

    Returns (u complex128 (Ny,Nz), prob).  Deterministic.
    """
    if operator is None:
        operator = C["operator_trusted"]
    if n_profile is None:
        n_profile = StandardAtmosphereProfile(M0=C["M0"], slope=C["slope_M_per_m"])
    if terrain is None:
        terrain = FLAT
    prob = PE3D(
        f_GHz=f_GHz, dx_m=C["dx_m"], dy_m=C["dy_m"], dz_m=C["dz_m"],
        Ny=C["Ny"], Nz=C["Nz"], y_max_m=C["y_max_m"], z_max_m=C["z_max_m"],
        ground=C["ground"], operator=operator,
        n_profile=n_profile, terrain=terrain,
        sponge_top_m=C["sponge_top_m"], sponge_y_m=C["sponge_y_m"],
        sponge_strength=C["sponge_strength"],
    )
    u0 = gaussian_aperture(prob.y_phys, prob.z_phys,
                          y0=0.0, z0=C["z0_m"],
                          sigma_y=C["sigma_y_m"], sigma_z=C["sigma_z_m"],
                          tilt_deg=C["tilt_deg"], k0=prob.k0)
    prob.initialize(u0)
    prob.run(C["x_end_m"])
    return prob.field().copy(), prob


# --------------------------------------------------------------------------- #
#  terrain helpers
# --------------------------------------------------------------------------- #
def multi_peak_terrain():
    """Two ridges at x=700 m (60 m) and x=1400 m (40 m), y-invariant."""
    def h(x, y):
        h1 = 60.0 * np.exp(-((x - 700.0) / 300.0) ** 2)
        h2 = 40.0 * np.exp(-((x - 1400.0) / 200.0) ** 2)
        return h1 + h2
    return Terrain(height_func=h, dx_slope=1.0)


# --------------------------------------------------------------------------- #
#  six scenario definitions
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "free_space": dict(
        n_profile=StandardAtmosphereProfile(M0=301.0, slope=0.118),
        terrain=FLAT,
        desc="standard atmosphere, flat ground (reference)",
    ),
    "evaporation_duct": dict(
        n_profile=EvaporationDuct(M0=301.0, delta=15.0, z0=0.1),
        terrain=FLAT,
        desc="logarithmic evaporation duct, trapping layer near sea surface",
    ),
    "gaussian_terrain": dict(
        n_profile=StandardAtmosphereProfile(M0=301.0, slope=0.118),
        terrain=GaussianHill(H=60.0, x0=1000.0, sigma_x=400.0),
        desc="Gaussian hill h=60 m at x=1000 m, terrain-following",
    ),
    "multi_peak": dict(
        n_profile=StandardAtmosphereProfile(M0=301.0, slope=0.118),
        terrain=multi_peak_terrain(),
        desc="two ridges (60 m @700 m, 40 m @1400 m), terrain-following",
    ),
    "surface_duct": dict(
        n_profile=SurfaceDuct(M0=301.0),
        terrain=FLAT,
        desc="tri-linear M, surface trapping layer 50-100 m",
    ),
    "elevated_duct": dict(
        n_profile=ElevatedDuct(M0=301.0),
        terrain=FLAT,
        desc="tri-linear M, elevated trapping layer ~100-120 m",
    ),
}


def terrain_height_at_x(scenario, x):
    """Terrain height (m) at range x for a scenario (y-invariant)."""
    return SCENARIOS[scenario]["terrain"].height(x, 0.0)


__all__ = ["run_field", "multi_peak_terrain", "SCENARIOS", "terrain_height_at_x"]
