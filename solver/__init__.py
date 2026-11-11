"""solver: trusted 3D wide-angle PE (SSFT) library.

Public API (stable for downstream qtt / scenarios modules):
    from solver import PE3D, StandardAtmosphere, LayeredDuct, n_from_M
    from solver import GaussianHill, SlopePlane, FLAT
    from solver import gaussian_aperture, sigma_z_from_beamwidth
    from solver import pade22, pade21, exact_sqrt, diffraction_kernel

All fields are complex128 (phase-preserving); dB is display-only.
"""
from .constants import C_LIGHT, R_EARTH
from .medium import (n_from_M, M_from_n, StandardAtmosphere, LayeredDuct,
                     n_profile_uniform)
from .operators import (exact_sqrt, pade22, pade21, narrow_angle,
                        diffraction_kernel, operator_error_table, OPERATORS)
from .terrain import Terrain, GaussianHill, SlopePlane, FLAT
from .antenna import gaussian_aperture, sigma_z_from_beamwidth
from .propagator import PE3D

__version__ = "1.0.0"
__all__ = [
    "C_LIGHT", "R_EARTH",
    "n_from_M", "M_from_n", "StandardAtmosphere", "LayeredDuct", "n_profile_uniform",
    "exact_sqrt", "pade22", "pade21", "narrow_angle", "diffraction_kernel",
    "operator_error_table", "OPERATORS",
    "Terrain", "GaussianHill", "SlopePlane", "FLAT",
    "gaussian_aperture", "sigma_z_from_beamwidth",
    "PE3D",
]
