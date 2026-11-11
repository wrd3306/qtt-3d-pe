"""Physical constants (SI). Single source of truth per REBUILD_SPEC.md §2."""
import numpy as np

# Exact vacuum speed of light (m/s), 1983 definition.
C_LIGHT = 2.99792458e8

# Mean Earth radius (m), used in modified-refractivity M -> n conversion.
R_EARTH = 6.371e6

# Small machine floor for amplitude/dB displays only (never used to fabricate data).
AMP_FLOOR = 1e-12

__all__ = ["C_LIGHT", "R_EARTH", "AMP_FLOOR"]
