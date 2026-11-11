"""scenarios: six propagation scenarios evaluated with all methods.

Methods (uniform interface fit/predict/storage_complex/name, key 'frequency_hz'):
  - QTTModel from qtt
  - DirectFieldInterpolation, POD_ROM, TruncatedSVD, KuttlerDockery from baselines
"""
from .profiles import (StandardAtmosphereProfile, EvaporationDuct,
                      SurfaceDuct, ElevatedDuct)
from .build import (run_field, multi_peak_terrain, SCENARIOS,
                    terrain_height_at_x)

__all__ = [
    "StandardAtmosphereProfile", "EvaporationDuct", "SurfaceDuct", "ElevatedDuct",
    "run_field", "multi_peak_terrain", "SCENARIOS", "terrain_height_at_x",
]
