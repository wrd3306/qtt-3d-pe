"""External baselines for the 3D-QTT-PE compression study.

Four independent, trusted external reference methods (all on complex128 fields,
phase-preserving; dB is display-only):

  1. DirectFieldInterpolation  -- frequency-wise interpolation of the raw complex
                                  field (real/imag vs magnitude/phase; the more
                                  stable variant is recorded).
  2. POD_ROM                   -- SVD proper-orthogonal decomposition of the
                                  training complex matrix; interpolate the modal
                                  coefficients vs frequency, then reconstruct.
  3. TruncatedSVD              -- truncated SVD of the (Nf, Ny*Nz) complex matrix,
                                  rank scanned; rank-accuracy / storage curve.
  4. KuttlerDockery            -- classic narrow-angle SSFT PE (Kuttler-Dockery
                                  tradition) as a domain-traditional solver
                                  baseline: no training data, full per-query solve.

Uniform interface (consistent with the qtt module):
    fit(train_complex, params)            train_complex: (Nf, Ny, Nz) complex128
    predict(query) -> complex128          query: dict with key 'frequency_hz'
    storage_complex() -> int              number of complex128 scalars retained
    .name                                 method name
"""
from .common import (BASELINE_CONFIG, build_field, make_valid_mask, db_error_stats,
                     rel_l2_complex, generate_frequency_set)
from .direct_interp import DirectFieldInterpolation
from .pod import POD_ROM
from .svd_baseline import TruncatedSVD
from .kuttler_dockery import KuttlerDockery

__all__ = [
    "BASELINE_CONFIG", "build_field", "make_valid_mask", "db_error_stats",
    "rel_l2_complex", "generate_frequency_set",
    "DirectFieldInterpolation", "POD_ROM", "TruncatedSVD", "KuttlerDockery",
]
