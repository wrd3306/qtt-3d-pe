"""Baseline 4: Kuttler-Dockery classic narrow-angle SSFT parabolic equation.

This is the domain-traditional reference (Kuttler & Dockery, traditional lens /
forward-propagation PE).  It is a *solver*, not a data-driven method: it needs no
training field, and every query costs a full narrow-angle SSFT propagation from
the same physical aperture.  Implemented on top of the trusted solver with the
narrow-angle operator R(q) = 1 - q/2 (the classical narrow-angle kernel), the
same PEC image-method ground, standard atmosphere and physical Gaussian aperture.

Per the task it is organised in the Kuttler-Dockery tradition and clearly noted:
it uses the narrow-angle operator (1 - q/2), NOT the wide-angle sqrt(1-q) /
Pade[2/2] reference used for the trusted fields.  On the coarse dz=2 m grid the
effective angle band is narrow (1 GHz 4.30 deg, 3 GHz 1.43 deg), so the
narrow-angle bias is small but real and is reported honestly.

Interface: fit() records configuration (train_complex is unused); predict() runs
the narrow PE at the query frequency; storage_complex() == 0 (no learned data).
"""
from __future__ import annotations
import numpy as np

from .common import BASELINE_CONFIG, build_field


class KuttlerDockery:
    name = "kuttler_dockery"

    def __init__(self):
        self._shape = None

    def fit(self, train_complex, params):
        """Record the grid shape; no training data is used (solver baseline)."""
        U = np.asarray(train_complex, dtype=np.complex128)
        self._shape = U.shape[1:]
        return self

    def predict(self, query):
        """query: {'frequency_hz': float}.  Runs the narrow-angle SSFT PE."""
        f_GHz = float(query["frequency_hz"]) / 1.0e9
        u, _ = build_field(f_GHz, operator=BASELINE_CONFIG["operator_narrow"])
        return u.astype(np.complex128)

    def storage_complex(self):
        # no learned data: zero complex scalars retained.
        return 0
