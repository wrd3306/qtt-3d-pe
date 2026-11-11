"""Baseline 1: direct frequency interpolation of the raw complex field.

No low-rank model at all.  For every spatial point we interpolate the complex
field as a function of frequency between the training frequencies, using two
independent encodings:

  (A) real / imaginary parts interpolated separately (linear).
  (B) magnitude / unwrapped phase interpolated separately (linear).

At fit time a leave-one-out residual on the training frequencies selects the
more stable variant (no peek at test data); the chosen variant is recorded and
used for prediction.  Complex128 throughout.
"""
from __future__ import annotations
import numpy as np


def _linear_along_axis0(x, y, xq):
    """Linear interpolation of y (N, P) at scalar xq along axis 0. Complex-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    j = int(np.searchsorted(x, xq) - 1)
    j = max(0, min(j, len(x) - 2))
    t = (xq - x[j]) / (x[j + 1] - x[j])
    return (1.0 - t) * y[j] + t * y[j + 1]


class DirectFieldInterpolation:
    name = "direct_interpolation"

    def __init__(self, variant="auto"):
        self.variant = variant          # 'auto' | 'real_imag' | 'mag_phase'
        self._f_train = None            # (Nf,) Hz
        self._U = None                  # (Nf, Ny*Nz) complex128
        self._shape = None
        self.chosen_variant = None
        self.loo_residual = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train_complex, params):
        """train_complex: (Nf, Ny, Nz) complex128.  params: {'frequency_hz': (Nf,)}.
        """
        self._f_train = np.asarray(params["frequency_hz"], dtype=float)
        U = np.asarray(train_complex, dtype=np.complex128)
        self._shape = U.shape[1:]
        self._U = U.reshape(U.shape[0], -1).copy()

        if self.variant == "auto":
            r_ri = self._loo_residual("real_imag")
            r_mp = self._loo_residual("mag_phase")
            self.loo_residual = {"real_imag": r_ri, "mag_phase": r_mp}
            self.chosen_variant = "real_imag" if r_ri <= r_mp else "mag_phase"
        else:
            self.chosen_variant = self.variant
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, query, variant=None):
        """query: {'frequency_hz': float}.  Returns (Ny, Nz) complex128."""
        fq = float(query["frequency_hz"])
        v = variant or self.chosen_variant
        if v == "real_imag":
            u = self._interp_real_imag(fq)
        else:
            u = self._interp_mag_phase(fq)
        return u.reshape(self._shape).astype(np.complex128)

    # ------------------------------------------------------- internal helpers
    def _interp_real_imag(self, fq):
        re = _linear_along_axis0(self._f_train, self._U.real, fq)
        im = _linear_along_axis0(self._f_train, self._U.imag, fq)
        return (re + 1j * im).astype(np.complex128)

    def _interp_mag_phase(self, fq):
        mag = np.abs(self._U)
        phase = np.unwrap(np.angle(self._U), axis=0)
        mag_i = _linear_along_axis0(self._f_train, mag, fq)
        ph_i = _linear_along_axis0(self._f_train, phase, fq)
        return (mag_i * np.exp(1j * ph_i)).astype(np.complex128)

    def _loo_residual(self, variant):
        """Leave-one-out relative L2 residual on training frequencies."""
        errs = []
        for i in range(self._U.shape[0]):
            mask = np.ones(self._U.shape[0], dtype=bool)
            mask[i] = False
            f_tr = self._f_train[mask]
            U_tr = self._U[mask]
            if variant == "real_imag":
                re = _linear_along_axis0(f_tr, U_tr.real, self._f_train[i])
                im = _linear_along_axis0(f_tr, U_tr.imag, self._f_train[i])
                pred = re + 1j * im
            else:
                mag = np.abs(U_tr)
                ph = np.unwrap(np.angle(U_tr), axis=0)
                pred = (_linear_along_axis0(f_tr, mag, self._f_train[i])
                        * np.exp(1j * _linear_along_axis0(f_tr, ph, self._f_train[i])))
            true = self._U[i]
            errs.append(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-30))
        return float(np.mean(errs))

    # ------------------------------------------------------------------ cost
    def storage_complex(self):
        # stores the full training complex field
        return int(self._U.size) if self._U is not None else 0
