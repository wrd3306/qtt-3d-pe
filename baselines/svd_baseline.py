"""Baseline 3: truncated SVD of the (Nf, Ny*Nz) complex training matrix.

The full economy SVD is computed once at fit time; the operating rank r is then
set by set_rank(r).  Prediction interpolates the r modal coefficients (real/imag)
vs frequency and reconstructs with the r leading spatial modes.  This yields the
rank-vs-test-error and rank-vs-storage Pareto curve requested by the task.
Max meaningful rank = Nf (the matrix has only Nf rows).
"""
from __future__ import annotations
import numpy as np

from .direct_interp import _linear_along_axis0


class TruncatedSVD:
    name = "truncated_svd"

    def __init__(self, rank=None):
        self._f_train = None
        self._Vh = None        # (Nf, Np)
        self._C = None         # (Nf, Nf) coefficients
        self._s = None
        self._shape = None
        self.rank = rank       # default after fit: 99.99% energy (overridable)

    def fit(self, train_complex, params):
        self._f_train = np.asarray(params["frequency_hz"], dtype=float)
        U = np.asarray(train_complex, dtype=np.complex128)
        self._shape = U.shape[1:]
        M = U.reshape(U.shape[0], -1)
        Uc, s, Vh = np.linalg.svd(M, full_matrices=False)
        self._s = s
        self._Vh = Vh
        self._C = (Uc * s).copy()          # (Nf, Nf)
        if self.rank is None:
            cum = np.cumsum(s) / np.sum(s)
            self.rank = int(max(1, np.searchsorted(cum, 0.9999) + 1))
        return self

    def set_rank(self, r):
        r = int(r)
        if not 1 <= r <= self._C.shape[1]:
            raise ValueError(f"rank {r} out of range 1..{self._C.shape[1]}")
        self.rank = r
        return self

    def predict(self, query):
        fq = float(query["frequency_hz"])
        r = self.rank
        cr = _linear_along_axis0(self._f_train, self._C[:, :r].real, fq)
        ci = _linear_along_axis0(self._f_train, self._C[:, :r].imag, fq)
        Cr = (cr + 1j * ci).astype(np.complex128)
        u = Cr @ self._Vh[:r, :]
        return u.reshape(self._shape).astype(np.complex128)

    def storage_complex(self):
        if self._Vh is None:
            return 0
        Np = self._Vh.shape[1]
        Nf = self._C.shape[0]
        r = self.rank or 0
        return int(r * Np + Nf * r)

    def singular_values(self):
        return self._s.copy()
