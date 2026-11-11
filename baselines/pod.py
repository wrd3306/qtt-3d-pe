"""Baseline 2: POD / projection-based ROM.

Proper Orthogonal Decomposition of the (Nf, Ny*Nz) complex training matrix via
economy SVD:  U_train = A diag(s) V^H.  The first r spatial modes V_r form the
reduced basis; the modal coefficients C = A_r diag(s_r) (Nf, r) are interpolated
linearly (real/imag) against frequency, then the field is reconstructed as
C(f) V_r.  Complex128 throughout.

Rank r is chosen at fit time by a cumulative-singular-value energy threshold
(99.99 % by default).  This is one operating point; the full rank-accuracy curve
is produced by TruncatedSVD.
"""
from __future__ import annotations
import numpy as np

from .direct_interp import _linear_along_axis0


class POD_ROM:
    name = "pod_rom"

    def __init__(self, energy=0.9999):
        self.energy = float(energy)
        self._f_train = None
        self._Vh = None        # (Nf, Np) unitary rows
        self._s = None          # (Nf,)
        self._C = None         # (Nf, r) modal coefficients
        self.rank = None
        self._shape = None

    def fit(self, train_complex, params):
        self._f_train = np.asarray(params["frequency_hz"], dtype=float)
        U = np.asarray(train_complex, dtype=np.complex128)
        self._shape = U.shape[1:]
        M = U.reshape(U.shape[0], -1)
        # economy SVD: M = Uc @ diag(s) @ Vh
        Uc, s, Vh = np.linalg.svd(M, full_matrices=False)
        self._s = s
        self._Vh = Vh                       # (Nf, Np)
        # energy threshold
        cum = np.cumsum(s) / np.sum(s)
        r = int(np.searchsorted(cum, self.energy) + 1)
        r = max(1, min(r, len(s)))
        self.rank = r
        self._C = (Uc[:, :r] * s[:r]).copy()   # (Nf, r)
        return self

    def predict(self, query):
        fq = float(query["frequency_hz"])
        cr = _linear_along_axis0(self._f_train, self._C.real, fq)
        ci = _linear_along_axis0(self._f_train, self._C.imag, fq)
        Cr = (cr + 1j * ci).astype(np.complex128)
        u = Cr @ self._Vh[:self.rank, :]
        return u.reshape(self._shape).astype(np.complex128)

    def storage_complex(self):
        if self._Vh is None:
            return 0
        Np = self._Vh.shape[1]
        Nf = self._C.shape[0]
        return int(self.rank * Np + Nf * self.rank)
