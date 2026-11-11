"""Phase-aware out-of-grid interpolation.

The complex multilinear interpolation in :class:`~qtt.model.QTTModel` fails off
the training grid because the fast ground-interference phase is under-sampled.
This method separates the amplitude and the phase, unwraps the phase along the
frequency axis (so it is a smooth, branch-free function of frequency at every
pixel), and interpolates the two smooth quantities independently before
recombining ``a * exp(1j * p)``.

This recovers the amplitude (path loss) to a small dB error even though the
absolute phase is still wrong, which is the quantity propagation engineering
needs. The class follows the same ``fit / predict / storage_complex / name``
interface as :class:`~qtt.model.QTTModel`.
"""
from __future__ import annotations
import numpy as np

FREQ_KEY = "frequency_hz"


class PhaseAwareInterp:
    """Magnitude + frequency-unwrapped-phase interpolation on a dense training
    grid of complex PE fields.

    train_complex : ndarray (Nf, Ny, Nz) complex128.
    params        : dict containing a 1-D ``frequency_hz`` grid of length Nf.
    """

    def __init__(self, name: str = "phase_aware_interp"):
        self.name = str(name)
        self._freq = None
        self._amp = None
        self._phase = None
        self.Ny = self.Nz = None
        self._storage = None

    def fit(self, train_complex: np.ndarray, params: dict):
        U = np.asarray(train_complex, dtype=np.complex128)
        if FREQ_KEY not in params:
            raise KeyError("params must contain key '%s'" % FREQ_KEY)
        if U.ndim != 3:
            raise ValueError("phase-aware interp expects a (Nf,Ny,Nz) tensor")
        freq = np.asarray(params[FREQ_KEY], dtype=np.float64)
        if freq.size != U.shape[0]:
            raise ValueError("frequency grid length %d != leading axis %d"
                             % (freq.size, U.shape[0]))

        self._freq = freq
        self.Ny, self.Nz = U.shape[1], U.shape[2]
        # amplitude and branch-free phase along the frequency axis (axis 0)
        self._amp = np.abs(U)
        self._phase = np.unwrap(np.angle(U), axis=0)
        # complex-equivalent storage: two real arrays (amp, phase) occupy the
        # same bytes as one complex array of the same shape
        self._storage = int(U.size)
        return self

    def _bracket(self, value):
        g = self._freq
        v = float(value)
        if v <= g[0]:
            return 0, 0, 1.0
        if v >= g[-1]:
            return g.size - 1, g.size - 1, 1.0
        i1 = int(np.searchsorted(g, v))
        i0 = i1 - 1
        w = (v - g[i0]) / (g[i1] - g[i0])
        return i0, i1, float(w)

    def predict(self, query: dict) -> np.ndarray:
        """Return the complex (Ny, Nz) field at the queried off-grid frequency
        by linear interpolation of amplitude and unwrapped phase."""
        if self._amp is None:
            raise RuntimeError("call fit() first")
        i0, i1, w = self._bracket(query[FREQ_KEY])
        a = (1.0 - w) * self._amp[i0] + w * self._amp[i1]
        p = (1.0 - w) * self._phase[i0] + w * self._phase[i1]
        return (a * np.exp(1j * p)).astype(np.complex128)

    def storage_complex(self) -> int:
        if self._storage is None:
            raise RuntimeError("call fit() first")
        return self._storage
