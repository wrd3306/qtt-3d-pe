"""Unified QTT surrogate interface for downstream scenario coupling.

Contract (task spec item 6):
    class with
        .fit(train_complex, params)
        .predict(query) -> complex128 ndarray (Ny, Nz)
        .storage_complex() -> int
        .name : str
    parameter key 'frequency_hz'.

The dense training tensor has leading *parameter* modes followed by the two
physical spatial modes (Ny, Nz).  Every parameter mode and BOTH spatial modes
are binary-quantized and TT-compressed -- spatial modes are never left as a
dense matrix (the old-draft bug).
"""
from __future__ import annotations
import numpy as np
from .tt import quantize, dequantize, tt_svd, tt_reconstruct, storage_complex

# canonical parameter order: frequency first, then optional extras in dict order
FREQ_KEY = "frequency_hz"


class QTTModel:
    """TT/QTT surrogate over a grid of complex PE fields.

    train_complex : ndarray shape (Nf, [Np,] Ny, Nz) complex128.
    params        : dict, must contain 'frequency_hz' of length Nf; optional
                    extra 1-D parameter arrays (length Np), e.g. 'source_height_m'.
    eps           : relative TT-SVD truncation threshold.
    """

    def __init__(self, eps: float = 1e-4, name: str = "qtt"):
        self.eps = float(eps)
        self.name = str(name)
        self.cores = None
        self.rank_info = None
        self.mode_sizes = None      # physical mode sizes (leading params then Ny,Nz)
        self.param_keys = None      # ordered list of parameter keys
        self.param_grid = None      # dict key -> 1-D grid array
        self.Ny = self.Nz = None
        self._tt_norm = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_complex: np.ndarray, params: dict):
        U = np.asarray(train_complex, dtype=np.complex128)
        if FREQ_KEY not in params:
            raise KeyError("params must contain key '%s'" % FREQ_KEY)

        # order parameter keys: frequency first, then remaining extras
        keys = [FREQ_KEY] + [k for k in params.keys() if k != FREQ_KEY]
        grids = [np.asarray(params[k], dtype=float) for k in keys]
        self.param_keys = keys
        self.param_grid = {k: g for k, g in zip(keys, grids)}

        expected = tuple(g.size for g in grids) + (U.shape[-2], U.shape[-1])
        if U.shape != expected:
            raise ValueError("train_complex shape %s != expected %s (params %s)"
                             % (U.shape, expected, [g.size for g in grids]))
        self.Ny, self.Nz = U.shape[-2], U.shape[-1]
        self.mode_sizes = expected

        vtensor, bits = quantize(U, expected)
        self.cores, self.rank_info = tt_svd(vtensor, self.eps)
        self._tt_norm = self.rank_info["norm"]
        self._bits = bits
        self._quant_shape = vtensor.shape
        return self

    # ------------------------------------------------------------ storage
    def storage_complex(self) -> int:
        if self.cores is None:
            raise RuntimeError("call fit() first")
        return storage_complex(self.cores)

    def storage_real8(self) -> int:
        return 2 * self.storage_complex()

    @property
    def ranks(self):
        return list(self.rank_info["ranks"])

    # ---------------------------------------------------------- prediction
    def _grid_weights(self, key, query_val):
        """Return bracketing grid indices (i0,i1) and linear weight w for value.
        Clamps to the edge grid values outside the range."""
        g = self.param_grid[key]
        v = float(query_val)
        if v <= g[0]:
            return 0, 0, 1.0
        if v >= g[-1]:
            return g.size - 1, g.size - 1, 1.0
        i1 = int(np.searchsorted(g, v))
        i0 = i1 - 1
        w = (v - g[i0]) / (g[i1] - g[i0])
        return i0, i1, float(w)

    def predict(self, query: dict) -> np.ndarray:
        """Reconstruct the dense (Ny,Nz) complex field at the queried parameter
        values by multilinear (linear) interpolation on the parameter grid.
        """
        if self.cores is None:
            raise RuntimeError("call fit() first")
        dense = dequantize(tt_reconstruct(self.cores), self.mode_sizes)  # mode_sizes...

        # bracketing indices + weights per parameter
        sel = []
        ws = []
        for k in self.param_keys:
            i0, i1, w = self._grid_weights(k, query[k])
            sel.append((i0, i1))
            ws.append(w)

        # multilinear interpolation over parameter leading dims
        nparam = len(self.param_keys)
        out = np.zeros((self.Ny, self.Nz), dtype=np.complex128)
        import itertools
        for combo in itertools.product([0, 1], repeat=nparam):
            idx = tuple(sel[d][combo[d]] for d in range(nparam))
            w = 1.0
            for d in range(nparam):
                w *= ws[d] if combo[d] == 1 else (1.0 - ws[d])
            out = out + w * dense[idx]
        return out.astype(np.complex128)

    # -------------------------------------------------------------- misc
    def dense_complex(self) -> int:
        n = 1
        for m in self.mode_sizes:
            n *= m
        return n
