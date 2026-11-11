"""Low-level TT (tensor-train) machinery on COMPLEX tensors with honest storage
accounting.

This module deliberately implements the *real* QTT: every physical mode
(frequency, extra parameter, cross-range y, height z) is binary-quantized into
virtual size-2 factors, and the whole 17-/19-way virtual tensor is TT-SVD
compressed.  Spatial modes ARE quantized and truncated -- the old bug of
compressing only a 16 x r coefficient block while leaving the spatial matrix
untouched cannot recur here, because every virtual factor is a TT core that gets
rank-truncated.

Conventions (REBUILD_SPEC.md sec.9):
  * complex128 values are phase-resolving; a complex number counts as ONE
    complex scalar = TWO real float64.  Both counters are reported; the ratio
    CR comes out identical either way because the factor 2 appears on both sides.
  * Storage = sum over ALL cores of prod(core.shape).  No mixed numerator/denominator.
  * No preset ranks: the rank is whatever the complex SVD truncation yields.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------
# binary quantization
# --------------------------------------------------------------------------
def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def quantize_shape(mode_sizes):
    """Given a list of physical mode sizes (each must be a power of two), return
    the virtual size-2 shape and the bit-length of each mode.

    Example: (16, 64, 128) -> (2,)*17, [4, 6, 7]
    """
    bits = []
    for n in mode_sizes:
        n = int(n)
        if not _is_pow2(n):
            raise ValueError(f"mode size {n} is not a power of two; QTT needs pow-2 modes")
        bits.append(n.bit_length() - 1)
    return (2,) * sum(bits), bits


def quantize(tensor, mode_sizes):
    """Reshape a dense tensor (mode_sizes...) into its binary-quantized virtual
    (2,2,...,2) form.  C-order reshape: the fastest-varying mode's least
    significant bit is the last virtual factor (standard QTT matricization)."""
    tensor = np.asarray(tensor)
    if tuple(tensor.shape) != tuple(mode_sizes):
        raise ValueError(f"shape {tensor.shape} != requested modes {mode_sizes}")
    vshape, bits = quantize_shape(mode_sizes)
    return tensor.reshape(vshape), bits


def dequantize(vtensor, mode_sizes):
    """Inverse of quantize: virtual (2,...,2) back to physical mode_sizes."""
    return np.asarray(vtensor).reshape(mode_sizes)


# --------------------------------------------------------------------------
# TT-SVD (Oseledets) with complex numpy SVD and relative truncation
# --------------------------------------------------------------------------
def tt_svd(tensor: np.ndarray, eps: float):
    """TT-SVD decomposition.

    Parameters
    ----------
    tensor : complex ndarray, any shape whose total size we decompose factor by
             factor along its existing axes.  (We call this on the quantized
             (2,2,...,2) tensor, but it works for any shape.)
    eps    : relative truncation threshold.  The per-cut Frobenius-norm budget is
             delta = eps * ||tensor||_F / sqrt(d-1) (Oseledets 2011).  A cut keeps
             singular values whose discarded tail energy sum(s_disc^2) <= delta^2.

    Returns
    -------
    cores : list of ndarray, core k has shape (r_{k-1}, n_k, r_k), r_0 = r_d = 1.
    info  : dict with ranks, singular-value tails, measured truncation error.
    """
    T = np.asarray(tensor, dtype=np.complex128)
    d = T.ndim
    n = T.shape
    normT = float(np.linalg.norm(T))
    if d == 1:
        # degenerate: single core (1, n, 1)
        return [T.reshape(1, n[0], 1)], {"ranks": [1, 1], "norm": normT}

    delta = eps / np.sqrt(d - 1) * normT
    cores = []
    ranks = []
    C = T
    r_prev = 1
    tails = []
    for k in range(d - 1):
        C = C.reshape(r_prev * n[k], -1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        # tail energy if we keep only S[:j]: sum_{l>=j} S[l]^2
        S2 = S.astype(np.float64) ** 2
        # tail_from[j] = sum_{l>=j} S[l]^2, for j = 0..len(S) (length len(S)+1)
        tail_from = np.append(np.cumsum(S2[::-1])[::-1], 0.0)
        # choose smallest j in [0, len(S)] with tail_from[j] <= delta^2
        ok = np.where(tail_from <= delta ** 2)[0]
        r_new = int(ok[0]) if ok.size else len(S)
        r_new = max(1, r_new)
        tails.append(float(tail_from[r_new]))
        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]
        cores.append(U.reshape(r_prev, n[k], r_new).copy())
        ranks.append(r_new)
        C = (S[:, None] * Vt)  # (r_new, rest)
        r_prev = r_new
    cores.append(C.reshape(r_prev, n[d - 1], 1).copy())
    ranks.append(1)
    return cores, {"ranks": ranks, "norm": normT, "tail_energy": tails, "delta": delta}


def tt_reconstruct(cores):
    """Contract all cores back into the dense tensor (shape = product of core dim1)."""
    res = cores[0]  # (1, n0, r1)
    for k in range(1, len(cores)):
        res = np.tensordot(res, cores[k], axes=([-1], [0]))
    # res shape (1, n0, n1, ..., n_{d-1}, 1)
    return res.reshape(tuple(c.shape[1] for c in cores))


# --------------------------------------------------------------------------
# honest storage accounting
# --------------------------------------------------------------------------
def core_complex_scalars(core) -> int:
    return int(np.prod(core.shape))


def storage_complex(cores) -> int:
    """Total storage in COMPLEX scalars (each = real+imag)."""
    return sum(core_complex_scalars(c) for c in cores)


def storage_real8(cores) -> int:
    """Total storage in REAL float64 (complex128 = 2 real8)."""
    return 2 * storage_complex(cores)


def dense_complex_scalars(mode_sizes) -> int:
    n = 1
    for m in mode_sizes:
        n *= int(m)
    return n
