"""Wide-angle diffraction operators for the 3D PE.

The Helmholtz factorization gives the outgoing branch
    u_x = i k0 ( sqrt(1 + Q/k0^2) - 1 ) u ,   Q = grad_perp^2 + k0^2 (n^2 - 1).
In spectral space grad_perp^2 -> -k_perp^2, so (after the refractivity half-step)
the diffraction multiplier is  exp[i k0 ( R(q) - 1 ) dx]  with  q = k_perp^2/k0^2 >= 0.

The reference function is sqrt(1 - q) (exact spectral operator).  The Padé
approximations are originally derived for sqrt(1+x) (sympy-verified) and
evaluated at x = -q:

    strict Padé[2/2] of sqrt(1+x):  R22(x) = (5x^2 + 20x + 16)/(x^2 + 12x + 16),  O(x^5)
    strict Padé[2/1] of sqrt(1+x):  R21(x) = (x^2 + 8x + 8)/(4 (x + 2)),          O(x^4)

Angle relation: q = sin^2(theta); q = 0.5 <=> theta = 45 deg.
The previously-used (1+3q/4+q^2/8)/(1+q/4) is NOT a valid [2/2] (2.06% error at q=0.5)
and is deliberately excluded.
"""
from __future__ import annotations
import numpy as np


def exact_sqrt(q):
    """Exact spectral operator sqrt(1 - q).  For q > 1 (evanescent spectrum)
    the principal branch gives +i sqrt(q-1), so those components decay by
    exp(-k0 sqrt(q-1) dx) per step."""
    q = np.asarray(q, dtype=float)
    return np.sqrt(1.0 - q + 0j)


def pade22(q):
    """Strict Padé[2/2] of sqrt(1+.) evaluated at x = -q: (16 - 20q + 5q^2)/(16 - 12q + q^2)."""
    q = np.asarray(q, dtype=float)
    num = 16.0 - 20.0 * q + 5.0 * q * q
    den = 16.0 - 12.0 * q + q * q
    return num / den


def pade21(q):
    """Strict Padé[2/1] of sqrt(1+.) evaluated at x = -q: (8 - 8q + q^2)/(4 (2 - q))."""
    q = np.asarray(q, dtype=float)
    num = 8.0 - 8.0 * q + q * q
    den = 4.0 * (2.0 - q)
    return num / den


def narrow_angle(q):
    """Narrow-angle (parabolic) operator: sqrt(1-q) ~ 1 - q/2 (SSFT legacy kernel)."""
    return 1.0 - 0.5 * np.asarray(q, dtype=float)


OPERATORS = {
    "exact": exact_sqrt,
    "pade22": pade22,
    "pade21": pade21,
    "narrow": narrow_angle,
}


def diffraction_kernel(q, k0, dx, operator="pade22"):
    """Spectral diffraction multiplier exp[i k0 (R(q) - 1) dx]."""
    R = OPERATORS[operator](q)
    return np.exp(1j * k0 * (R - 1.0) * dx)


def operator_error_table(q_grid=None):
    """Diagnostic: relative operator error |R - sqrt(1-q)|/sqrt(1-q) per operator."""
    if q_grid is None:
        q_grid = np.linspace(0.0, 1.0, 2001)
    ref = exact_sqrt(q_grid)
    out = {"q": q_grid, "theta_deg": np.degrees(np.arcsin(np.sqrt(np.clip(q_grid, 0, 1))))}
    for name in ("pade22", "pade21", "narrow"):
        out[name] = np.abs(OPERATORS[name](q_grid) - ref) / np.abs(ref)
    return out


__all__ = ["exact_sqrt", "pade22", "pade21", "narrow_angle", "diffraction_kernel",
           "operator_error_table", "OPERATORS"]
