"""Physical antenna aperture initialization: Gaussian aperture with specified
vertical half-power beamwidth and beam tilt.

A Gaussian aperture u0(zeta) = exp(-zeta^2/(2 sigma_z^2)) produces, in the
far field, a Gaussian beam whose half-power (amplitude -3 dB) half-angle is
    theta_3dB = sqrt(ln 2) * lambda / (2 pi sigma_z).
"""
from __future__ import annotations
import numpy as np
from .constants import C_LIGHT


def sigma_z_from_beamwidth(theta_3dB_deg, f_GHz):
    """Vertical 1/e aperture half-width sigma_z [m] for a given HPBW half-angle."""
    lam = C_LIGHT / (f_GHz * 1.0e9)
    return np.sqrt(np.log(2.0)) * lam / (2.0 * np.pi * np.radians(theta_3dB_deg))


def gaussian_aperture(y, zeta, y0=0.0, z0=30.0, sigma_y=5.0, sigma_z=5.0,
                      tilt_deg=0.0, k0=1.0):
    """Complex128 aperture field on the (y, zeta) plane at x = 0.

    tilt_deg: vertical beam-pointing angle off the horizontal (upward positive),
    implemented as the aperture phase slope exp(i k0 zeta sin(tilt)).
    """
    Y, Z = np.meshgrid(np.asarray(y, dtype=float), np.asarray(zeta, dtype=float),
                       indexing="ij")
    amp = np.exp(-((Y - y0) ** 2) / (2.0 * sigma_y ** 2)
                 - ((Z - z0) ** 2) / (2.0 * sigma_z ** 2))
    phase = np.exp(1j * k0 * (Z - z0) * np.sin(np.radians(tilt_deg)))
    return (amp * phase).astype(np.complex128)


__all__ = ["sigma_z_from_beamwidth", "gaussian_aperture"]
