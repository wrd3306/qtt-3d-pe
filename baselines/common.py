"""Shared configuration, trusted-field generation, masks and error statistics
for the external baselines.

Deterministic training/test complex fields are produced by the *trusted* PE3D
solver (exact spectral diffraction operator sqrt(1-q), complex128, phase kept),
using exactly the same grid / atmosphere / aperture / boundary that the QTT
compression study uses (REBUILD_SPEC.md section 2, 9, 10):

    f in [0.5, 3.0] GHz, Nf = 16 training
    Ny = 64, dy = 5 m ; Nz = 128, dz = 2 m
    PEC ground (horizontal polarization, Dirichlet via image method)
    standard atmosphere M0=301, slope +0.118 M-units/m
    source at y=0, z=30 m, physical Gaussian aperture (sigma_y=5 m, sigma_z=3 m)
    absorbing sponge on top and y sides.

Coarse-grid Nyquist effective angle band (REBUILD_SPEC section 7, must be stated,
no wide-angle claim): with dz=2 m, sin(theta_max)=lambda/(2 dz) gives
    1 GHz -> 4.30 deg ,  3 GHz -> 1.43 deg.
"""
from __future__ import annotations
import numpy as np

from solver import PE3D, StandardAtmosphere, gaussian_aperture
from solver.constants import C_LIGHT


# --------------------------------------------------------------------------- #
#  Deterministic configuration (single source of truth for baselines)
# --------------------------------------------------------------------------- #
BASELINE_CONFIG = dict(
    # grid
    Ny=64, Nz=128,
    dy_m=5.0, dz_m=2.0, dx_m=2.0,
    y_max_m=160.0,          # Ny*dy/2 = 160  (periodic y window)
    z_max_m=256.0,          # Nz*dz = 256    (PEC ground at z=0)
    # propagation
    x_end_m=2000.0,
    # ground / medium / boundary
    ground="horizontal",    # H-pol Dirichlet PEC (image method)
    operator_trusted="exact",   # sqrt(1-q) spectral operator (reference)
    operator_narrow="narrow",   # 1 - q/2  (Kuttler-Dockery traditional)
    M0=301.0, slope_M_per_m=0.118,
    sponge_top_m=40.0, sponge_y_m=16.0, sponge_strength=8.0,
    # aperture
    z0_m=30.0, sigma_y_m=5.0, sigma_z_m=3.0, tilt_deg=0.0,
    # frequency set
    f_lo_GHz=0.5, f_hi_GHz=3.0, nf_train=16,
    # held-out test frequencies (NOT on the training grid)
    f_test_GHz=(0.75, 1.25, 1.75, 2.25, 2.75),
    # error stats
    amp_floor_db=-40.0,     # dB floor relative to per-field peak for dB stats
)


def generate_frequency_set():
    """Return (f_train_GHz, f_test_GHz) as float64 arrays."""
    c = BASELINE_CONFIG
    f_train = np.linspace(c["f_lo_GHz"], c["f_hi_GHz"], c["nf_train"])
    f_test = np.asarray(c["f_test_GHz"], dtype=float)
    return f_train, f_test


def nyquist_angle_deg(f_GHz):
    """Effective narrow-angle band of the coarse dz=2 m grid (REBUILD_SPEC 7)."""
    lam = C_LIGHT / (f_GHz * 1.0e9)
    return float(np.degrees(np.arcsin(lam / (2.0 * BASELINE_CONFIG["dz_m"]))))


# --------------------------------------------------------------------------- #
#  Trusted field generation
# --------------------------------------------------------------------------- #
def build_field(f_GHz, operator=None):
    """Run the trusted PE3D to a slice at x_end_m for one frequency.

    Returns (u, prob) where u is complex128 (Ny, Nz), phase-preserved.
    Deterministic: no randomness anywhere in the propagator.
    """
    c = BASELINE_CONFIG
    if operator is None:
        operator = c["operator_trusted"]
    prob = PE3D(
        f_GHz=f_GHz, dx_m=c["dx_m"], dy_m=c["dy_m"], dz_m=c["dz_m"],
        Ny=c["Ny"], Nz=c["Nz"], y_max_m=c["y_max_m"], z_max_m=c["z_max_m"],
        ground=c["ground"], operator=operator,
        n_profile=StandardAtmosphere(M0=c["M0"], slope_M_per_m=c["slope_M_per_m"]),
        sponge_top_m=c["sponge_top_m"], sponge_y_m=c["sponge_y_m"],
        sponge_strength=c["sponge_strength"],
    )
    u0 = gaussian_aperture(prob.y_phys, prob.z_phys,
                          y0=0.0, z0=c["z0_m"],
                          sigma_y=c["sigma_y_m"], sigma_z=c["sigma_z_m"],
                          tilt_deg=c["tilt_deg"], k0=prob.k0)
    prob.initialize(u0)
    prob.run(c["x_end_m"])
    return prob.field().copy(), prob


# --------------------------------------------------------------------------- #
#  Masks and metrics
# --------------------------------------------------------------------------- #
def make_valid_mask(y, z):
    """Effective-region mask (Ny, Nz) bool.

    Excludes: the PEC ground row z=0, the top sponge band, and the y-side sponge
    bands.  Used for the complex relative-error norm.  (REBUILD_SPEC section 10:
    interference nulls and absorbing layers are excluded.)
    """
    c = BASELINE_CONFIG
    yy = y[None, :]              # (1, Ny)
    zz = z[:, None]              # (Nz, 1) -> meshgrid ij gives (Ny, Nz)
    # build (Ny, Nz) explicitly
    Y = np.broadcast_to(yy, (c["Nz"], c["Ny"])).T   # (Ny, Nz)
    Z = np.broadcast_to(zz, (c["Nz"], c["Ny"])).T   # (Ny, Nz)
    interior = (np.abs(Y) <= np.abs(y).max() - c["sponge_y_m"] + 1e-9) \
        & (Z <= z.max() - c["sponge_top_m"] + 1e-9) \
        & (Z > 0.0)
    return interior


def rel_l2_complex(u_pred, u_ref, mask):
    """Relative L2 error of the complex field over the valid mask.

    ||u_pred - u_ref||_2 / ||u_ref||_2  (complex128, phase kept).
    """
    a = np.asarray(u_pred, dtype=np.complex128)[mask]
    b = np.asarray(u_ref, dtype=np.complex128)[mask]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def db_error_stats(u_pred, u_ref, mask, amp_floor_db=None):
    """Signed dB error 20 log10(|u_pred|/|u_ref|) over the dB-valid region.

    dB-valid = complex valid mask AND |u_ref| >= floor * max|u_ref|, where
    floor = 10^(amp_floor_db/20).  This excludes interference nulls where a dB
    ratio is meaningless.  Returns RMS, p50, p90, p95 (signed dB error), and
    the count of pixels used.
    """
    c = BASELINE_CONFIG
    if amp_floor_db is None:
        amp_floor_db = c["amp_floor_db"]
    floor = 10.0 ** (amp_floor_db / 20.0)

    a = np.abs(np.asarray(u_pred, dtype=np.complex128))
    b = np.abs(np.asarray(u_ref, dtype=np.complex128))
    peak = float(b[mask].max())
    db_mask = mask & (b >= floor * peak)

    db_err = 20.0 * np.log10((a[db_mask] + 1e-30) / (b[db_mask] + 1e-30))
    db_err = np.sort(db_err)
    n = db_err.size
    return dict(
        rms_db=float(np.sqrt(np.mean(db_err ** 2))),
        mean_db=float(np.mean(db_err)),
        p50_db=float(np.percentile(db_err, 50)),
        p90_db=float(np.percentile(db_err, 90)),
        p95_db=float(np.percentile(db_err, 95)),
        n_pixels=int(n),
        peak_abs=peak,
    )
