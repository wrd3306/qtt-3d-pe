"""Canonical grid constants, valid-region mask, error statistics, and trusted
direct-PE field generation for the QTT study.

This is the single source of truth shared by scripts/qtt_run.py and
scripts/phase_aware_run.py so that both report exactly the same mask, amplitude
floor, and error accounting. No numbers are hard-coded; the direct field is
produced by the deterministic trusted solver.
"""
from __future__ import annotations
import numpy as np

from solver.propagator import PE3D
from solver.medium import StandardAtmosphere
from solver.antenna import gaussian_aperture

# ---- grid constants (must match qtt_dataset.py) ----
DX, DY, DZ = 2.0, 5.0, 2.0
NY, NZ = 64, 128
Y_MAX, Z_MAX = 160.0, 256.0
X_END = 4000.0
SPONGE_TOP, SPONGE_Y, SPONGE_STR = 40.0, 16.0, 8.0
SIGMA_Y, SIGMA_Z = 20.0, 5.0
N_F = 16
SOURCE_Z0 = 30.0


def valid_mask(y, z):
    """Effective-region mask (Ny, Nz): excludes the PEC ground row, the top
    sponge band, and the y-side sponge bands."""
    y_sp = Y_MAX - SPONGE_Y
    z_top = Z_MAX - SPONGE_TOP
    m = (np.abs(y) < y_sp)[:, None] & (z[None, :] > 5.0) & (z[None, :] < z_top)
    return m


def field_errors(U_ref, U_rec, m, floor_rel=1e-3):
    """Complex rel-L2 over the valid region plus amplitude dB statistics above
    the per-field floor. Returns a dict."""
    vm = np.broadcast_to(m[None, :, :], U_ref.shape)
    rel = float(np.linalg.norm((U_rec - U_ref)[vm]) / np.linalg.norm(U_ref[vm]))
    amp = np.abs(U_ref)
    floor = amp.max() * floor_rel
    both = (amp > floor) & (np.abs(U_rec) > floor) & vm
    db = 20.0 * np.log10(np.abs(U_rec[both]) / np.abs(U_ref[both]))
    return dict(rel_l2=rel,
                db_rms=float(np.sqrt(np.mean(db ** 2))),
                db_p50=float(np.percentile(np.abs(db), 50)),
                db_p90=float(np.percentile(np.abs(db), 90)),
                db_p95=float(np.percentile(np.abs(db), 95)),
                db_max=float(np.max(np.abs(db))),
                n_points_valid=int(vm.sum()),
                n_points_db=int(both.sum()))


def direct_pe_field(f_GHz, z0=SOURCE_Z0):
    """Run the trusted deterministic PE3D (Pade[2/2], PEC ground) to the terminal
    slice and return the complex (Ny, Nz) field. Ground truth for held-out work."""
    prob = PE3D(f_GHz=f_GHz, dx_m=DX, dy_m=DY, dz_m=DZ, Ny=NY, Nz=NZ,
                y_max_m=Y_MAX, z_max_m=Z_MAX, ground="horizontal",
                operator="pade22",
                n_profile=StandardAtmosphere(M0=301.0, slope_M_per_m=0.118),
                sponge_top_m=SPONGE_TOP, sponge_y_m=SPONGE_Y,
                sponge_strength=SPONGE_STR)
    u0 = gaussian_aperture(prob.y_phys, prob.z_phys, y0=0.0, z0=z0,
                           sigma_y=SIGMA_Y, sigma_z=SIGMA_Z, k0=prob.k0,
                           tilt_deg=0.0)
    prob.initialize(u0)
    prob.run(x_end_m=X_END)
    return prob.field().copy()
