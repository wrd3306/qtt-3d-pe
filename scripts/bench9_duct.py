"""Bench 9 (P0-07): surface duct with an INDEPENDENT Airy-function reference.

We deliberately choose a refractivity profile whose Helmholtz/PE normal modes
have a closed special-function form: linear n^2.
    n^2(z) = 1 + alpha * z,   alpha < 0   (n^2 decreases with height -> trapping)

Reference (modal, independent of the SSFT propagator):
The narrow-angle envelope PE  u_x = i/(2k0) u_zz + i k0/2 (n^2-1) u  with PEC
Dirichlet at z=0 has normal modes
    psi_m(z) = Ai(c z + a_m),   c = (k0^2 |alpha|)^(1/3),
where a_m are the negative zeros of Ai.  With u = exp(i k0 gamma_m x) psi_m,
    gamma_m = a_m |alpha| / (2 c)  (<0, trapped slower than free space).
Project the launching Gaussian onto these modes and propagate each phase.

The corrected PE3D refraction half-step is exp(i k0 (n^2-1) dx/4); two Strang
halves close on the correct one-step phase exp(i k0 (n^2-1) dx/2), so the modal
reference uses the physical profile alpha directly (no effective rescaling).

The launching Gaussian is y-invariant, so it is exactly periodic on the y grid
and no y-sponge is applied (a y-sponge on a y-invariant field would manufacture
a spurious ky spectrum). The top sponge absorbs the evanescent tail. The
residual rel.L2 then measures the modal-truncation floor and the finite-dx
Strang diffraction/refraction commutator, not a factor mismatch.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.fieldio import save_field_npz

t0 = time.time()
f_GHz = 1.0

# ---- duct profile: n^2 = 1 + alpha z, alpha<0 (trapping) ----
ALPHA = -2.0e-4          # slope of n^2 with height [1/m] (as fed to solver)
Z_S = 4.0                # source height [m]
SIG_Z = 2.0              # source Gaussian width [m]
X_END = 50.0             # short range so Strang commutator stays small
DX = 1.0
DZ = 0.05
Z_MAX = 50.0
NZ = int(round(Z_MAX / DZ))
NY = 32
DY = 1.0

# corrected solver closes on the physical phase -> modal reference uses alpha
ALPHA_EFF = ALPHA

# negative zeros of Ai computed numerically (independent of the PE); the m-th
# zero is bracketed around its asymptotic estimate -(1.5*pi*(m-0.25))^(2/3).
from scipy.optimize import brentq


def _ai(x):
    return airy(x)[0]


def _airy_zeros(n):
    out = []
    for m in range(1, n + 1):
        c = -(1.5 * np.pi * (m - 0.25)) ** (2.0 / 3.0)
        lo, hi = c - 0.5, c + 0.5
        while _ai(lo) * _ai(hi) > 0:
            lo -= 0.5
            hi += 0.5
        out.append(brentq(_ai, lo, hi, xtol=1e-12))
    return np.array(out)


N_MODES = 25
AIRY_ZEROS = _airy_zeros(N_MODES)

k0 = 2.0 * np.pi * f_GHz * 1.0e9 / 2.99792458e8
C_AIRY = (k0 ** 2 * abs(ALPHA_EFF)) ** (1.0 / 3.0)


def n_profile(_x, _y, z):
    """n(z) = sqrt(1 + alpha z), y-invariant (fed to the solver)."""
    return np.sqrt(1.0 + ALPHA * np.asarray(z, dtype=float))


# ---- Airy mode shapes on the PE z-grid (effective profile) ----
z = np.arange(NZ) * DZ
psi_modes = airy(C_AIRY * z[None, :] + AIRY_ZEROS[:, None])[0]   # (M, Nz)
gamma_m = AIRY_ZEROS * abs(ALPHA_EFF) / (2.0 * C_AIRY)            # (M,) <0

# ---- project launching Gaussian onto modes ----
u0 = np.exp(-(z - Z_S) ** 2 / (2 * SIG_Z ** 2))
norm = np.sum(psi_modes ** 2, axis=1) * DZ
coeffs = np.array([np.sum(psi_modes[m, :] * u0) * DZ / norm[m]
                   for m in range(N_MODES)])
u0_proj = np.sum(coeffs[:, None] * psi_modes, axis=0)
proj_residual = float(np.linalg.norm(u0_proj - u0) / np.linalg.norm(u0))

# ---- run the PE ----
prob = PE3D(f_GHz=f_GHz, dx_m=DX, dy_m=DY, dz_m=DZ, Ny=NY, Nz=NZ,
            y_max_m=NY * DY / 2.0, z_max_m=Z_MAX, ground="horizontal",
            operator="narrow", terrain=None, n_profile=n_profile,
            sponge_top_m=20.0, sponge_y_m=0.0, sponge_strength=8.0)
Yg, Zg = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
u0_grid = np.exp(-(Zg - Z_S) ** 2 / (2 * SIG_Z ** 2))
prob.initialize(u0_grid.astype(np.complex128))
prob.run(X_END)

# ---- modal reference at x=X_END ----
phase = np.exp(1j * k0 * gamma_m * X_END)
u_ref = np.sum(coeffs[:, None] * phase[:, None] * psi_modes, axis=0)
u_ref_2d = np.broadcast_to(u_ref[None, :], (NY, NZ))

# interior mask: beam region in z AND off the y-sponge rows (|y|<6 m)
mask = (Zg > 1.0) & (Zg < 15.0) & (np.abs(Yg) < 6.0)
rel_l2_complex = rel_l2(prob.u, u_ref_2d, mask)
iy = NY // 2
m1d = mask[iy, :]
amp_pe = np.abs(prob.u[iy, m1d]); amp_ref = np.abs(u_ref_2d[iy, m1d])
amp_err = float(np.linalg.norm(amp_pe - amp_ref) / np.linalg.norm(amp_ref))
# amplitude-weighted phase residual (only where both fields are non-negligible)
w = amp_ref / (np.max(amp_ref) + 1e-30)
good = w > 0.1
pha_pe = np.unwrap(np.angle(prob.u[iy, m1d][good]))
pha_ref = np.unwrap(np.angle(u_ref_2d[iy, m1d][good]))
pha_resid = float(np.mean(np.abs(pha_pe - pha_ref)))

results = dict(
    setup=dict(
        f_GHz=f_GHz, lambda_m=prob.lam, k0=k0,
        alpha_n2_per_m=ALPHA,
        alpha_eff_for_modal_ref=ALPHA_EFF,
        n2_profile="n^2(z) = 1 + alpha z  (linear; Airy modes)",
        c_airy_per_m=C_AIRY,
        n_modes=N_MODES,
        source_z=Z_S, sigma_z=SIG_Z, x_end_m=X_END, dx_m=DX, dz_m=DZ,
        reference=("modal superposition u_ref=sum c_m exp(i k0 gamma_m x) Ai(c z + a_m); "
                   "physical alpha; corrected refraction half-step closes on the PDE phase"),
        solver_note=("PE3D refraction factor exp(0.25 i k0(n^2-1)dx) applied twice per "
                     "Strang step = exp(i k0(n^2-1)dx/2), matching the analytic PDE."),
    ),
    metrics=dict(
        projection_rel_l2_at_x0=proj_residual,
        rel_l2_complex=rel_l2_complex,
        amplitude_rel_l2=amp_err,
        mean_phase_residual_rad=pha_resid,
        mode_turning_points_m=(-AIRY_ZEROS / C_AIRY).tolist(),
        gammas=gamma_m.tolist(),
    ),
    runtime_s=time.time() - t0,
)
results["pass"] = bool(
    proj_residual < 0.05
    and rel_l2_complex < 0.15
    and amp_err < 0.08
    and abs(pha_resid) < 0.2)
save_json("bench9_duct", results)
save_field_npz(f"{DATA}/bench9_field_pe.npz", prob.u, prob.y_phys, prob.z_phys,
              dict(bench=9, case="linear_n2_duct", x_end=X_END,
                   alpha=ALPHA, n_modes=N_MODES))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(z, np.abs(prob.u[iy, :]), label="SSFT PE |u|")
ax[0].plot(z, np.abs(u_ref_2d[iy, :]), "k--", label="Airy modal ref |u|")
ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("|u|")
ax[0].set_xlim(0, 15); ax[0].set_ylim(bottom=0)
ax[0].set_title(f"(a) x={X_END:.0f} m, rel.L2={rel_l2_complex:.3e}")
ax[0].legend(fontsize=8)
ax[1].plot(z, np.angle(prob.u[iy, :]), label="SSFT PE phase")
ax[1].plot(z, np.angle(u_ref_2d[iy, :]), "k--", label="Airy modal phase")
ax[1].set_xlabel("z [m]"); ax[1].set_ylabel("phase [rad]")
ax[1].set_xlim(0, 15)
ax[1].set_title(f"(b) phase, mean resid={pha_resid:.3f} rad")
ax[1].legend(fontsize=8)
savefig("bench9_airy_duct")

print(f"BENCH9 done. proj resid={proj_residual:.4e}, rel.L2={rel_l2_complex:.4e}, "
      f"amp rel.L2={amp_err:.4e}, phase resid={pha_resid:.3f} rad, "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
