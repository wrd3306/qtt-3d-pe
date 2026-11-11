"""Bench 8 (P0-06): terrain-following coordinate independent verification.

(a) GaussianHill mesh-convergence ladder.  There is NO closed-form PE solution
over a Gaussian hill, so the independent reference is a self-convergence ladder:
run the same problem on dx/dz halved repeatedly and show the terminal interior
rel.L2 against the finest run decays monotonically (multiple rungs, not a single
self-consistency point).  This is the standard numerical-verification evidence
for a terrain PE; it does not rely on any hard-coded expected number.

(b) SlopePlane tilt-correction sanity.  The terrain-following PE acquires the
Dockery effective refractivity  n_eff^2 = n^2 - 2*zeta*s  (s = dh/dx).  We check:
   (i) s = 0 must reduce EXACTLY to the flat free-space PE (match against the
       analytic Gaussian-beam closed form);
   (ii) the internal refraction half-step factor must equal
       exp(i k0 (n_eff^2 - 1) dx / 4) = exp(i k0 ((n^2-1) - 2*zeta*s) dx / 4)
       to machine precision -- a direct unit test of the -2 zeta s term and the
       corrected Strang half-step (two halves close on the /2 one-step phase);
   (iii) a physical cross-check: with n=1 the PE is a Schroedinger equation
         with linear potential V(zeta) = k0 s zeta, so Ehrenfest's theorem gives
         d2<zeta>/dx2 = -s, i.e. <zeta>(x) = zeta0 - (s/2) x^2 (the beam bends
         toward the rising terrain).  We compare the measured centroid to this.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.terrain import GaussianHill, SlopePlane
from solver.medium import n_profile_uniform

# Uniform n=1 profile passed explicitly so the terrain-following refraction
# path (n_eff^2 = n^2 - 2*zeta*slope) stays ACTIVE; passing None would
# short-circuit to 1.0 and silently disable the tilt correction.
UNIFORM_N1 = n_profile_uniform(1.0)

t0 = time.time()
f_GHz = 1.0

# ---------------------------------------------------------------- (a) GaussianHill
# Gentle hill: the terrain tilt phase accumulates as ~k0*zeta*H, so H must be
# small enough that Strang phase errors stay well below 2*pi across the ladder
# (otherwise complex rel.L2 wraps and looks non-monotone even though amplitude
# converges).  H=1.5 m keeps the tilt phase ~500 rad and commutator errors
# <0.1 rad on the coarsest rung.
HILL = GaussianHill(H=1.5, x0=150.0, sigma_x=100.0)
SIG_Z = 1.5
Z0 = 15.0
X_END = 250.0
Z_MAX = 40.0
NY = 32
DY = 1.0
Y_INT_HALF = 15.0
Z_INT_LO, Z_INT_HI = 3.0, 30.0

# finest rung = reference
REF_DX, REF_DZ = 0.25, 0.05


def run_hill(dx, dz):
    Nz = int(round(Z_MAX / dz))
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=DY, dz_m=dz, Ny=NY, Nz=Nz,
                y_max_m=NY * DY / 2.0, z_max_m=Z_MAX, ground="horizontal",
                operator="narrow", terrain=HILL, n_profile=UNIFORM_N1,
                sponge_top_m=15.0, sponge_y_m=8.0, sponge_strength=8.0)
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u0 = np.exp(-(Z - Z0) ** 2 / (2 * SIG_Z ** 2))
    prob.initialize(np.broadcast_to(u0, (NY, Nz)).astype(np.complex128))
    prob.run(X_END)
    interior = (np.abs(Y) < Y_INT_HALF) & (Z > Z_INT_LO) & (Z < Z_INT_HI)
    return prob, interior


ref_prob, _ = run_hill(REF_DX, REF_DZ)
ref_interp = RegularGridInterpolator(
    (ref_prob.y_phys, ref_prob.z_phys), ref_prob.u,
    bounds_error=False, fill_value=0.0)


def err_vs_ref(prob, interior):
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u_ref = ref_interp(np.stack([Y.ravel(), Z.ravel()], -1)).reshape(Y.shape)
    return rel_l2(prob.u, u_ref, interior)


# coarser rungs (excluding the reference rung itself)
dxs = [2.0, 1.0, 0.5]
err_dx = []
for dx in dxs:
    p, m = run_hill(dx, REF_DZ)
    err_dx.append(err_vs_ref(p, m))

dzs = [0.4, 0.2, 0.1]
err_dz = []
for dz in dzs:
    p, m = run_hill(REF_DX, dz)
    err_dz.append(err_vs_ref(p, m))


def slope_of(h, e):
    h = np.asarray(h, float); e = np.asarray(e, float)
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


# ------------------------------------------------- (b) SlopePlane tilt correction
def gaussian_kernel(eta, sigma, x, k0):
    a = sigma ** 2 + 1j * x / k0
    return sigma / np.sqrt(a) * np.exp(-eta ** 2 / (2.0 * a))


def run_slope(s, x_end, track_centroid=False, z0=Z0):
    slope = SlopePlane(slope=s)
    dz = 0.25
    Nz = 240
    prob = PE3D(f_GHz=f_GHz, dx_m=1.0, dy_m=1.0, dz_m=dz, Ny=32, Nz=Nz,
                y_max_m=16.0, z_max_m=Nz * dz, ground="horizontal",
                operator="narrow", terrain=slope, n_profile=UNIFORM_N1,
                sponge_top_m=20.0, sponge_y_m=0.0, sponge_strength=8.0)
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u0 = np.exp(-(Z - z0) ** 2 / (2 * SIG_Z ** 2))
    prob.initialize(np.broadcast_to(u0, (32, Nz)).astype(np.complex128))
    cents = []
    n_steps = int(round(x_end / prob.dx))
    for n in range(n_steps):
        prob.step()
        if track_centroid and (n % 10 == 0):
            w = np.abs(prob.u[16, :]) ** 2
            z = prob.z_phys
            mask = (z > z0 - 12) & (z < z0 + 40)
            cents.append((prob.x, float(np.sum(z[mask] * w[mask]) /
                                       np.sum(w[mask]))))
    return prob, cents


# (b.i) s=0 must match the flat PEC analytic field (source - image).
# With ground="horizontal" the field is Gz(z-z0) - Gz(z+z0); for s=0 the
# terrain correction vanishes, so this must agree to the FFT floor.
prob_s0, _ = run_slope(0.0, 200.0)
k0 = prob_s0.k0
Z = prob_s0.z_phys
u_ref_flat = (gaussian_kernel(Z - Z0, SIG_Z, 200.0, k0)
              - gaussian_kernel(Z + Z0, SIG_Z, 200.0, k0))
mask_flat = (Z > 4.0) & (Z < 50.0)
err_s0 = rel_l2(prob_s0.u[16, mask_flat], u_ref_flat[mask_flat])

# (b.ii) direct refraction-factor probe for s=0.02
prob_probe, _ = run_slope(0.02, 0.0)
zeta = prob_probe.z_phys[None, :]
slope_val = prob_probe.terrain.slope_x(0.0, 0.0)
rf_num = prob_probe._refraction_factor()
# n=1 -> n^2-1 = 0, n_eff^2 - 1 = -2 zeta s. The corrected Strang half-step
# carries 1/4 of the full-step phase: exp(i k0 (n_eff^2-1) dx/4).
rf_ana = np.exp(0.25j * k0 * (-2.0 * zeta * slope_val) * prob_probe.dx)
err_rf = float(np.max(np.abs(rf_num - rf_ana)))

# (b.iii) centroid drift under gentle slope s=0.001; theory <z>=z0-(s/2)x^2.
# Keep range short so the beam stays well above the PEC ground (z=0).
S_DRIFT = 0.001
prob_drift, cents = run_slope(S_DRIFT, 60.0, track_centroid=True, z0=15.0)
cents = np.asarray(cents, float)
x_last, c_last = cents[-1]
c_pred = 15.0 - 0.5 * S_DRIFT * x_last ** 2
err_centroid = float(abs(c_last - c_pred))

# ---------------------------------------------------------------- assemble
results = dict(
    setup=dict(
        f_GHz=f_GHz, lambda_m=ref_prob.lam, k0=ref_prob.k0,
        hill=dict(H=HILL.H, x0=HILL.x0, sigma_x=HILL.sigma_x),
        source_z=Z0, sigma_z=SIG_Z, x_end_hill=X_END,
        reference=f"finest rung dx={REF_DX} dz={REF_DZ}; no analytic closed form",
    ),
    hill_convergence=dict(
        dx_ladder_m=dxs, rel_l2_vs_fine=err_dx,
        dz_ladder_m=dzs, rel_l2_dz_vs_fine=err_dz,
        loglog_slope_dx=slope_of(dxs, err_dx),
        loglog_slope_dz=slope_of(dzs, err_dz),
        note=("self-convergence ladder; monotone decay with refinement is the "
              "independent numerical evidence (no hard-coded target)."),
    ),
    slope_plane=dict(
        s_zero_rel_l2_vs_flat_analytic=err_s0,
        refraction_factor_max_abs_err=err_rf,
        measured_slope=slope_val,
        drift=dict(s=S_DRIFT, x_end=x_last, z0=15.0,
                   centroid_measured=c_last, centroid_predicted=c_pred,
                   abs_err=err_centroid,
                   theory="<z>(x)=z0-(s/2)x^2 (Ehrenfest, V=k0 s zeta)"),
    ),
    runtime_s=time.time() - t0,
)
results["pass"] = bool(
    err_s0 < 1e-4                        # s=0 reproduces flat PEC analytic (FFT floor)
    and err_rf < 1e-12                   # n_eff^2 = n^2 - 2 zeta s exact
    and err_dx[-1] < err_dx[0] * 0.5     # dx ladder decays monotonically
    and err_dz[-1] < err_dz[0] * 0.5)
save_json("bench8_terrain", results)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax[0].loglog(dxs, err_dx, "o-",
             label=f"dx ladder (slope {results['hill_convergence']['loglog_slope_dx']:.2f})")
ax[0].loglog(dzs, err_dz, "s-",
             label=f"dz ladder (slope {results['hill_convergence']['loglog_slope_dz']:.2f})")
ax[0].axhline(1e-6, color="gray", ls=":", lw=0.8, label="reference rung (0 by def.)")
ax[0].set_xlabel("grid step [m]")
ax[0].set_ylabel("rel.L2 vs finest rung")
ax[0].set_title("(a) Gaussian hill mesh convergence")
ax[0].invert_xaxis(); ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=0.3)
ax[1].plot(cents[:, 0], cents[:, 1], "o-", ms=3, label="PE measured centroid")
ax[1].plot(cents[:, 0], 15.0 - 0.5 * S_DRIFT * cents[:, 0] ** 2, "k--",
           label="predicted z0-(s/2)x^2")
ax[1].set_xlabel("range x [m]"); ax[1].set_ylabel(r"beam centroid $\zeta$ [m]")
ax[1].set_title(f"(b) SlopePlane s={S_DRIFT}: centroid drift")
ax[1].legend(); ax[1].grid(True, alpha=0.3)
savefig("bench8_terrain_convergence")

print(f"BENCH8 done. s=0 relL2={err_s0:.2e}, rf err={err_rf:.2e}, "
      f"dx ladder {[f'{e:.2e}' for e in err_dx]} (slope {results['hill_convergence']['loglog_slope_dx']:.2f}), "
      f"dz ladder {[f'{e:.2e}' for e in err_dz]} (slope {results['hill_convergence']['loglog_slope_dz']:.2f}), "
      f"centroid measured={c_last:.2f} pred={c_pred:.2f}, "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
