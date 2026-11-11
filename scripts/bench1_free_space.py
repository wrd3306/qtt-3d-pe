"""Bench 1: free-space CYLINDRICAL spreading vs analytic closed form.

A y-invariant (line-source) aperture reduces the 3D PE to the 2D (range-height)
Schroedinger equation.  For u0(z)=exp(-z^2/2 sz^2) the exact solution is
    u(x,z) = sz / sqrt(sz^2 + i x/k0) * exp(-z^2/(2(sz^2 + i x/k0)))
which carries the cylindrical-spreading law |u| ~ x^{-1/2} in the far field
(x >> k0 sz^2).

Metrics:
  (a) relative L2 error of propagated slice vs closed form (interior, off-sponge);
  (b) fitted log-log decay exponent of |u(x,0)| over x >> k0 sz^2 (expect -1/2);
  (c) Padé[2/2] vs exact spectral operator (near-axis agreement).
Note: y-invariant problem -> no y sponge (it would modulate a uniform field);
the z sponge only lives outside the beam tails.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.fieldio import save_field_npz

t0 = time.time()
f_GHz = 1.0
sz = 3.0
dx = 1.0; dy = 2.0; dz = 0.25
Ny = 32; Nz = 1024
x_end = 2000.0
K_SIG = None  # set after run


def closed_form(x, z, k0):
    az = sz ** 2 + 1j * x / k0
    return sz / np.sqrt(az) * np.exp(-z ** 2 / (2 * az))


def run_case(operator):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=32.0, z_max_m=128.0, ground="none", operator=operator,
                sponge_top_m=25.0, sponge_y_m=0.0, sponge_strength=8.0)
    u0 = np.empty((Ny, Nz), dtype=np.complex128)
    u0[:, :] = np.exp(-prob.z_phys ** 2 / (2 * sz ** 2))[None, :]
    prob.initialize(u0)
    xs, center, err = [], [], []
    z = prob.z_phys
    interior = np.abs(z) < 80.0
    iz = np.argmin(np.abs(z - 0.0))
    for n in range(int(x_end / prob.dx)):
        prob.step()
        if (n + 1) % 50 == 0:
            x = prob.x
            xs.append(x)
            center.append(abs(prob.u[Ny // 2, iz]))
            u_ref = closed_form(x, z, prob.k0)[None, :] * np.ones((Ny, 1))
            err.append(rel_l2(prob.u[:, interior], u_ref[:, interior]))
    return prob, np.array(xs), np.array(center), np.array(err)


prob_ex, xs, center, err_ex = run_case("exact")
prob_p22, _, _, err_p22 = run_case("pade22")

# decay exponent in the true far field: x >> k0 sz^2
k_sig = prob_ex.k0 * sz ** 2
mask = xs >= 4.0 * k_sig
slope, intercept = np.polyfit(np.log(xs[mask]), np.log(center[mask]), 1)

z = prob_ex.z_phys
interior = np.abs(z) < 80.0
u_ref_final = closed_form(x_end, z, prob_ex.k0)[None, :] * np.ones((Ny, 1))
final_rel_l2 = rel_l2(prob_ex.u[:, interior], u_ref_final[:, interior])

results = dict(
    setup=dict(f_GHz=f_GHz, dx=dx, dy=dy, dz=dz, Ny=Ny, Nz=Nz, sigma_z=sz,
               line_source_y_invariant=True, x_end=x_end,
               lambda_m=prob_ex.lam, k0=prob_ex.k0, k0_sigma2=k_sig,
               nyquist_angle_deg=float(np.degrees(np.arcsin(prob_ex.lam / (2 * dz)))),
               fit_range_x_m=[float(xs[mask][0]), float(x_end)]),
    metrics=dict(
        rel_l2_final_slice=final_rel_l2,
        decay_exponent=float(slope),
        decay_exponent_expected=-0.5,
        rel_l2_pade22_final=float(err_p22[-1]),
        rel_l2_exact_max_sampled=float(err_ex.max()),
        runtime_s=time.time() - t0,
    ),
    sampled=dict(x_m=xs.tolist(), center_amplitude=center.tolist(),
                 rel_l2_exact=err_ex.tolist(), rel_l2_pade22=err_p22.tolist()),
)
results["metrics"]["pass_rel_l2"] = bool(final_rel_l2 < 0.05)
results["metrics"]["pass_decay"] = bool(abs(slope + 0.5) < 0.05)
results["pass"] = bool(results["metrics"]["pass_rel_l2"] and results["metrics"]["pass_decay"])
save_json("bench1_free_space", results)
save_field_npz(f"{DATA}/bench1_field_exact.npz", prob_ex.u, prob_ex.y_phys, prob_ex.z_phys,
               dict(operator="exact", x_end=x_end, bench=1))
save_field_npz(f"{DATA}/bench1_field_pade22.npz", prob_p22.u, prob_p22.y_phys, prob_p22.z_phys,
               dict(operator="pade22", x_end=x_end, bench=1))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
sl = np.abs(prob_ex.u[Ny // 2, :])
ax[0].plot(z, sl, label="PE |u|")
ax[0].plot(z, np.abs(u_ref_final[Ny // 2, :]), "k--", label="closed form")
ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("|u|")
ax[0].set_title(f"(a) slice at x={x_end:.0f} m"); ax[0].legend()
ax[1].loglog(xs, center, "o-", ms=3, label="PE beam center")
ax[1].loglog(xs, center[mask][0] * (xs / xs[mask][0]) ** (-0.5), "k--",
             label=r"$\propto x^{-1/2}$ cylindrical")
ax[1].axvline(4 * k_sig, color="r", ls=":", label=r"$4\,k_0\sigma^2$")
ax[1].set_xlabel("range x [m]"); ax[1].set_ylabel("|u(0)|")
ax[1].set_title(f"(b) fitted exponent = {slope:.3f} (expect -0.5)")
ax[1].legend()
savefig("bench1_free_space_spreading")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(xs, err_ex, "o-", ms=3, label="exact spectral operator")
ax.semilogy(xs, err_p22, "s-", ms=3, label="Padé[2/2]")
ax.set_xlabel("range x [m]"); ax.set_ylabel("relative L2 error vs closed form")
ax.set_title("Bench1: free-space accuracy vs range")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
savefig("bench1_error_vs_range")

print(f"BENCH1 done. relL2={final_rel_l2:.4e}, decay={slope:.4f}, "
      f"pade22 relL2={err_p22[-1]:.4e}, PASS={results['pass']}")
