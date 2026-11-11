"""Bench 2: PEC ground image interference vs analytical image superposition.

A source at height z_s above a PEC plane has the exact solution
    u(x,z) = U(x, z - z_s) -/+ U(x, z + z_s)
with minus for horizontal polarization (Dirichlet, E_y=0 at z=0) and plus for
vertical polarization (Neumann, dH_y/dz=0 at z=0).  U is the free-space
Gaussian-beam closed form.  The propagator implements the ground by the image
method (extended-z FFT symmetry); this benchmark verifies the interference
fringes against the analytical superposition.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.fieldio import save_field_npz

t0 = time.time()
f_GHz = 1.0
sz = 1.5
z_s = 10.0
dx = 1.0; dy = 2.0; dz = 0.25
Ny = 32; Nz = 512
x_end = 400.0


def beam(x, z, k0):
    az = sz ** 2 + 1j * x / k0
    return sz / np.sqrt(az) * np.exp(-z ** 2 / (2 * az))


def run_case(ground):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=32.0, z_max_m=128.0, ground=ground, operator="exact",
                sponge_top_m=25.0, sponge_y_m=0.0, sponge_strength=8.0)
    u0 = np.empty((Ny, Nz), dtype=np.complex128)
    u0[:, :] = np.exp(-(prob.z_phys - z_s) ** 2 / (2 * sz ** 2))[None, :]
    prob.initialize(u0)
    prob.run(x_end)
    return prob


results = dict(setup=dict(f_GHz=f_GHz, dx=dx, dy=dy, dz=dz, Ny=Ny, Nz=Nz,
                         sigma_z=sz, source_height=z_s, x_end=x_end,
                         lambda_m=None, k0=None,
                         nyquist_angle_deg=None))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

for ax, ground, sign, tag in [(axes[0], "horizontal", -1, "H-pol (Dirichlet)"),
                              (axes[1], "vertical", +1, "V-pol (Neumann)")]:
    prob = run_case(ground)
    k0 = prob.k0
    z = prob.z_phys
    u_ana = beam(x_end, z - z_s, k0) + sign * beam(x_end, z + z_s, k0)
    m = (z > 2.0) & (z < 80.0)
    err = rel_l2(prob.u[Ny // 2, m], u_ana[m])
    ground_res = float(abs(prob.u[Ny // 2, 0]))
    results[ground] = dict(
        rel_l2_interior=err,
        value_at_ground=ground_res,
        expected_ground=0.0 if ground == "horizontal" else None,
    )
    ax.plot(z, np.abs(prob.u[Ny // 2, :]), label="PE (image method)")
    ax.plot(z, np.abs(u_ana), "k--", label="analytic superposition")
    ax.axvline(z_s, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("z [m]"); ax.set_ylabel("|u|")
    ax.set_title(f"{tag}: relL2={err:.3e}")
    ax.set_xlim(0, 70); ax.legend()
    save_field_npz(f"{DATA}/bench2_field_{ground}.npz", prob.u, prob.y_phys, prob.z_phys,
                   dict(ground=ground, x_end=x_end, bench=2, analytic_available=True))
    results[ground]["pass"] = bool(err < 0.05 and (
        ground_res < 1e-6 if ground == "horizontal" else True))

results["setup"]["lambda_m"] = prob.lam
results["setup"]["k0"] = prob.k0
results["setup"]["nyquist_angle_deg"] = float(np.degrees(np.arcsin(prob.lam / (2 * dz))))
results["pass"] = bool(results["horizontal"]["pass"] and results["vertical"]["pass"])
save_json("bench2_pec_mirror", results)
savefig("bench2_pec_mirror_interference")
print(f"BENCH2 done. H-pol relL2={results['horizontal']['rel_l2_interior']:.4e} "
      f"(u@ground={results['horizontal']['value_at_ground']:.2e}), "
      f"V-pol relL2={results['vertical']['rel_l2_interior']:.4e}, "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
