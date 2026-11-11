"""Bench 7 (P0-05): TRUE 3-D (y,z) free-space verification with an INDEPENDENT
analytic reference, plus the 3-D PEC image check.

The narrow-angle parabolic equation in a uniform medium is SEPARABLE in y and z:
    u_x = i/(2 k0) (u_yy + u_zz)
so the exact solution of a 2-D Gaussian aperture
    u(0, y, z) = exp(-(y-y0)^2/(2 sy^2) - (z-z0)^2/(2 sz^2))
is the PRODUCT of two 1-D free-space Gaussian-beam kernels:
    G_q(eta; x) = sigma / sqrt(sigma^2 + i x/k0) * exp(-eta^2 / (2 (sigma^2 + i x/k0)))
    u_exact(x,y,z) = G_y(y-y0; x) * G_z(z-z0; x)                       (ground="none")

For a PEC horizontal ground (Dirichlet u(z=0)=0) the image method gives
    u_exact(x,y,z) = G_y(y-y0; x) * ( G_z(z-z0; x) - G_z(z+z0; x) )
(source at +z0, negative image at -z0).  This is exactly the boundary condition
the propagator enforces by the extended-z FFT symmetry, so it is an independent
end-to-end test of the image machinery in 3-D (not just the y-invariant bench2).

Primary comparison uses operator="narrow" so the PE solves *this* equation
exactly; residual rel.L2 then measures only the FFT/sponge/finite-domain
discretisation.  operator="exact" is also run to report the additional
wide-angle (sqrt(1-q) vs 1-q/2) error on the same beam.

Metrics: terminal (y,z) complex rel.L2 over the interior (off-sponge) region,
plus |u(y,0)| at the PEC surface for the ground case.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.fieldio import save_field_npz

t0 = time.time()
f_GHz = 1.0
sy = 3.0
sz = 3.0
dx = 1.0
dy = 0.5
dz = 0.25
Ny = 128
Nz = 512
Y_MAX = Ny * dy / 2.0
x_end = 300.0

# interior (off-sponge) window in physical coords.
# free-space grid: z in [-64,64), sponge_top=30 -> taper starts at |z|=34 m;
#                  y in [-32,32), sponge_y=15 -> taper starts at |y|=17 m.
# beam at x=300 m spreads to ~5 m sigma (<=15 m at 3 sigma), so these
# windows stay fully clear of the sponge.
Y_INT = 12.0
Z_INT_FREE = 25.0
Z_INT_LO, Z_INT_HI = 2.0, 60.0


def gaussian_kernel(eta, sigma, x, k0):
    """1-D free-space narrow-angle Gaussian-beam closed form G(eta; x)."""
    a = sigma ** 2 + 1j * x / k0
    return sigma / np.sqrt(a) * np.exp(-eta ** 2 / (2.0 * a))


def run_free(operator):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=Y_MAX, z_max_m=Nz * dz / 2.0, ground="none",
                operator=operator,
                sponge_top_m=30.0, sponge_y_m=15.0, sponge_strength=8.0)
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u0 = np.exp(-Y ** 2 / (2 * sy ** 2) - Z ** 2 / (2 * sz ** 2))
    prob.initialize(u0.astype(np.complex128))
    prob.run(x_end)
    return prob


def run_mirror(operator):
    z_s = 10.0
    szm = 1.5
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=Y_MAX, z_max_m=Nz * dz, ground="horizontal",
                operator=operator,
                sponge_top_m=40.0, sponge_y_m=15.0, sponge_strength=8.0)
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u0 = np.exp(-Y ** 2 / (2 * sy ** 2) - (Z - z_s) ** 2 / (2 * szm ** 2))
    prob.initialize(u0.astype(np.complex128))
    prob.run(x_end)
    return prob, z_s, szm


# ---------------------------------------------------------------- (a) free space
res = {}
for op in ("narrow", "exact"):
    prob = run_free(op)
    k0 = prob.k0
    Yg, Zg = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u_ref = (gaussian_kernel(Yg, sy, x_end, k0)
             * gaussian_kernel(Zg, sz, x_end, k0))
    mask = (np.abs(Yg) < Y_INT) & (np.abs(Zg) < Z_INT_FREE)
    res[op] = dict(prob=prob, rel_l2=rel_l2(prob.u, u_ref, mask), u_ref=u_ref, mask=mask)

prob_narr = res["narrow"]["prob"]

# ---------------------------------------------------------------- (b) PEC mirror
prob_m, z_s, szm = run_mirror("narrow")
k0m = prob_m.k0
Ym, Zm = np.meshgrid(prob_m.y_phys, prob_m.z_phys, indexing="ij")
u_ref_m = (gaussian_kernel(Ym, sy, x_end, k0m)
           * (gaussian_kernel(Zm - z_s, szm, x_end, k0m)
              - gaussian_kernel(Zm + z_s, szm, x_end, k0m)))
mask_m = (np.abs(Ym) < Y_INT) & (Zm > Z_INT_LO) & (Zm < Z_INT_HI)
err_mirror = rel_l2(prob_m.u, u_ref_m, mask_m)
u_ground = float(np.max(np.abs(prob_m.u[:, 0])))

# ---------------------------------------------------------------- assemble JSON
results = dict(
    setup=dict(
        f_GHz=f_GHz, lambda_m=prob_narr.lam, k0=prob_narr.k0,
        dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz, y_max_m=Y_MAX,
        sigma_y=sy, sigma_z=sz, x_end_m=x_end,
        analytic_ref=(
            "u_exact = Gy(y-y0;x)*Gz(z-z0;x), "
            "G_q(eta;x)=sigma/sqrt(sigma^2+i x/k0)*exp(-eta^2/(2(sigma^2+i x/k0)))"
        ),
        interior_window=dict(y_half=Y_INT, z_half_free=Z_INT_FREE,
                             z_lo=Z_INT_LO, z_hi=Z_INT_HI),
    ),
    free_space=dict(
        rel_l2_narrow=res["narrow"]["rel_l2"],
        rel_l2_exact=res["exact"]["rel_l2"],
        note=("narrow operator solves the separable equation exactly; residual "
              "is FFT/sponge/finite-domain error.  exact adds wide-angle error."),
    ),
    pec_mirror=dict(
        source_z=z_s, sigma_z=szm,
        rel_l2_interior=err_mirror,
        max_abs_u_at_ground=u_ground,
        expected_ground=0.0,
        analytic_ref="u = Gy(y;x)*(Gz(z-zs;x) - Gz(z+zs;x))  (Dirichlet image)",
    ),
    runtime_s=time.time() - t0,
)
results["pass"] = bool(
    results["free_space"]["rel_l2_narrow"] < 0.02
    and results["free_space"]["rel_l2_exact"] < 0.05
    and results["pec_mirror"]["rel_l2_interior"] < 0.05
    and u_ground < 1e-6)
save_json("bench7_3d_gaussian", results)
save_field_npz(f"{DATA}/bench7_field_free_narrow.npz", prob_narr.u,
               prob_narr.y_phys, prob_narr.z_phys,
               dict(bench=7, case="free_narrow", x_end=x_end))
save_field_npz(f"{DATA}/bench7_field_mirror.npz", prob_m.u,
               prob_m.y_phys, prob_m.z_phys,
               dict(bench=7, case="pec_mirror", x_end=x_end, z_s=z_s))

# ---------------------------------------------------------------- figures
# (a) free-space terminal slice at y=0 (z cut) and at z=0 (y cut)
iz0 = np.argmin(np.abs(prob_narr.z_phys - 0.0))
iy0 = np.argmin(np.abs(prob_narr.y_phys - 0.0))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax[0].plot(prob_narr.z_phys, np.abs(res["narrow"]["prob"].u[iy0, :]),
           label="PE narrow |u(y=0,:)|")
ax[0].plot(prob_narr.z_phys, np.abs(res["narrow"]["u_ref"][iy0, :]), "k--",
           label="analytic product")
ax[0].plot(prob_narr.z_phys, np.abs(res["exact"]["prob"].u[iy0, :]), ":",
           lw=1.2, label="PE exact |u|")
ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("|u|")
ax[0].set_title(f"(a) free space, y=0 cut, x={x_end:.0f} m")
ax[0].set_xlim(-45, 45); ax[0].legend()
ax[1].plot(prob_narr.y_phys, np.abs(res["narrow"]["prob"].u[:, iz0]),
           label="PE narrow |u(:,z=0)|")
ax[1].plot(prob_narr.y_phys, np.abs(res["narrow"]["u_ref"][:, iz0]), "k--",
           label="analytic product")
ax[1].set_xlabel("y [m]"); ax[1].set_ylabel("|u|")
ax[1].set_title(f"(b) free space, z=0 cut, x={x_end:.0f} m")
ax[1].set_xlim(-30, 30); ax[1].legend()
savefig("bench7_3d_gaussian_free")

# (c) PEC mirror: z cut at y=0
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(prob_m.z_phys, np.abs(prob_m.u[iy0, :]), label="PE (image method)")
ax.plot(prob_m.z_phys, np.abs(u_ref_m[iy0, :]), "k--", label="analytic source-image")
ax.axvline(z_s, color="gray", ls=":", lw=0.8, label=f"source z={z_s}")
ax.axvline(0.0, color="r", ls="-", lw=0.6, label="PEC ground z=0")
ax.set_xlabel("z [m]"); ax.set_ylabel("|u|")
ax.set_xlim(0, 70)
ax.set_title(f"Bench7(c): 3-D PEC mirror, relL2={err_mirror:.3e}, "
             f"|u@ground|={u_ground:.1e}")
ax.legend(); ax.grid(True, alpha=0.3)
savefig("bench7_pec_mirror_3d")

print(f"BENCH7 done. free narrow relL2={res['narrow']['rel_l2']:.4e}, "
      f"free exact relL2={res['exact']['rel_l2']:.4e}, "
      f"mirror relL2={err_mirror:.4e}, |u@ground|={u_ground:.1e}, "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
