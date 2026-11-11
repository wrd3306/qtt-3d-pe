"""Bench 4: grid convergence in dx / dy / dz.

2D-Gaussian aperture propagated through a weakly stratified medium; the reference
is a very-fine-grid run of the same solver.  Sweeping one step at a time:
  - dx: Strang symmetric splitting should show ~2nd-order convergence;
  - dz / dy: FFT spectral discretization error.
Domains are large enough that sponge tails never touch the beam interior.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D
from solver.medium import StandardAtmosphere, LayeredDuct

t0 = time.time()
f_GHz = 1.0
sig = 2.0
x_end = 400.0
# strong evaporation-duct M profile -> large refraction/diffraction commutator,
# so the Strang O(dx^2) splitting error dominates the FFT noise floor.
MEDIUM = LayeredDuct(breaks_z=[0.0, 10.0, 30.0, 120.0],
                     breaks_M=[301.0, 293.0, 300.0, 318.0])
Y_MAX, Z_MAX = 50.0, 120.0
REF = dict(dx=0.5, dy=0.5, dz=0.125, Ny=200, Nz=1920)


def run_one(dx, dy, dz, Ny, Nz):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=Y_MAX, z_max_m=Z_MAX, ground="none", operator="exact",
                n_profile=MEDIUM,
                sponge_top_m=25.0, sponge_y_m=12.0, sponge_strength=8.0)
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u0 = np.exp(-Y ** 2 / (2 * sig ** 2) - Z ** 2 / (2 * sig ** 2))
    prob.initialize(u0.astype(np.complex128))
    prob.run(x_end)
    interior = (np.abs(Y) < 30.0) & (np.abs(Z) < 90.0)
    return prob, interior


# --- fine-grid reference ---
ref_prob, _ = run_one(REF["dx"], REF["dy"], REF["dz"], REF["Ny"], REF["Nz"])
ref_u = ref_prob.u
ref_y = ref_prob.y_phys
ref_z = ref_prob.z_phys
interp_ref = RegularGridInterpolator((ref_y, ref_z), ref_u,
                                     bounds_error=False, fill_value=0.0)


def err_vs_ref(prob, interior):
    Y, Z = np.meshgrid(prob.y_phys, prob.z_phys, indexing="ij")
    u_ref = interp_ref(np.stack([Y.ravel(), Z.ravel()], -1)).reshape(Y.shape)
    return rel_l2(prob.u, u_ref, interior)


# --- sweeps ---
dxs = [1.0, 2.0, 4.0, 8.0, 16.0]
err_dx = []
for dx in dxs:
    p, m = run_one(dx, REF["dy"], REF["dz"], REF["Ny"], REF["Nz"])
    err_dx.append(err_vs_ref(p, m))

dzs = [0.25, 0.5, 1.0, 2.0]
err_dz = []
for dz in dzs:
    Nz = int(round(2 * Z_MAX / dz))
    p, m = run_one(REF["dx"], REF["dy"], dz, REF["Ny"], Nz)
    err_dz.append(err_vs_ref(p, m))

dys = [1.0, 2.0, 4.0]
err_dy = []
for dy in dys:
    Ny = int(round(2 * Y_MAX / dy))
    p, m = run_one(REF["dx"], dy, REF["dz"], Ny, REF["Nz"])
    err_dy.append(err_vs_ref(p, m))


def slope_of(h, e):
    h = np.array(h, float); e = np.array(e, float)
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])

results = dict(
    setup=dict(f_GHz=f_GHz, sigma=sig, x_end=x_end,
               medium="LayeredDuct M=[301,293,300,318] at z=[0,10,30,120]",
               reference=f"fine grid dx={REF['dx']} dy={REF['dy']} dz={REF['dz']}"),
    sweep_dx=dict(dx_m=dxs, rel_l2=err_dx, loglog_slope=slope_of(dxs, err_dx)),
    sweep_dz=dict(dz_m=dzs, rel_l2=err_dz, loglog_slope=slope_of(dzs, err_dz)),
    sweep_dy=dict(dy_m=dys, rel_l2=err_dy, loglog_slope=slope_of(dys, err_dy)),
    runtime_s=time.time()-t0,
)
results["pass"] = bool(
    err_dx[-1] > err_dx[0] * 2 and
    err_dz[-1] > err_dz[0] * 2 and
    err_dy[-1] > err_dy[0] * 2)
save_json("bench4_grid_convergence", results)

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.loglog(dxs, err_dx, "o-", label=f"dx (slope {results['sweep_dx']['loglog_slope']:.2f})")
ax.loglog(dzs, err_dz, "s-", label=f"dz (slope {results['sweep_dz']['loglog_slope']:.2f})")
ax.loglog(dys, err_dy, "^-", label=f"dy (slope {results['sweep_dy']['loglog_slope']:.2f})")
ax.set_xlabel("grid step [m]"); ax.set_ylabel("relative L2 error vs fine-grid reference")
ax.set_title("Bench4: grid convergence (stratified medium, fine-grid reference)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
savefig("bench4_grid_convergence")

print(f"BENCH4 done. dx slope={results['sweep_dx']['loglog_slope']:.2f}, "
      f"dz slope={results['sweep_dz']['loglog_slope']:.2f}, "
      f"dy slope={results['sweep_dy']['loglog_slope']:.2f}, PASS={results['pass']} "
      f"({time.time()-t0:.1f}s)")
print("  dx:", [f"{e:.3e}" for e in err_dx])
print("  dz:", [f"{e:.3e}" for e in err_dz])
print("  dy:", [f"{e:.3e}" for e in err_dy])
