"""Bench 3: knife-edge diffraction vs Fresnel-Kirchhoff analytical solution.

Source at (0, z_s); absorbing half-plane screen at x=x_e, z<z_e (a physical
object, NOT a terrain staircase mask).  Beyond the edge the field relative to
free space is the Fresnel-Kirchhoff diffraction factor
    V(nu) = (1 - j)/2 * [ (0.5 - C(nu)) - j (0.5 - S(nu)) ]
with Fresnel integrals C, S (scipy.special.fresnel) and
    nu = (z_e - z_line) * sqrt(2/lambda * (1/x_e + 1/(x-x_e))),
    z_line = z_s + (z - z_s) * x_e / x  (direct-ray height at the edge).

Reference landmarks: nu=0 shadow boundary -> |V|=0.5 (-6 dB); lit region -> 0 dB;
deep shadow -> deep fade.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from common_bench import (DATA, FIGS, savefig, save_json, rel_l2)
from solver.propagator import PE3D

t0 = time.time()
f_GHz = 1.0
sz = 1.0
z_s = 0.0
x_e = 200.0
z_e = 3.0
dx = 1.0; dy = 2.0; dz = 0.1
Ny = 32; Nz = 1024
x_end = 400.0


def fresnel_kr_v(nu):
    """Fresnel-Kirchhoff diffraction factor V(nu)."""
    S, C = fresnel(nu)
    I = (0.5 - C) - 1j * (0.5 - S)
    return (1 - 1j) / 2.0 * I


def run(edge_mask: bool):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=32.0, z_max_m=51.2, ground="none", operator="exact",
                sponge_top_m=15.0, sponge_y_m=0.0, sponge_strength=8.0)
    u0 = np.empty((Ny, Nz), dtype=np.complex128)
    u0[:, :] = np.exp(-(prob.z_phys - z_s) ** 2 / (2 * sz ** 2))[None, :]
    prob.initialize(u0)
    n_edge = int(round(x_e / dx))
    for n in range(int(x_end / dx)):
        prob.step()
        if edge_mask and (n + 1) == n_edge:
            prob.u *= (prob.z_phys[None, :] >= z_e)
    return prob


prob_free = run(False)
prob_edge = run(True)

z = prob_edge.z_phys
k0 = prob_edge.k0
lam = prob_edge.lam

# receiver points in dB on a vertical slice at x = x_end
zz = np.linspace(-10, 12, 221)
ratio_pe = []
v_theory = []
for zr in zz:
    iz = np.argmin(np.abs(z - zr))
    r = abs(prob_edge.u[Ny // 2, iz]) / (abs(prob_free.u[Ny // 2, iz]) + 1e-30)
    ratio_pe.append(r)
    z_line = z_s + (zr - z_s) * x_e / x_end
    h = z_e - z_line
    nu = h * np.sqrt(2.0 / lam * (1.0 / x_e + 1.0 / (x_end - x_e)))
    v_theory.append(abs(fresnel_kr_v(nu)))
ratio_pe = np.array(ratio_pe); v_theory = np.array(v_theory)

# sample landmark points
def sample(zr):
    iz = np.argmin(np.abs(z - zr))
    r = abs(prob_edge.u[Ny // 2, iz]) / (abs(prob_free.u[Ny // 2, iz]) + 1e-30)
    z_line = z_s + (zr - z_s) * x_e / x_end
    nu = (z_e - z_line) * np.sqrt(2.0 / lam * (1.0 / x_e + 1.0 / (x_end - x_e)))
    return r, abs(fresnel_kr_v(nu)), 20*np.log10(r), 20*np.log10(abs(fresnel_kr_v(nu)))

landmarks = dict(
    shadow_boundary=sample(6.0),   # z_line = z_e -> nu = 0 -> |V|=0.5
    lit_region=sample(11.0),
    deep_shadow=sample(-5.0),
)
err_dB = {k: abs(landmarks[k][2] - landmarks[k][3]) for k in landmarks}
max_err = max(err_dB.values())

results = dict(
    setup=dict(f_GHz=f_GHz, dx=dx, dy=dy, dz=dz, Ny=Ny, Nz=Nz, sigma_z=sz,
               source_z=z_s, edge_x=x_e, edge_z=z_e, x_end=x_end,
               lambda_m=lam, k0=k0),
    landmarks={k: dict(pe_ratio=landmarks[k][0], theory_ratio=landmarks[k][1],
                       pe_dB=landmarks[k][2], theory_dB=landmarks[k][3],
                       abs_err_dB=err_dB[k]) for k in landmarks},
    sampled=dict(z_m=zz.tolist(), pe_ratio=ratio_pe.tolist(),
                 theory_ratio=v_theory.tolist()),
    metrics=dict(max_abs_err_dB=max_err, runtime_s=time.time()-t0),
)
results["pass"] = bool(max_err < 2.0)  # dB at landmark points
save_json("bench3_knife_edge", results)
prob_edge.save(f"{DATA}/bench3_field_edge.npz")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(zz, 20*np.log10(ratio_pe + 1e-12), "b-", label="PE |edge/free|")
ax.plot(zz, 20*np.log10(v_theory + 1e-12), "r--", label="Fresnel-Kirchhoff |V(nu)|")
ax.axvline(6.0, color="gray", ls=":", lw=0.8, label="shadow boundary")
ax.set_xlabel("receiver height z [m]"); ax.set_ylabel("field relative level [dB]")
ax.set_title(f"Bench3: knife-edge diffraction (max landmark err {max_err:.2f} dB)")
ax.set_ylim(-40, 5); ax.legend(); ax.grid(True, alpha=0.3)
savefig("bench3_knife_edge")

print(f"BENCH3 done. max landmark err={max_err:.3f} dB; "
      f"boundary PE={landmarks['shadow_boundary'][2]:.2f} dB (theory -6.02), "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
