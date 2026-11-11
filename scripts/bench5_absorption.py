"""Bench 5: absorbing-layer reflection-coefficient measurement.

Launch a Gaussian-windowed tilted plane wave; the incident spectral peak is
measured at launch (x=0).  The wave strikes the top sponge, reflects, and its
spectral peak (now at -k_z) is measured in the interior on the way back.
Reflection coefficient R = |U(-k_z) after reflection| / |U(+k_z) at launch|,
which cancels the window/FFT normalization.  Measured vs incidence angle and
sponge thickness.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from common_bench import (DATA, FIGS, savefig, save_json)
from solver.propagator import PE3D

t0 = time.time()
f_GHz = 1.0
Z_HALF = 100.0
dz = 0.25
Nz = int(2 * Z_HALF / dz)
Ny = 16
z0 = -40.0
sig = 3.0
dx = 2.0


def measure(theta0_deg, L_sponge):
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=4.0, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=Ny * 4.0 / 2, z_max_m=Z_HALF, ground="none", operator="exact",
                sponge_top_m=L_sponge, sponge_y_m=0.0, sponge_strength=8.0)
    th = np.radians(theta0_deg)
    u1d = (np.exp(-(prob.z_phys - z0) ** 2 / (2 * sig ** 2))
           * np.exp(1j * prob.k0 * prob.z_phys * np.sin(th)))
    prob.initialize(np.tile(u1d[None, :], (Ny, 1)))

    def peak(u_row, kz_target):
        kz = np.fft.fftshift(np.fft.fftfreq(len(prob.z_phys), dz)) * 2 * np.pi
        U = np.fft.fftshift(np.fft.fft(u_row))
        i = np.argmin(np.abs(kz - kz_target))
        lo, hi = max(0, i - 3), min(len(U), i + 4)
        return np.max(np.abs(U[lo:hi]))

    k_inc = prob.k0 * np.sin(th)
    U_inc = peak(prob.u[Ny // 2, :], k_inc)

    x_end = (240.0 - 2 * L_sponge) / np.tan(th)
    prob.run(x_end)
    U_ref = peak(prob.u[Ny // 2, :], -k_inc)
    R = U_ref / U_inc
    return R, x_end


thetas = [5, 10, 15, 20, 25, 30, 35]
Ls = [2.0, 5.0, 10.0, 20.0]
table = {}
for L in Ls:
    row = []
    for th in thetas:
        R, xend = measure(th, L)
        row.append(R)
        print(f"  L={L:5.1f}m theta={th:3d} deg: R={R:.3e} ({20*np.log10(R+1e-30):.1f} dB)")
    table[L] = row

results = dict(
    setup=dict(f_GHz=f_GHz, dz=dz, dx=dx, z0=z0, sigma=sig,
               sponge_profile="exp(-strength*t^2), strength=8",
               lambda_m=0.2998,
               nyquist_angle_deg=float(np.degrees(np.arcsin(0.2998/(2*dz))))),
    angles_deg=thetas,
    sponge_thickness_m=Ls,
    R_vs_L_theta={str(L): row for L, row in table.items()},
    runtime_s=time.time()-t0,
)
design = results["R_vs_L_theta"]["10.0"]
results["metrics"] = dict(
    max_R_dB_design_L10_0_30deg=float(20*np.log10(max(design[:6])+1e-30)),
    min_R_dB=float(20*np.log10(min(min(r) for r in table.values())+1e-30)),
)
results["pass"] = bool(results["metrics"]["max_R_dB_design_L10_0_30deg"] < -20.0)
save_json("bench5_absorption", results)

fig, ax = plt.subplots(figsize=(6.5, 4))
for L in Ls:
    ax.plot(thetas, 20*np.log10(np.array(table[L])+1e-30), "o-", label=f"L={L} m")
ax.set_xlabel("incidence angle [deg]"); ax.set_ylabel("reflection coefficient [dB]")
ax.set_title("Bench5: sponge reflection vs angle and layer thickness")
ax.legend(); ax.grid(True, alpha=0.3)
savefig("bench5_sponge_reflection")

print(f"BENCH5 done. design L=10m max R(0-30deg)={results['metrics']['max_R_dB_design_L10_0_30deg']:.1f} dB, "
      f"PASS={results['pass']} ({time.time()-t0:.1f}s)")
