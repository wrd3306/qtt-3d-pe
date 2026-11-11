"""Bench 6: fine-grid wide-angle operator accuracy at 0-30 deg+.

Per Nyquist (REBUILD_SPEC.md §7): +/-30 deg needs dz <= lambda.  At f=1 GHz,
dz = 0.30 m (=lambda), z domain +-128 m -> 854 points.  We launch a known
angular spectrum (a comb of plane waves at theta = 0,3,...,33 deg), propagate
with each diffraction operator, and compare the per-angle spectrum at x=R
against the exact spectral operator run.  Report amplitude and phase error.

Panel (b) overlays the *analytic* operator error |R(q)-sqrt(1-q)|/sqrt(1-q)
for Padé[2/2], Padé[2/1] and the banned old formula, so the measured curves
can be checked against theory.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from common_bench import (DATA, FIGS, savefig, save_json)
from solver.propagator import PE3D
from solver.operators import pade22, pade21, exact_sqrt, operator_error_table

t0 = time.time()
f_GHz = 1.0
lam = 0.299792458
dz = lam                     # dz = lambda : +/-30 deg resolved
z_half = 128.0
Nz = int(round(2 * z_half / dz))
dx = 0.5
Ny = 64
dy = 1.0
x_end = 100.0
thetas = np.arange(0, 34, 3)  # deg

prob0 = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
             y_max_m=Ny*dy/2, z_max_m=z_half, ground="none", operator="exact",
             sponge_top_m=40.0, sponge_y_m=0.0, sponge_strength=8.0)
k0 = prob0.k0

# angular-spectrum aperture: sum of plane waves at known angles
u0 = np.zeros((Ny, Nz), dtype=np.complex128)
for th in thetas:
    u0 += np.exp(1j * k0 * prob0.z_phys * np.sin(np.radians(th)))[None, :]
u0 *= np.exp(-(prob0.y_phys[:, None] ** 2) / (2 * 10.0 ** 2))

runs = {}
for op in ["exact", "pade22", "pade21"]:
    prob = PE3D(f_GHz=f_GHz, dx_m=dx, dy_m=dy, dz_m=dz, Ny=Ny, Nz=Nz,
                y_max_m=Ny*dy/2, z_max_m=z_half, ground="none", operator=op,
                sponge_top_m=40.0, sponge_y_m=0.0, sponge_strength=8.0)
    prob.initialize(u0.copy())
    prob.run(x_end)
    runs[op] = prob

# decompose final spectrum in z (average over y)
kz = np.fft.fftshift(np.fft.fftfreq(Nz, dz)) * 2 * np.pi
spectrum = {}
for op, prob in runs.items():
    U = np.fft.fftshift(np.fft.fft(prob.u.mean(axis=0)))
    spectrum[op] = U

amp_err, pha_err = {}, {}
for op in ["pade22", "pade21"]:
    ae, pe = [], []
    for th in thetas:
        kz0 = k0 * np.sin(np.radians(th))
        i = np.argmin(np.abs(kz - kz0))
        a = spectrum[op][i] / (spectrum["exact"][i] + 1e-30)
        ae.append(np.abs(a) - 1.0)
        pe.append(np.degrees(np.angle(a)))
    amp_err[op] = ae
    pha_err[op] = pe

# analytic reference operator errors
oe = operator_error_table(np.linspace(0, np.sin(np.radians(35)) ** 2, 400))
ang = oe["theta_deg"]

results = dict(
    setup=dict(f_GHz=f_GHz, lambda_m=lam, dz_m=dz, Nz=Nz, dx_m=dx, x_end_m=x_end,
               angles_deg=thetas.tolist(),
               nyquist_angle_deg=float(np.degrees(np.arcsin(lam/(2*dz))))),
    measured=dict(
        pade22_amp_rel_err=amp_err["pade22"], pade22_phase_err_deg=pha_err["pade22"],
        pade21_amp_rel_err=amp_err["pade21"], pade21_phase_err_deg=pha_err["pade21"]),
    analytic_ref=dict(theta_deg=ang.tolist(),
                      pade22_rel_err=oe["pade22"].tolist(),
                      pade21_rel_err=oe["pade21"].tolist()),
    runtime_s=time.time()-t0,
)
# pass: at 30 deg, R22 phase error < 2 deg and amplitude rel err < 1%
i30 = list(thetas).index(30)
results["metrics"] = dict(
    pade22_phase_at_30_deg=pha_err["pade22"][i30],
    pade22_amp_at_30_rel=amp_err["pade22"][i30],
    pade21_phase_at_30_deg=pha_err["pade21"][i30],
)
results["pass"] = bool(abs(pha_err["pade22"][i30]) < 2.0 and abs(amp_err["pade22"][i30]) < 0.01)
save_json("bench6_wide_angle", results)
runs["exact"].save(f"{DATA}/bench6_field_exact.npz")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(thetas, pha_err["pade22"], "o-", label="Padé[2/2] measured")
ax[0].plot(thetas, pha_err["pade21"], "s-", label="Padé[2/1] measured")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xlabel("angle [deg]"); ax[0].set_ylabel("phase error vs exact [deg]")
ax[0].set_title("(a) propagated phase error")
ax[0].legend(); ax[0].grid(True, alpha=0.3)
ax[1].semilogy(ang, oe["pade22"], "-", label="Padé[2/2]")
ax[1].semilogy(ang, oe["pade21"], "-", label="Padé[2/1]")
ax[1].semilogy(ang, oe["narrow"], "--", label="narrow-angle")
ax[1].set_xlabel("angle [deg]"); ax[1].set_ylabel("|R(q)-sqrt(1-q)|/sqrt(1-q)")
ax[1].set_title("(b) analytic operator error")
ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
savefig("bench6_wide_angle")

print(f"BENCH6 done. R22 @30deg: phase {pha_err['pade22'][i30]:.3f} deg, "
      f"amp {amp_err['pade22'][i30]:.2e}; PASS={results['pass']} ({time.time()-t0:.1f}s)")
print("  phase R22:", [f"{p:.2f}" for p in pha_err["pade22"]])
print("  phase R21:", [f"{p:.2f}" for p in pha_err["pade21"]])
