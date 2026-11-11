"""Supplementary figure: show the HONEST failure mode -- on-grid TT reconstruction
is exact, but off-grid frequency linear interpolation is poor because the complex
field oscillates rapidly in k0 across the 16-point log grid."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qtt.model import QTTModel
from solver.propagator import PE3D
from solver.medium import StandardAtmosphere
from solver.antenna import gaussian_aperture

DATA = os.path.join(ROOT, "data", "qtt"); FIGS = os.path.join(ROOT, "figures", "qtt")
d3 = np.load(os.path.join(DATA, "fields_3d.npz"), allow_pickle=False)
U, freqs_ghz, y, z = d3["U"], d3["freqs_ghz"], d3["y"], d3["z"]

model = QTTModel(eps=1e-3, name="qtt_demo").fit(U, {"frequency_hz": freqs_ghz*1e9})

def run_direct(f_GHz):
    prob = PE3D(f_GHz=f_GHz, dx_m=2.0, dy_m=5.0, dz_m=2.0, Ny=64, Nz=128,
                y_max_m=160.0, z_max_m=256.0, ground="horizontal", operator="pade22",
                n_profile=StandardAtmosphere(M0=301.0, slope_M_per_m=0.118),
                sponge_top_m=40.0, sponge_y_m=16.0, sponge_strength=8.0)
    u0 = gaussian_aperture(prob.y_phys, prob.z_phys, y0=0.0, z0=30.0,
                           sigma_y=20.0, sigma_z=5.0, k0=prob.k0, tilt_deg=0.0)
    prob.initialize(u0); prob.run(x_end_m=4000.0)
    return prob.field()

# (a) on-grid: reconstruct at training index 7 -> should overlay
# (b) off-grid: direct at 1.25 GHz vs predicted
igrid = 7
u_grid_rec = model.predict({"frequency_hz": freqs_ghz[igrid]*1e9})
fhold = 1.25
u_dir = run_direct(fhold)
u_pred = model.predict({"frequency_hz": fhold*1e9})
iy = 32

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(z, np.abs(U[igrid, iy, :]), "k-", label="dense PE (grid)")
ax[0].plot(z, np.abs(u_grid_rec[iy, :]), "r--", label="TT predict (grid)")
ax[0].set_xlim(0, 220); ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("|u|")
ax[0].set_title("(a) on-grid f=%.2f GHz (faithful)" % freqs_ghz[igrid]); ax[0].legend()
ax[1].plot(z, np.abs(u_dir[iy, :]), "k-", label="direct PE")
ax[1].plot(z, np.abs(u_pred[iy, :]), "r--", label="TT linear interp")
ax[1].set_xlim(0, 220); ax[1].set_xlabel("z [m]"); ax[1].set_ylabel("|u|")
ax[1].set_title("(b) off-grid f=%.2f GHz (interp fails)" % fhold); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "qtt_heldout_compare.png"), dpi=600)
plt.savefig(os.path.join(FIGS, "qtt_heldout_compare.pdf")); plt.close()
print("saved qtt_heldout_compare")
