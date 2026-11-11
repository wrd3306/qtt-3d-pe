"""Build the deterministic QTT training fields with the TRUSTED solver (PE3D).

3-D tensor U[f, y, z]:
  * f = 16 frequencies, log-spaced 0.5 .. 3.0 GHz  (2^4)
  * y = 64 cross-range cells, dy = 5 m             (2^6)
  * z = 128 height cells,     dz = 2 m             (2^7)
  * PEC ground (horizontal-pol Dirichlet, image method), top + y-side sponge,
    standard atmosphere M = 301 + 0.118 z, Gaussian aperture at y=0, z=30 m.

4-D tensor U[f, z0, y, z]: same + source-height parameter z0 in {20,30,40,50} m.

NOTE (REBUILD_SPEC sec.7): this coarse grid has a NARROW effective angular band
because dz = 2 m: sin(theta_max) = lambda/(2 dz)  ->  1 GHz: 4.30 deg,
3 GHz: 1.43 deg.  This is a near-axis compression study, NOT a wide-angle claim.

Outputs (data/qtt/):
  fields_3d.npz : U3d (16,64,128) complex128 + grids + timing
  fields_4d.npz : U4d (16,4,64,128) complex128 + grids + timing
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from solver.propagator import PE3D
from solver.medium import StandardAtmosphere
from solver.antenna import gaussian_aperture
from solver.constants import C_LIGHT

DATA = os.path.join(ROOT, "data", "qtt")
os.makedirs(DATA, exist_ok=True)

# ---- physical setup (coarse compression grid) ----
DX = 2.0
DY = 5.0
DZ = 2.0
NY = 64
NZ = 128
Y_MAX = 160.0
Z_MAX = 256.0
X_END = 4000.0
SPONGE_TOP = 40.0
SPONGE_Y = 16.0
SPONGE_STR = 8.0
SIGMA_Y = 20.0
SIGMA_Z = 5.0
Z0_3D = 30.0
N_F = 16
F_LO, F_HI = 0.5, 3.0           # GHz
Z0_GRID = np.array([20.0, 30.0, 40.0, 50.0])  # 4 = 2^2


def run_one(f_GHz, z0):
    prob = PE3D(f_GHz=f_GHz, dx_m=DX, dy_m=DY, dz_m=DZ, Ny=NY, Nz=NZ,
                y_max_m=Y_MAX, z_max_m=Z_MAX, ground="horizontal",
                operator="pade22",
                n_profile=StandardAtmosphere(M0=301.0, slope_M_per_m=0.118),
                sponge_top_m=SPONGE_TOP, sponge_y_m=SPONGE_Y,
                sponge_strength=SPONGE_STR)
    u0 = gaussian_aperture(prob.y_phys, prob.z_phys, y0=0.0, z0=z0,
                           sigma_y=SIGMA_Y, sigma_z=SIGMA_Z, k0=prob.k0, tilt_deg=0.0)
    prob.initialize(u0)
    prob.run(x_end_m=X_END)
    return prob.field().copy(), prob


def main():
    freqs_ghz = np.geomspace(F_LO, F_HI, N_F)
    # ---- 3D tensor ----
    U3 = np.empty((N_F, NY, NZ), dtype=np.complex128)
    t_pe_3d = []
    for i, f in enumerate(freqs_ghz):
        t0 = time.time()
        u, prob = run_one(float(f), Z0_3D)
        dt = time.time() - t0
        t_pe_3d.append(dt)
        U3[i] = u
        print("3D f=%.3f GHz  t_pe=%.3fs  |u|max=%.4f" % (f, dt, abs(u).max()))
    y = prob.y_phys.copy(); z = prob.z_phys.copy()

    np.savez_compressed(
        os.path.join(DATA, "fields_3d.npz"),
        U=U3, freqs_ghz=freqs_ghz, y=y, z=z,
        setup=json.dumps(dict(dx=DX, dy=DY, dz=DZ, Ny=NY, Nz=NZ, y_max=Y_MAX,
                              z_max=Z_MAX, x_end=X_END, sigma_y=SIGMA_Y,
                              sigma_z=SIGMA_Z, z0=Z0_3D, ground="horizontal",
                              operator="pade22", sponge_top=SPONGE_TOP,
                              sponge_y=SPONGE_Y, sponge_strength=SPONGE_STR)))
    print("saved fields_3d.npz", U3.shape, U3.dtype,
          "mean t_pe=%.3fs" % np.mean(t_pe_3d))

    # ---- 4D tensor: add source-height parameter ----
    U4 = np.empty((N_F, Z0_GRID.size, NY, NZ), dtype=np.complex128)
    t_pe_4d = []
    for i, f in enumerate(freqs_ghz):
        for j, z0 in enumerate(Z0_GRID):
            t0 = time.time()
            u, _ = run_one(float(f), float(z0))
            t_pe_4d.append(time.time() - t0)
            U4[i, j] = u
        print("4D f=%.3f GHz done" % f)
    np.savez_compressed(
        os.path.join(DATA, "fields_4d.npz"),
        U=U4, freqs_ghz=freqs_ghz, z0_m=Z0_GRID, y=y, z=z,
        setup=json.dumps(dict(dx=DX, dy=DY, dz=DZ, Ny=NY, Nz=NZ, y_max=Y_MAX,
                              z_max=Z_MAX, x_end=X_END, sigma_y=SIGMA_Y,
                              sigma_z=SIGMA_Z, ground="horizontal",
                              operator="pade22", sponge_top=SPONGE_TOP,
                              sponge_y=SPONGE_Y, sponge_strength=SPONGE_STR)))
    print("saved fields_4d.npz", U4.shape, U4.dtype,
          "total PE runs=%d" % len(t_pe_4d))

    # timing record for break-even
    timing = dict(t_pe_3d_each=t_pe_3d, t_pe_3d_mean=float(np.mean(t_pe_3d)),
                  t_pe_4d_total=float(np.sum(t_pe_4d)),
                  freqs_ghz=freqs_ghz.tolist(),
                  nyquist_angle_deg={str(fr): float(np.degrees(np.arcsin(
                      (C_LIGHT/(fr*1e9))/(2*DZ)))) for fr in freqs_ghz})
    with open(os.path.join(DATA, "dataset_timing.json"), "w", encoding="utf-8") as fp:
        json.dump(timing, fp, indent=2)
    print("DONE dataset")


if __name__ == "__main__":
    main()
