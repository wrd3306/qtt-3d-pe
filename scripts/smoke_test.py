"""Smoke test: operator accuracy + tiny free-space run. Not a benchmark."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from solver.operators import pade22, pade21, exact_sqrt
from solver.propagator import PE3D
from solver.antenna import gaussian_aperture

# --- operator table vs sympy-verified numbers ---
q = np.array([0.5])
ref = exact_sqrt(q)[0]
r22 = pade22(q)[0]; r21 = pade21(q)[0]
print(f"q=0.5: sqrt(1-q)={ref:.6f}")
print(f"  R22={r22:.6f}  rel err={(r22-ref)/ref*100:+.5f}%  (expect -0.00210%)")
print(f"  R21={r21:.6f}  rel err={(r21-ref)/ref*100:+.5f}%  (expect +0.0208%)")

# --- tiny free-space run, exact operator, ground none ---
prob = PE3D(f_GHz=1.0, dx_m=1.0, dy_m=2.0, dz_m=1.0, Ny=64, Nz=256,
            y_max_m=64.0, z_max_m=128.0, ground="none", operator="exact",
            sponge_top_m=30.0, sponge_y_m=15.0)
u0 = gaussian_aperture(prob.y_phys, prob.z_phys, z0=0.0, sigma_y=3.0, sigma_z=3.0, k0=prob.k0)
prob.initialize(u0)
prob.run(200.0)
u = prob.field()
print(f"ran x={prob.x} m, max|u|={np.abs(u).max():.4e}, dtype={u.dtype}")
# beam should stay centered, width grows ~ sqrt(1+(x/(k sigma^2))^2)
yc = np.abs(u[:, :]).max(axis=1).argmax()
print(f"beam-center y index={yc} (expect ~{64//2}), center y={prob.y_phys[yc]:.1f} m")
print("SMOKE OK")
