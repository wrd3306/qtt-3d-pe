# solver — 3-D Wide-Angle PE (SSFT) Library

A trusted three-dimensional wide-angle parabolic-equation (PE) solver based on
the split-step Fourier (SSFT) method. The field is always stored as
`complex128` and keeps its phase; decibel values are display-only.

## Dependencies

- Python 3.10 with NumPy, SciPy, and Matplotlib (see `requirements.txt`).

## Running the benchmarks

From the repository root:

```
python scripts/run_all_benchmarks.py
```

Individual benchmarks are `scripts/bench1_free_space.py` through
`scripts/bench9_duct.py`. Each run writes its machine-readable result to
`data/solver/*.json` and renders figures to `figures/solver/`.

## Public API

```python
from solver import PE3D, StandardAtmosphere, LayeredDuct, n_from_M
from solver import GaussianHill, SlopePlane, FLAT
from solver import gaussian_aperture, sigma_z_from_beamwidth
from solver import pade22, pade21, exact_sqrt, diffraction_kernel

prob = PE3D(
    f_GHz=1.0, dx_m=1.0, dy_m=2.0, dz_m=0.25,
    Ny=64, Nz=256, y_max_m=64.0, z_max_m=512.0,
    ground="horizontal",          # horizontal=H-pol Dirichlet / vertical=V-pol Neumann / none
    operator="pade22",            # pade22 | pade21 | exact | narrow
    n_profile=StandardAtmosphere(M0=301.0, slope_M_per_m=0.118),
    terrain=FLAT,                 # or GaussianHill(...)/SlopePlane(...)
    sponge_top_m=40.0, sponge_y_m=16.0, sponge_strength=8.0,
)
prob.initialize(gaussian_aperture(prob.y_phys, prob.z_phys,
                                 z0=30.0, sigma_z=3.0, k0=prob.k0, tilt_deg=0.0))
prob.run(x_end_m=20000.0)
u = prob.field()                  # complex128 (Ny, Nz), phase preserved
```

## Modules

- `constants.py` — physical constants (speed of light, Earth radius).
- `medium.py` — M-profile to refractive index; `StandardAtmosphere`, `LayeredDuct`.
- `operators.py` — strict Padé[2/2], Padé[2/1], and the exact sqrt(1-q) spectral operator.
- `propagator.py` — the `PE3D` class (Strang-split SSFT, image-method ground, sponge).
- `terrain.py` — terrain-following mapping (not a staircase).
- `antenna.py` — physical Gaussian aperture with vertical-beamwidth conversion.
- `fieldio.py` — complex-field `.npz` and result `.json` input/output.
