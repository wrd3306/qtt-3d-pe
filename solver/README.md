# solver — 3D Wide-Angle PE (SSFT) Library

可信的三维宽角抛物方程（PE）分步傅里叶（SSFT）求解库。复场 complex128 保留相位；
dB 仅用于展示。基准报告见 `docs/solver_benchmark_report.md`（六个基准全部 PASS）。

## 依赖

- Python 3.10
- numpy, scipy, matplotlib

## 一键复跑基准

```
cd direction_3d_qtt_rebuild
py -3.10 scripts/run_all_benchmarks.py
```

单个基准：`py -3.10 scripts/bench1_free_space.py` … `bench6_wide_angle.py`。
产物：`data/solver/*.npz`（复场）、`data/solver/*.json`（机器可读结果）、
`figures/solver/*.{png,pdf}`（600 dpi + 矢量）。

## 公共 API（供 qtt / scenarios 模块导入）

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
    terrain=FLAT,                 # 或 GaussianHill(...)/SlopePlane(...)
    sponge_top_m=40.0, sponge_y_m=16.0, sponge_strength=8.0,
)
prob.initialize(gaussian_aperture(prob.y_phys, prob.z_phys,
                                 z0=30.0, sigma_z=3.0, k0=prob.k0, tilt_deg=0.0))
prob.run(x_end_m=20000.0)
u = prob.field()                  # complex128 (Ny, Nz)，相位保留
prob.save("data/solver/myfield.npz")
```

## 模块结构

- `constants.py`：c、Re。
- `medium.py`：M 剖面 → n；StandardAtmosphere / LayeredDuct。
- `operators.py`：严格 Padé[2/2]、Padé[2/1]、精确 √(1−q) 谱算子。
- `propagator.py`：PE3D 类（Strang 分裂 SSFT、镜像法地面、sponge）。
- `terrain.py`：terrain-following 映射（非 staircase）。
- `antenna.py`：物理高斯口径（含垂直波束宽度换算）。
- `fieldio.py`：复场 .npz / 结果 .json 读写。
