# QTT Compression of the Terminal-Slice Complex Field of a 3-D PE Solver — Reproducibility Code

This repository contains the code that reproduces the solver benchmarks, the
quantics tensor-train (QTT) compression study, the external baselines, and the
six propagation scenarios reported in the associated manuscript. The code
builds a deterministic three-dimensional (3-D) split-step Fourier (SSFT)
parabolic-equation (PE) solver and applies a TT-SVD QTT decomposition to the
**terminal-slice complex field** (not to a decibel display array). No result is
hard-coded, no fitted or interpolated surrogate stands in for a computed
field, and failures are reported with their physical cause rather than
hidden.

> **Repository.** The public source code is hosted on GitHub at
> [https://github.com/wrd3306/qtt-3d-pe](https://github.com/wrd3306/qtt-3d-pe)
> and is archived on Zenodo (version v1.0.0) under the persistent DOI
> [10.5281/zenodo.24140022](https://doi.org/10.5281/zenodo.24140022).

## 1. Repository layout

| Path | Contents |
|---|---|
| `solver/` | Trusted 3-D SSFT PE library: medium, operators, propagator, terrain, antenna, field I/O |
| `qtt/` | Binary quantization and complex TT-SVD, uniform `QTTModel` interface, phase-aware interpolation |
| `baselines/` | Direct interpolation, POD/ROM, truncated SVD, and Kuttler–Dockery narrow-angle PE |
| `scenarios/` | Refractivity profiles and the six-scenario head-to-head builder |
| `scripts/` | Benchmark, dataset, decomposition, baseline, and scenario drivers; smoke test |
| `data/results/` | Recorded scalar result snapshot (`.json`) used in the manuscript; large field archives are regenerated, not shipped |
| `figures/` | Generated vector PDFs and 600 dpi PNGs (created on demand; not shipped) |

## 2. Environment

Verified on a single workstation: Intel Core i5-14600KF (14 cores, 20 logical
processors), 64 GB RAM, 64-bit Windows 10 Enterprise LTSC (build 19044),
Python 3.10.11. Pinned dependencies:

```
numpy==2.2.6
scipy==1.15.3
sympy==1.13.1
matplotlib==3.10.9
Pillow==12.2.0
```

Install:

```
python -m pip install -r requirements.txt
```

`sympy` is used by the operator verification path; `Pillow` is only needed
when inspecting figure DPI.

## 3. Quick self-check (no dataset required)

The smoke test exercises the Padé wide-angle operators and a tiny free-space
propagation run without generating any dataset:

```
python scripts/smoke_test.py
```

It prints the operator relative errors and the beam-center index and ends with
`SMOKE OK`.

## 4. One-click reproduction

From the repository root:

```
python reproduce.py
```

This runs, in order, and aborts on the first failure:

1. solver benchmarks (`scripts/run_all_benchmarks.py`, bench1 … bench9);
2. QTT dataset generation (`scripts/qtt_dataset.py`);
3. QTT decomposition and accounting (`scripts/qtt_run.py`);
4. phase-aware versus complex-linear held-out comparison
   (`scripts/phase_aware_run.py`);
5. external baselines (`scripts/run_baselines.py`);
6. six-scenario head-to-head (`scripts/run_scenarios.py`).

A transcript is written to `data/reproduce_run.log`. Individual stages can
also be run directly from `scripts/`.

### Regenerated data and figures

No `.npz` field archive is stored in this release. The drivers create their
own `data/<stage>/` and `figures/<stage>/` directories and recompute every
field, TT rank, storage count, and error metric from scratch. The recorded
scalar results shipped under `data/results/` are a reference snapshot for
cross-checking the manuscript numbers; a fresh run overwrites the working
`data/` directory, not this snapshot.

All fields, TT ranks, storage counts, and error metrics are deterministic and
reproduce bit-for-bit. Only wall-clock timings vary with hardware and system
load; the reported timings are therefore approximate cost estimates, while the
ranks, storage counts, compression ratios, and errors are machine-invariant.

## 5. Key results (reproducible from this code)

### Solver validation

- Free-space spreading: rel. $L_2 = 7.75\times10^{-4}$, far-field exponent
  $-0.489$ (theory $-0.5$).
- PEC mirror (H / V): $2.84\times10^{-3}$ / $2.96\times10^{-3}$.
- Knife edge: max. landmark error $0.197$ dB.
- Grid convergence: log-log slopes $1.54$ ($dx$), $2.88$ ($dz$), $8.70$ ($dy$).
- Sponge reflection: $-21.2$ dB design, best measured $-91.4$ dB.
- Wide angle (fine grid, $\Delta z=\lambda$): Padé[2/2] phase error
  $0.392^\circ$ and amplitude $-1.7\times10^{-5}$ at $30^\circ$.
- True 3-D free-space Gaussian: rel. $L_2 = 6.44\times10^{-6}$ against the
  closed-form product of two one-dimensional beam kernels.
- Terrain: flat-ground regression $4.38\times10^{-5}$, effective-refractivity
  factor probe $4.97\times10^{-16}$, Ehrenfest centroid agreement
  $6.70\times10^{-11}$.
- Airy surface duct: rel. $L_2 = 2.86\times10^{-2}$, amplitude
  $2.09\times10^{-2}$, phase residual $0.022$ rad.

### Honest QTT compression, (16, 64, 128) complex field

At the operating truncation $\varepsilon=10^{-3}$:

- peak rank 49;
- 14,762 complex scalars against 131,072 dense complex values;
- compression ratio **8.88**;
- in-grid rel. $L_2 = 5.62\times10^{-4}$, path-loss RMS $0.13$ dB
  (95th percentile $0.24$ dB, maximum $1.85$ dB).

The sweep spans CR $0.43$ at $\varepsilon=0$ (full rank, TT rank overhead)
to $26.4$ at $\varepsilon=3\times10^{-2}$.

Four-dimensional extension adding source height
$(16,4,64,128)=524{,}288$: peak rank 99, 71,900 scalars, CR **7.29**,
rel. $L_2 = 6.19\times10^{-4}$, $0.13$ dB RMS at the same $\varepsilon$.

### Cost break-even

Offline: $18.31$ s for the 16 training PE solves plus $0.10$ s for the TT-SVD
decomposition ($18.41$ s total). One online TT prediction takes $0.0029$ s; one
direct PE solve takes $0.728$ s. Equating cumulative costs gives the
break-even $Q\approx25.4$ queries. In storage, the 14,762 core scalars equal
1.80 single-frequency field slices. These timings were measured on the
workstation in Section 2; on different hardware the wall-clock values change
while the query count shifts only modestly.

### Off-grid failure and phase-aware method

On the 16-point training grid the coherent complex field cannot be predicted
off-grid because the Lloyd-mirror fringes are under-sampled. The mechanism
check with $N_f=16/32/64$ training frequencies reduces the held-out complex
error to $0.689 / 0.122 / 0.037$. The phase-aware interpolation separates
magnitude and unwrapped phase and cuts the off-grid path-loss error to
$0.42$ dB mean (raw complex interpolation $4.92$ dB mean), but it retains the
dense 131,072-value field and leaves the absolute complex phase off-grid; it
is a precision method, not a compression method.

### External baselines and scenarios

Four reference methods (direct interpolation, POD/ROM, truncated SVD,
Kuttler–Dockery narrow-angle PE) are run on the same fields under a uniform
`fit/predict/storage` interface, and the comparison is repeated over six
propagation scenarios (free space, evaporation duct, Gaussian hill, two
ridges, surface duct, elevated duct).

## 6. Module overview

- `solver/constants.py` — speed of light, Earth radius.
- `solver/medium.py` — M-profile to refractive index; standard atmosphere, layered duct.
- `solver/operators.py` — strict Padé[2/2], Padé[2/1], exact $\sqrt{1-q}$ spectral operators.
- `solver/propagator.py` — `PE3D` class (Strang-split SSFT, image-method ground, sponge layers).
- `solver/terrain.py` — terrain-following mapping and effective-refractivity correction.
- `solver/antenna.py` — physical Gaussian aperture (vertical beamwidth conversion).
- `solver/fieldio.py` — complex-field `.npz` and result `.json` I/O.
- `qtt/tt.py` — quantization, dequantization, complex TT-SVD, reconstruction, storage accounting.
- `qtt/model.py` — uniform `QTTModel` interface over the TT cores.
- `qtt/metrics.py` — validity mask, field error metrics, direct PE reference field.
- `qtt/phase_interp.py` — phase-aware held-out interpolation.
- `baselines/` — direct field interpolation, POD-ROM, truncated SVD, Kuttler–Dockery PE.
- `scenarios/` — refractivity profiles and the six-scenario builder.

## 7. License

This release is provided under the MIT License (see `LICENSE`).

## 8. Citing

If you use this code, cite the associated manuscript. The archived software
(version v1.0.0) can be cited as:

> R. Wang, M. Wang, Y. Ma, and R. Zhang, "qtt-3d-pe: Reproducible code for a
> systematic assessment of quantics tensor-train compression for 3-D
> parabolic-equation propagation fields," Zenodo, 2026,
> DOI: 10.5281/zenodo.24140022.

The full bibliographic entry for the manuscript is added after publication.
