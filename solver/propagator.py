"""3D wide-angle PE propagator: Strang symmetric split-step Fourier (SSFT).

Governing equation:
    u_x = i/(2 k0) grad_perp^2 u + i k0 (n^2 - 1)/2 u
after the outgoing factorization of Helmholtz, with wide-angle spectral
multiplier exp[i k0 (R(q) - 1) dx], q = k_perp^2/k0^2, R in {exact sqrt(1-q),
Padé[2/2], Padé[2/1]}.

Per range step dx (Strang symmetric splitting), with the refraction
generator L_r = i k0 (n^2-1)/2 so each Strang half carries L_r dx/2:
    u <- exp[i k0 (n_eff^2 - 1) dx/4] u          (refraction half-step)
    u <- F^{-1}{ exp[i k0 (R(q)-1) dx] F{u} }   (diffraction step)
    u <- exp[i k0 (n_eff^2 - 1) dx/4] u          (refraction half-step)
    u <- sponge taper(y sides, top)               (absorbing layer)
The two halves close on exp[i k0 (n^2-1) dx/2], the correct one-step
refraction of the narrow-angle equation.

Ground PEC by the *image method*:
    horizontal polarization (H-pol, E_y): Dirichlet  u(z=0) = 0 -> antisymmetric mirror
    vertical   polarization (V-pol, H_y): Neumann  du/dz|_0 = 0 -> symmetric mirror
Implemented as an extended-z FFT grid (u_ext[-z] = +/- u_ext[+z]); NOT a hard mask.

Complex128 field carries phase everywhere; dB is display-only.
"""
from __future__ import annotations
import numpy as np
from .constants import C_LIGHT
from .operators import diffraction_kernel
from .terrain import FLAT


class PE3D:
    """3D parabolic-equation slice propagator.  Field u is complex128 (Ny, Nz)."""

    def __init__(self, f_GHz=1.0, dx_m=2.0, dy_m=2.0, dz_m=2.0,
                 Ny=64, Nz=256, y_max_m=64.0, z_max_m=512.0,
                 ground="horizontal", operator="pade22",
                 n_profile=None, terrain=None,
                 sponge_top_m=40.0, sponge_y_m=16.0, sponge_strength=8.0):
        assert ground in ("horizontal", "vertical", "none")
        assert operator in ("exact", "pade22", "pade21", "narrow")
        self.f_GHz = float(f_GHz)
        self.k0 = 2.0 * np.pi * f_GHz * 1.0e9 / C_LIGHT
        self.lam = C_LIGHT / (f_GHz * 1.0e9)
        self.dx = float(dx_m)
        self.ground = ground
        self.operator = operator
        self.terrain = terrain if terrain is not None else FLAT
        self.n_profile = n_profile  # callable (x, y, z) -> n

        # --- grids (physical) ---
        self.Ny = int(Ny); self.Nz = int(Nz)
        self.dy = float(dy_m); self.dz = float(dz_m)
        self.y = np.fft.fftfreq(self.Ny, d=self.dy) * 2.0 * np.pi  # k_y
        self.y_phys = np.arange(self.Ny) * self.dy - y_max_m      # periodic [-Ymax, Ymax)
        if self.ground == "none":
            # free space: centered periodic z grid so sponges sit on both ends
            self.z_phys = (np.arange(self.Nz) - self.Nz // 2) * self.dz
        else:
            self.z_phys = np.arange(self.Nz) * self.dz            # PEC ground at z=0
        self.y_max = float(y_max_m)
        self.z_max = float(z_max_m)

        # --- extended z grid for the image method (if a ground exists) ---
        self._has_ground = self.ground != "none"
        if self._has_ground:
            self.N_ext = 2 * self.Nz
        else:
            self.N_ext = self.Nz
        kz = np.fft.fftfreq(self.N_ext, d=self.dz) * 2.0 * np.pi
        ky = self.y
        self._q = (ky[:, None] ** 2 + kz[None, :] ** 2) / self.k0 ** 2

        # --- spectral diffraction multiplier (constant medium; rebuilt per step
        #     only if operator or dx changes) ---
        self._kernel = diffraction_kernel(self._q, self.k0, self.dx, operator)

        # --- sponge masks (raised cosine-like quadratic taper) ---
        self.sponge_top = float(sponge_top_m)
        self.sponge_y = float(sponge_y_m)
        self.sponge_strength = float(sponge_strength)
        self._build_sponge()

        # --- state ---
        self.u = np.zeros((self.Ny, self.Nz), dtype=np.complex128)
        self.x = 0.0
        self.n_steps_done = 0

    # ------------------------------------------------------------------ grid utils
    def _build_sponge(self):
        """Sponge taper.  t=0 in the interior (factor 1), t=1 at the outer edge
        (factor exp(-strength)); quadratic raised profile exp(-strength t^2)."""
        y = self.y_phys
        z = self.z_phys
        sy = np.ones(self.Ny)
        if self.sponge_y > 0:
            # periodic y; sponge occupies the outer L_y of each end
            d = np.abs(y) - (np.abs(y).max() - self.sponge_y)
            t = np.clip(d / self.sponge_y, 0.0, 1.0)
            sy = np.exp(-self.sponge_strength * t ** 2)
        sz = np.ones(self.Nz)
        if self.sponge_top > 0:
            if self._has_ground:
                # z in [0, z_max]; sponge occupies z in [z_max-L, z_max]
                d = z - (self.z_phys.max() - self.sponge_top)
            else:
                # centered periodic z; sponges on both ends
                half = np.abs(z).max()
                d = np.abs(z) - (half - self.sponge_top)
            t = np.clip(d / self.sponge_top, 0.0, 1.0)
            sz = np.exp(-self.sponge_strength * t ** 2)
        self._sponge = np.outer(sy, sz).astype(np.float64)

    # ------------------------------------------------------------------ I/O
    def initialize(self, u0):
        """Set the aperture field u0 with shape (Ny, Nz), complex128."""
        assert np.shape(u0) == (self.Ny, self.Nz), (np.shape(u0), (self.Ny, self.Nz))
        self.u = np.asarray(u0, dtype=np.complex128).copy()
        self.x = 0.0
        self.n_steps_done = 0
        if self.ground == "horizontal":
            self.u[:, 0] = 0.0  # Dirichlet at PEC ground

    # ------------------------------------------------------------------ stepping
    def _refraction_factor(self):
        """exp[i k0 (n_eff^2 - 1) dx/4] on the physical grid (Strang half-step).

        The PDE refraction generator is i k0 (n^2-1)/2; half of it over one
        range step is i k0 (n^2-1) dx/4. Two applications close on the correct
        one-step phase exp[i k0 (n^2-1) dx/2].
        """
        if self.n_profile is None:
            return 1.0
        slope = self.terrain.slope_x(self.x, 0.0)
        zeta = self.z_phys[None, :] - self.terrain.height(self.x, 0.0)
        zeta = np.clip(zeta, 0.0, None)
        n = np.asarray(self.n_profile(self.x, self.y_phys, self.z_phys), dtype=float)
        n2_eff = n ** 2 - 2.0 * zeta * slope
        return np.exp(0.25j * self.k0 * (n2_eff - 1.0) * self.dx)

    def _extend(self, u):
        """Physical (Ny, Nz) -> extended (Ny, N_ext) with image symmetry."""
        if not self._has_ground:
            return u
        uext = np.empty((self.Ny, self.N_ext), dtype=np.complex128)
        uext[:, self.Nz:2 * self.Nz] = u
        sigma = -1.0 if self.ground == "horizontal" else +1.0
        # mirror j=1..Nz-1 at extended index Nz-j
        uext[:, self.Nz - 1:0:-1] = sigma * u[:, 1:self.Nz]
        uext[:, self.Nz] = u[:, 0]
        # periodic wrap of the mirror (extended index 0 == index N_ext by DFT
        # periodicity): without this, column 0 is left uninitialised and reads
        # arbitrary heap garbage, causing non-deterministic field blow-ups.
        uext[:, 0] = sigma * u[:, 0]
        if self.ground == "horizontal":
            uext[:, self.Nz] = 0.0
        return uext

    def _extract(self, uext):
        if not self._has_ground:
            return uext
        u = uext[:, self.Nz:2 * self.Nz].copy()
        if self.ground == "horizontal":
            u[:, 0] = 0.0
        return u

    def step(self):
        """Advance one dx.  Strang symmetric split-step."""
        rf = self._refraction_factor()
        self.u = self.u * rf
        uext = self._extend(self.u)
        U = np.fft.fft2(uext)
        U = U * self._kernel
        uext = np.fft.ifft2(U)
        self.u = self._extract(uext)
        self.u = self.u * rf
        self.u = self.u * self._sponge
        self.x += self.dx
        self.n_steps_done += 1

    def run(self, x_end_m):
        n_more = int(round((float(x_end_m) - self.x) / self.dx))
        for _ in range(n_more):
            self.step()

    # ------------------------------------------------------------------ queries
    def field(self):
        """Current complex field slice (Ny, Nz), complex128, phase preserved."""
        return self.u

    def grid(self):
        return {"x": self.x, "y": self.y_phys, "z": self.z_phys,
                "dy": self.dy, "dz": self.dz, "dx": self.dx}

    def save(self, path, extra_meta=None):
        import numpy as np
        meta = dict(f_GHz=self.f_GHz, k0=self.k0, lam=self.lam, dx=self.dx,
                    dy=self.dy, dz=self.dz, Ny=self.Ny, Nz=self.Nz,
                    ground=self.ground, operator=self.operator,
                    x_end=self.x, n_steps=self.n_steps_done)
        if extra_meta:
            meta.update(extra_meta)
        np.savez_compressed(path, u=self.u, y=self.y_phys, z=self.z_phys,
                            x_end=np.float64(self.x), meta=str(meta))
        return path


__all__ = ["PE3D"]
