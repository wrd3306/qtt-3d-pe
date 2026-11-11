"""Formal comparison of the two out-of-grid prediction strategies on the same
16-point training grid, with real PE ground truth:

  A) complex-field multilinear interpolation (the QTTModel predict path)
  B) phase-aware interpolation (qtt.phase_interp.PhaseAwareInterp)

Mask, amplitude floor, and error accounting are imported from qtt.metrics, so
they are identical to scripts/qtt_run.py. Writes a machine-readable JSON and a
600 dpi PNG + vector PDF comparison figure. No hard-coded numbers.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qtt import QTTModel, PhaseAwareInterp
from qtt.metrics import (valid_mask, field_errors, direct_pe_field,
                         NY, NZ, N_F)

DATA = os.path.join(ROOT, "data", "qtt")
FIGS = os.path.join(ROOT, "figures", "qtt")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans",
})

HELD_OUT = [0.75, 1.25, 1.75, 2.5, 2.85]
EPS_OP = 1e-3


def main():
    d = np.load(os.path.join(DATA, "fields_3d.npz"), allow_pickle=False)
    U = d["U"]
    freqs = d["freqs_ghz"]
    y, z = d["y"], d["z"]
    m = valid_mask(y, z)

    # A) QTT surrogate (complex multilinear interpolation on TT-reconstructed field)
    qtt_model = QTTModel(eps=EPS_OP, name="qtt_3d_freq")
    qtt_model.fit(U, {"frequency_hz": freqs * 1e9})
    # B) phase-aware interpolation
    phase_model = PhaseAwareInterp()
    phase_model.fit(U, {"frequency_hz": freqs * 1e9})

    rows = []
    for fq in HELD_OUT:
        ref = direct_pe_field(float(fq))
        u_complex = qtt_model.predict({"frequency_hz": fq * 1e9})
        u_phase = phase_model.predict({"frequency_hz": fq * 1e9})
        e_c = field_errors(ref[None], u_complex[None], m)
        e_p = field_errors(ref[None], u_phase[None], m)
        rows.append(dict(frequency_hz=float(fq * 1e9),
                         complex_linear=dict(rel_l2=e_c["rel_l2"], db_rms=e_c["db_rms"],
                                             db_p95=e_c["db_p95"], db_max=e_c["db_max"]),
                         phase_aware=dict(rel_l2=e_p["rel_l2"], db_rms=e_p["db_rms"],
                                          db_p95=e_p["db_p95"], db_max=e_p["db_max"])))
        print("f=%.2f GHz | complex relL2=%.3f dB=%.3f | phase relL2=%.3f dB=%.3f"
              % (fq, e_c["rel_l2"], e_c["db_rms"], e_p["rel_l2"], e_p["db_rms"]))

    mean = dict(
        complex_linear=dict(
            rel_l2=float(np.mean([r["complex_linear"]["rel_l2"] for r in rows])),
            db_rms=float(np.mean([r["complex_linear"]["db_rms"] for r in rows]))),
        phase_aware=dict(
            rel_l2=float(np.mean([r["phase_aware"]["rel_l2"] for r in rows])),
            db_rms=float(np.mean([r["phase_aware"]["db_rms"] for r in rows]))))
    print("mean dB RMS: complex=%.3f phase=%.3f"
          % (mean["complex_linear"]["db_rms"], mean["phase_aware"]["db_rms"]))

    out = dict(eps=EPS_OP, heldout_frequencies_ghz=HELD_OUT,
               valid_pixels=int(m.sum()),
               storage_complex_linear=qtt_model.storage_complex(),
               storage_phase_aware=phase_model.storage_complex(),
               rows=rows, mean=mean)
    with open(os.path.join(DATA, "phase_aware_results.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------- figure
    fs = HELD_OUT
    db_c = [r["complex_linear"]["db_rms"] for r in rows]
    db_p = [r["phase_aware"]["db_rms"] for r in rows]
    rl_c = [r["complex_linear"]["rel_l2"] for r in rows]
    rl_p = [r["phase_aware"]["rel_l2"] for r in rows]
    x = np.arange(len(fs)); w = 0.38

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].bar(x - w/2, db_c, w, label="complex linear")
    ax[0].bar(x + w/2, db_p, w, label="phase-aware")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["%.2f" % f for f in fs])
    ax[0].set_xlabel("held-out frequency [GHz]"); ax[0].set_ylabel("path-loss dB RMS")
    ax[0].set_title("(a) amplitude (path-loss) error"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")

    ax[1].bar(x - w/2, rl_c, w, label="complex linear")
    ax[1].bar(x + w/2, rl_p, w, label="phase-aware")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["%.2f" % f for f in fs])
    ax[1].set_xlabel("held-out frequency [GHz]"); ax[1].set_ylabel("complex rel-$L_2$")
    ax[1].set_title("(b) complex-field (absolute phase) error"); ax[1].legend()
    ax[1].grid(alpha=0.3, axis="y")

    # path-loss cross-range profile at the worst-case frequency
    fshow = 2.5
    ref = direct_pe_field(fshow)
    uc = qtt_model.predict({"frequency_hz": fshow * 1e9})
    up = phase_model.predict({"frequency_hz": fshow * 1e9})
    iy = NY // 2
    zz = z[z <= 220]
    sl = slice(0, zz.size)
    ax[2].plot(zz, 20*np.log10(np.abs(ref[iy, sl])+1e-12), "k-", label="PE ground truth")
    ax[2].plot(zz, 20*np.log10(np.abs(uc[iy, sl])+1e-12), "b:", lw=1.6, label="complex linear")
    ax[2].plot(zz, 20*np.log10(np.abs(up[iy, sl])+1e-12), "r--", label="phase-aware")
    ax[2].set_xlabel("z [m]"); ax[2].set_ylabel("path loss [dB]")
    ax[2].set_title("(c) path-loss profile at 2.5 GHz"); ax[2].legend()
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "phase_aware_comparison.png"), dpi=600)
    plt.savefig(os.path.join(FIGS, "phase_aware_comparison.pdf"))
    plt.close()
    print("saved phase_aware_comparison (png+pdf)")


if __name__ == "__main__":
    main()
