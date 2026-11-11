"""Main QTT driver: real quantized TT decomposition, honest compression
accounting, break-even cost analysis, and the 4-D multi-parameter extension.

All numbers below come from the trusted solver fields in data/qtt/*.npz and
from the numpy complex-SVD TT code in qtt/tt.py.  No ranks are preset; the
reported ranks are whatever TT-SVD returns at each epsilon.
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qtt.tt import quantize, dequantize, tt_svd, tt_reconstruct, storage_complex
from qtt.model import QTTModel
from qtt.metrics import (valid_mask, field_errors, direct_pe_field,
                         DX, DY, DZ, NY, NZ, Y_MAX, Z_MAX, X_END,
                         SPONGE_TOP, SPONGE_Y, SPONGE_STR,
                         SIGMA_Y, SIGMA_Z, N_F, SOURCE_Z0)
from solver.constants import C_LIGHT

DATA = os.path.join(ROOT, "data", "qtt")
FIGS = os.path.join(ROOT, "figures", "qtt")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans",
})


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, name + ".png"), dpi=600)
    plt.savefig(os.path.join(FIGS, name + ".pdf"))
    plt.close()


# ====================================================================== main
def main():
    t_start = time.time()
    d3 = np.load(os.path.join(DATA, "fields_3d.npz"), allow_pickle=False)
    U = d3["U"]                      # (16,64,128) complex128
    freqs_ghz = d3["freqs_ghz"]
    y, z = d3["y"], d3["z"]
    setup = json.loads(str(d3["setup"]))
    m = valid_mask(y, z)
    dense = int(U.size)

    timing = json.load(open(os.path.join(DATA, "dataset_timing.json"), encoding="utf-8"))
    t_pe_mean = timing["t_pe_3d_mean"]

    # ------------------------------------------------- 1) epsilon sweep
    v, bits = quantize(U, (N_F, NY, NZ))
    eps_grid = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    sweep = []
    eps_cores = {}
    for eps in eps_grid:
        t0 = time.time()
        cores, info = tt_svd(v, eps)
        t_decomp = time.time() - t0
        rec = dequantize(tt_reconstruct(cores), (N_F, NY, NZ))
        err = field_errors(U, rec, m)
        sto = storage_complex(cores)
        sweep.append(dict(
            eps=eps, ranks=info["ranks"], peak_rank=int(max(info["ranks"])),
            storage_complex=sto, storage_real8=2 * sto,
            dense_complex=dense, dense_real8=2 * dense,
            CR=dense / sto, CR_real8=(2*dense)/(2*sto),
            rel_l2=err["rel_l2"], db_rms=err["db_rms"], db_p95=err["db_p95"],
            db_max=err["db_max"], t_decomp_s=t_decomp))
        eps_cores[eps] = cores
        print("eps=%-7g peak_r=%-4d sto=%-7d CR=%6.2fx relL2=%.3e dB_rms=%.3f" %
              (eps, max(info["ranks"]), sto, dense/sto, err["rel_l2"], err["db_rms"]))

    # operating point: eps = 1e-3 (CR ~3x, well below 1 dB RMS)
    EPS_OP = 1e-3
    op = next(s for s in sweep if abs(s["eps"] - EPS_OP) < 1e-12)

    # ------------------------------------------------- 2) model + online cost
    params3d = {"frequency_hz": freqs_ghz * 1e9}
    model = QTTModel(eps=EPS_OP, name="qtt_3d_freq")
    model.fit(U, params3d)
    assert model.storage_complex() == op["storage_complex"]

    # online predict timing (avg over several off-grid queries)
    query_freqs = np.array([0.75, 1.25, 1.75, 2.5, 2.85])  # GHz, off-grid
    N_REPEAT = 30
    t0 = time.time()
    for _ in range(N_REPEAT):
        for fq in query_freqs:
            model.predict({"frequency_hz": fq * 1e9})
    t_online = (time.time() - t0) / (N_REPEAT * len(query_freqs))

    # direct PE timing + held-out accuracy at off-grid frequencies
    heldout = []
    t_direct_list = []
    for fq in query_freqs:
        t0 = time.time()
        u_direct = direct_pe_field(float(fq))
        t_direct_list.append(time.time() - t0)
        u_pred = model.predict({"frequency_hz": fq * 1e9})
        e = field_errors(u_direct[None, :, :], u_pred[None, :, :], m)
        heldout.append(dict(frequency_hz=float(fq*1e9), rel_l2=e["rel_l2"],
                            db_rms=e["db_rms"], db_p95=e["db_p95"], db_max=e["db_max"]))
        print("heldout f=%.2f GHz  relL2=%.3e dB_rms=%.3f" % (fq, e["rel_l2"], e["db_rms"]))
    t_direct = float(np.mean(t_direct_list))

    # ------------------------------------------------- 3) break-even
    t_offline_pe = N_F * t_pe_mean
    t_offline_decomp = op["t_decomp_s"]
    t_offline = t_offline_pe + t_offline_decomp
    if t_direct > t_online:
        Q_be = t_offline / (t_direct - t_online)
    else:
        Q_be = float("inf")
    # storage break-even: number of single-frequency dense fields whose raw
    # storage equals the TT cores (all complex scalars, same counting)
    Q_storage = op["storage_complex"] / (NY * NZ)
    breakeven = dict(
        t_offline_pe_train_s=t_offline_pe,
        t_offline_decomp_s=t_offline_decomp,
        t_offline_total_s=t_offline,
        t_online_predict_s=t_online,
        t_direct_pe_query_s=t_direct,
        Q_breakeven_queries=Q_be,
        Q_storage_equivalent_fields=Q_storage,
        note="TT pipeline cumulative = t_offline + Q*t_online ; "
             "direct PE cumulative = Q*t_direct ; equality solves Q=...")

    # ------------------------------------------------- 4) 4D extension
    d4 = np.load(os.path.join(DATA, "fields_4d.npz"), allow_pickle=False)
    U4 = d4["U"]                       # (16,4,64,128)
    z0_grid = d4["z0_m"]
    dense4 = int(U4.size)
    v4, bits4 = quantize(U4, (N_F, z0_grid.size, NY, NZ))
    sweep4 = []
    for eps in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        cores4, info4 = tt_svd(v4, eps)
        rec4 = dequantize(tt_reconstruct(cores4), (N_F, z0_grid.size, NY, NZ))
        vm4 = np.broadcast_to(m[None, None, :, :], U4.shape)
        rel = float(np.linalg.norm((rec4 - U4)[vm4]) / np.linalg.norm(U4[vm4]))
        amp = np.abs(U4); floor = amp.max()*1e-3
        both = (amp > floor) & (np.abs(rec4) > floor) & vm4
        db = 20*np.log10(np.abs(rec4[both])/np.abs(U4[both]))
        sto4 = storage_complex(cores4)
        sweep4.append(dict(eps=eps, ranks=info4["ranks"], peak_rank=int(max(info4["ranks"])),
                           storage_complex=sto4, dense_complex=dense4,
                           CR=dense4/sto4, rel_l2=rel,
                           db_rms=float(np.sqrt(np.mean(db**2))),
                           db_max=float(np.max(np.abs(db)))))
        print("4D eps=%-7g peak_r=%-4d CR=%6.2fx relL2=%.3e dB_rms=%.3f" %
              (eps, max(info4["ranks"]), dense4/sto4, rel, np.sqrt(np.mean(db**2))))

    # 4D held-out: predict an off-grid frequency at a KNOWN z0, compare direct
    model4 = QTTModel(eps=1e-3, name="qtt_4d_freq_z0")
    model4.fit(U4, {"frequency_hz": freqs_ghz*1e9, "source_height_m": z0_grid})
    heldout4 = []
    for z0test in [25.0, 45.0]:   # off-grid source heights
        for fq in [1.25, 2.2]:
            u_direct = direct_pe_field(float(fq), z0=z0test)
            u_pred = model4.predict({"frequency_hz": fq*1e9, "source_height_m": z0test})
            e = field_errors(u_direct[None, :, :], u_pred[None, :, :], m)
            heldout4.append(dict(frequency_hz=fq*1e9, source_height_m=z0test,
                                 rel_l2=e["rel_l2"], db_rms=e["db_rms"], db_max=e["db_max"]))
            print("4D heldout z0=%.0f f=%.2f relL2=%.3e dB_rms=%.3f" %
                  (z0test, fq, e["rel_l2"], e["db_rms"]))

    # ------------------------------------------------- figures
    # Fig1: eps sweep (CR + error)
    epss = [s["eps"] for s in sweep]
    crs = [s["CR"] for s in sweep]
    rels = [s["rel_l2"] for s in sweep]
    dbs = [s["db_rms"] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].semilogx(epss, crs, "o-", label="TT CR")
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="no compression (CR=1)")
    ax[0].set_xlabel(r"truncation $\varepsilon$"); ax[0].set_ylabel("compression ratio (x)")
    ax[0].set_title("(a) storage CR vs $\\varepsilon$"); ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].loglog(epss, rels, "s-", label="complex rel-L2")
    ax[1].loglog(epss, np.maximum(dbs, 1e-3), "^-", label="RMS dB error")
    ax[1].set_xlabel(r"truncation $\varepsilon$"); ax[1].set_ylabel("error")
    ax[1].set_title("(b) accuracy vs $\\varepsilon$"); ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    savefig("qtt_eps_sweep")

    # Fig2: peak rank vs virtual core position at operating eps
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(range(1, len(op["ranks"])), op["ranks"][1:], "o-", ms=4)
    ax.set_xlabel("virtual core index (left to right: f-bits, y-bits, z-bits)")
    ax.set_ylabel("TT rank $r_k$")
    ax.set_title("True TT ranks (17 virtual dims) at $\\varepsilon$=%g" % EPS_OP)
    ax.grid(alpha=0.3)
    ax.axvline(4, color="gray", ls="--", lw=1); ax.axvline(10, color="gray", ls="--", lw=1)
    ax.text(2, ax.get_ylim()[1]*0.9, "f (4b)", ha="center")
    ax.text(7, ax.get_ylim()[1]*0.9, "y (6b)", ha="center")
    ax.text(13.5, ax.get_ylim()[1]*0.9, "z (7b)", ha="center")
    savefig("qtt_ranks")

    # Fig3: magnitude slice true vs reconstructed (one freq, center y)
    rec_op = dequantize(tt_reconstruct(eps_cores[EPS_OP]), (N_F, NY, NZ))
    iy = NY // 2
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fi = 7
    ax.plot(z, np.abs(U[fi, iy, :]), "k-", label="dense PE")
    ax.plot(z, np.abs(rec_op[fi, iy, :]), "r--", label="TT reconstructed")
    ax.set_xlabel("z [m]"); ax.set_ylabel("|u|"); ax.set_xlim(0, 220)
    ax.set_title("Cross-range slice f=%.2f GHz, y=0" % freqs_ghz[fi]); ax.legend()
    savefig("qtt_slice_compare")

    # Fig4: break-even cumulative cost
    Qq = np.linspace(0, max(5, Q_be*3 if np.isfinite(Q_be) else 50), 200)
    tt_cost = t_offline + Qq * t_online
    pe_cost = Qq * t_direct
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(Qq, tt_cost/60, label="TT surrogate (offline + online)")
    ax.plot(Qq, pe_cost/60, label="direct PE each query")
    if np.isfinite(Q_be):
        ax.axvline(Q_be, color="r", ls=":", label="break-even Q=%.1f" % Q_be)
    ax.set_xlabel("number of query frequencies Q"); ax.set_ylabel("cumulative time [min]")
    ax.set_title("Online break-even"); ax.legend(); ax.grid(alpha=0.3)
    savefig("qtt_breakeven")

    # ------------------------------------------------- save JSON
    out = dict(
        setup=setup,
        grid=dict(Nf=N_F, Ny=NY, Nz=NZ, dy=DY, dz=DZ, dx=DX, x_end=X_END,
                  y_max=Y_MAX, z_max=Z_MAX,
                  nyquist_angle_deg_at_1GHz=float(np.degrees(np.arcsin(
                      (C_LIGHT/1e9)/(2*DZ)))),
                  nyquist_angle_deg_at_3GHz=float(np.degrees(np.arcsin(
                      (C_LIGHT/3e9)/(2*DZ))))),
        sweep_3d=sweep,
        operating=dict(eps=EPS_OP, ranks=op["ranks"], peak_rank=op["peak_rank"],
                       storage_complex=op["storage_complex"],
                       storage_real8=op["storage_real8"],
                       dense_complex=dense, dense_real8=2*dense,
                       CR=op["CR"], rel_l2=op["rel_l2"], db_rms=op["db_rms"]),
        heldout_3d=heldout,
        breakeven=breakeven,
        sweep_4d=sweep4,
        heldout_4d=heldout4,
        total_runtime_s=time.time()-t_start,
    )
    with open(os.path.join(DATA, "qtt_results.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print("saved qtt_results.json; total %.1fs" % (time.time()-t_start))
    print("operating: CR=%.2fx relL2=%.3e dB_rms=%.3f  Q_be=%.2f  Q_storage=%.2f" %
          (op["CR"], op["rel_l2"], op["db_rms"], Q_be, Q_storage))


if __name__ == "__main__":
    main()
