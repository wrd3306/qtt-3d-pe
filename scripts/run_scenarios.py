"""Driver: six propagation scenarios, all methods, real numbers.

For each scenario:
  - 16 log-spaced training frequencies (0.5-3 GHz) + 5 held-out test freqs
  - trusted exact-spectral complex128 fields (phase preserved)
  - fit QTTModel, DirectFieldInterpolation, POD_ROM, TruncatedSVD, KuttlerDockery
  - evaluate IN-GRID (training freqs) and OUT-OF-GRID (test freqs)
  - record storage, offline/online time, complex relL2, dB RMS + p50/p90/p95

Plus a bounded mechanism confirmation: free-space Nf in {16,32,64} shows the
held-out error dropping as the PEC interference fringes become Nyquist-resolved.

Outputs:
  data/scenarios/{train,test}_<scenario>.npz
  data/scenarios/results.json
  figures/scenarios/{scenario_field,summary,mechanism}.{png,pdf}
"""
import os, sys, time, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.common import (BASELINE_CONFIG as C, make_valid_mask, rel_l2_complex,
                              db_error_stats)
from baselines import (DirectFieldInterpolation, POD_ROM, TruncatedSVD, KuttlerDockery)
from qtt import QTTModel
from scenarios import run_field, SCENARIOS

DATA = os.path.join(ROOT, "data", "scenarios")
FIGS = os.path.join(ROOT, "figures", "scenarios")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans",
})

# frequency grids (log-spaced per task spec)
F_TRAIN = np.geomspace(0.5, 3.0, 16)          # GHz
F_TEST = np.array([0.75, 1.25, 1.75, 2.25, 2.75])  # GHz, off-grid


def make_methods():
    return {
        "qtt": QTTModel(eps=1e-4),
        "direct": DirectFieldInterpolation(variant="real_imag"),
        "pod": POD_ROM(energy=0.9999),
        "svd": TruncatedSVD(),
        "kd": KuttlerDockery(),
    }


def gen_fields(n_profile, terrain, freqs):
    """Generate trusted exact-spectral fields for a list of frequencies."""
    U = np.empty((len(freqs), C["Ny"], C["Nz"]), dtype=np.complex128)
    prob0 = None
    for i, f in enumerate(freqs):
        u, prob = run_field(float(f), n_profile=n_profile, terrain=terrain,
                            operator=C["operator_trusted"])
        U[i] = u
        prob0 = prob
    return U, prob0


def eval_at(model, freqs_ref, mask, prob):
    """Predict at each freq, compare to reference, return per-freq metrics."""
    per = []
    for j, f in enumerate(freqs_ref):
        u_ref = freqs_ref  # placeholder
    return per


def run_scenario(name, sc):
    print(f"\n=== scenario: {name} ===")
    n_profile = sc["n_profile"]; terrain = sc["terrain"]
    # train / test fields
    Utr, prob = gen_fields(n_profile, terrain, F_TRAIN)
    Ute, _ = gen_fields(n_profile, terrain, F_TEST)
    y, z = prob.y_phys, prob.z_phys
    mask = make_valid_mask(y, z)
    # for terrain scenarios, exclude points below terrain height at x_end
    if name in ("gaussian_terrain", "multi_peak"):
        h = terrain.height(C["x_end_m"], 0.0)
        Z = z[None, :]
        mask = mask & (Z > h + 1e-9)
        print(f"  terrain height at x_end={h:.1f} m, valid pixels now {int(mask.sum())}")

    fit_params = {"frequency_hz": F_TRAIN * 1e9}
    methods = make_methods()

    out = {"desc": sc["desc"], "n_train": len(F_TRAIN), "n_test": len(F_TEST),
           "frequency_train_GHz": F_TRAIN.tolist(),
           "frequency_test_GHz": F_TEST.tolist(),
           "methods": {}}

    for mname, model in methods.items():
        t0 = time.time()
        model.fit(Utr, fit_params)
        t_fit = time.time() - t0
        # in-grid: predict at training freqs
        ingrid = []
        for j, f in enumerate(F_TRAIN):
            u_pred = model.predict({"frequency_hz": float(f) * 1e9})
            rel = rel_l2_complex(u_pred, Utr[j], mask)
            db = db_error_stats(u_pred, Utr[j], mask)
            ingrid.append({"f_GHz": float(f), "rel_l2": rel, **db})
        # out-of-grid: predict at test freqs
        t_on = 0.0
        outgrid = []
        for j, f in enumerate(F_TEST):
            t1 = time.time()
            u_pred = model.predict({"frequency_hz": float(f) * 1e9})
            t_on += time.time() - t1
            rel = rel_l2_complex(u_pred, Ute[j], mask)
            db = db_error_stats(u_pred, Ute[j], mask)
            outgrid.append({"f_GHz": float(f), "rel_l2": rel, **db})
        out["methods"][mname] = {
            "storage_complex": int(model.storage_complex()),
            "offline_fit_s": t_fit,
            "online_per_query_s": t_on / len(F_TEST),
            "in_grid": ingrid,
            "out_grid": outgrid,
            "in_grid_rel_l2_mean": float(np.mean([p["rel_l2"] for p in ingrid])),
            "out_grid_rel_l2_mean": float(np.mean([p["rel_l2"] for p in outgrid])),
            "out_grid_db_rms_mean": float(np.mean([p["rms_db"] for p in outgrid])),
            "out_grid_db_p95_mean": float(np.mean([p["p95_db"] for p in outgrid])),
        }
        print(f"  {mname:6s} storage={model.storage_complex():>8d} "
              f"in-grid relL2={out['methods'][mname]['in_grid_rel_l2_mean']:.4e} "
              f"out-grid relL2={out['methods'][mname]['out_grid_rel_l2_mean']:.4e} "
              f"db_rms={out['methods'][mname]['out_grid_db_rms_mean']:.2f}")

    # save fields
    np.savez_compressed(os.path.join(DATA, f"train_{name}.npz"),
                        fields=Utr, f_GHz=F_TRAIN, y=y, z=z, mask=mask)
    np.savez_compressed(os.path.join(DATA, f"test_{name}.npz"),
                        fields=Ute, f_GHz=F_TEST, y=y, z=z, mask=mask)
    return out, Utr, Ute, y, z, mask, prob


def mechanism_confirmation():
    """Free space with Nf = 16/32/64: held-out error should drop as fringes resolve."""
    print("\n=== mechanism confirmation: free space Nf sweep ===")
    sc = SCENARIOS["free_space"]
    results = {}
    for Nf in (16, 32, 64):
        ftr = np.linspace(0.5, 3.0, Nf)  # linear for clean Nyquist analysis
        Utr, prob = gen_fields(sc["n_profile"], sc["terrain"], ftr)
        y, z = prob.y_phys, prob.z_phys
        mask = make_valid_mask(y, z)
        fit_params = {"frequency_hz": ftr * 1e9}
        # QTT
        qtt = QTTModel(eps=1e-4).fit(Utr, fit_params)
        # direct
        direct = DirectFieldInterpolation(variant="real_imag").fit(Utr, fit_params)
        err_q, err_d = [], []
        for f in F_TEST:
            ue, _ = run_field(float(f), n_profile=sc["n_profile"],
                              terrain=sc["terrain"], operator=C["operator_trusted"])
            uq = qtt.predict({"frequency_hz": float(f) * 1e9})
            ud = direct.predict({"frequency_hz": float(f) * 1e9})
            err_q.append(rel_l2_complex(uq, ue, mask))
            err_d.append(rel_l2_complex(ud, ue, mask))
        results[Nf] = {"qtt_rel_l2_mean": float(np.mean(err_q)),
                       "direct_rel_l2_mean": float(np.mean(err_d)),
                       "spacing_GHz": float(ftr[1] - ftr[0])}
        print(f"  Nf={Nf:2d} spacing={ftr[1]-ftr[0]:.3f} GHz  "
              f"QTT relL2={np.mean(err_q):.4f}  direct relL2={np.mean(err_d):.4f}")
    return results


def make_field_figures(all_out):
    """Per-scenario |u| dB field plots (2x3)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    rep_f = 1.5  # representative frequency
    for ax, (name, sc) in zip(axes.ravel(), SCENARIOS.items()):
        # load train field at rep_f
        d = np.load(os.path.join(DATA, f"train_{name}.npz"))
        U = d["fields"]; f = d["f_GHz"]; z = d["z"]; y = d["y"]
        j = int(np.argmin(np.abs(f - rep_f)))
        Udb = 20*np.log10(np.abs(U[j]) + 1e-12)
        Vmax = Udb.max()
        im = ax.pcolormesh(y, z, Udb.T, shading="auto", cmap="viridis",
                           vmin=Vmax - 40, vmax=Vmax)
        # terrain overlay
        if name in ("gaussian_terrain", "multi_peak"):
            h = sc["terrain"].height(C["x_end_m"], 0.0)
            ax.axhline(h, color="red", ls="--", lw=1.2, label=f"terrain {h:.0f}m")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("cross-range y [m]"); ax.set_ylabel("height z [m]")
        ax.set_ylim(0, 220)
        plt.colorbar(im, ax=ax, label="|u| [dB]")
    plt.suptitle("Trusted field |u| (dB) at 1.5 GHz, each scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "scenarios_fields.png"), dpi=600)
    plt.savefig(os.path.join(FIGS, "scenarios_fields.pdf"))
    plt.close()


def make_summary_fig(all_out):
    """Bar chart: out-grid relL2 per scenario per method."""
    methods = ["qtt", "direct", "pod", "svd", "kd"]
    labels = ["QTT", "Direct", "POD", "SVD", "KD"]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    scen_names = list(SCENARIOS.keys())
    x = np.arange(len(scen_names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, m in enumerate(methods):
        vals = [all_out[s]["methods"][m]["out_grid_rel_l2_mean"] for s in scen_names]
        ax.bar(x + (k - 2) * w, vals, w, label=labels[k], color=colors[k])
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in scen_names], fontsize=8)
    ax.set_ylabel("out-grid complex rel. L2 error")
    ax.set_yscale("log")
    ax.set_title("Held-out prediction accuracy across six scenarios")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "scenarios_summary.png"), dpi=600)
    plt.savefig(os.path.join(FIGS, "scenarios_summary.pdf"))
    plt.close()


def make_mechanism_fig(mech):
    """Nf sweep: held-out error vs training frequency count."""
    Nfs = sorted(mech.keys())
    qtt_err = [mech[n]["qtt_rel_l2_mean"] for n in Nfs]
    direct_err = [mech[n]["direct_rel_l2_mean"] for n in Nfs]
    spacing = [mech[n]["spacing_GHz"] for n in Nfs]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(Nfs, qtt_err, "o-", color="C0", label="QTT")
    ax.semilogy(Nfs, direct_err, "s-", color="C1", label="Direct interp.")
    for n, sp in zip(Nfs, spacing):
        ax.annotate(f"Δf={sp:.2f}GHz", (n, qtt_err[Nfs.index(n)]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("number of training frequencies Nf")
    ax.set_ylabel("held-out complex rel. L2 error (free space)")
    ax.set_title("Mechanism: error drops as fringes become Nyquist-resolved")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "scenarios_mechanism.png"), dpi=600)
    plt.savefig(os.path.join(FIGS, "scenarios_mechanism.pdf"))
    plt.close()


def main():
    t0 = time.time()
    all_out = {}
    for name, sc in SCENARIOS.items():
        out, *_ = run_scenario(name, sc)
        all_out[name] = out
    mech = mechanism_confirmation()
    all_out["_mechanism"] = mech
    all_out["_timing"] = {"total_s": time.time() - t0}
    with open(os.path.join(DATA, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False, default=_json_default)
    make_field_figures(all_out)
    make_summary_fig(all_out)
    make_mechanism_fig(mech)
    print(f"\nDONE in {time.time()-t0:.1f}s")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    return str(o)


if __name__ == "__main__":
    main()
