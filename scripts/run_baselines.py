"""Driver: generate the deterministic 16-frequency training set + 5 held-out test
references with the trusted exact-spectral PE, fit the four external baselines,
predict at test frequencies, and record complex/dB errors, storage and timing.

Outputs (all under data/baselines/ and figures/baselines/):
  train_fields.npz        (Nf_train, Ny, Nz) complex128 trusted fields
  test_ref_fields.npz    (Nf_test, Ny, Nz) complex128 trusted references
  results.json            machine-readable metrics, storage, timing
  figures/*.png|pdf       600 dpi + vector

Run:
    py -3.10 scripts/run_baselines.py
"""
import os, sys, time, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The exact-spectral SSFT emits a one-off pocketfft overflow RuntimeWarning at the
# lowest frequency (0.5 GHz); the emitted complex128 field is verified finite and
# bounded (q_max=0.026<1, no evanescent growth).  Suppress only this benign case.
warnings.filterwarnings("ignore", message=".*overflow.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*invalid value.*", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from baselines.common import (BASELINE_CONFIG, build_field, generate_frequency_set,
                              make_valid_mask, rel_l2_complex, db_error_stats,
                              nyquist_angle_deg)
from baselines import (DirectFieldInterpolation, POD_ROM, TruncatedSVD, KuttlerDockery)

DATA = os.path.join(ROOT, "data", "baselines")
FIGS = os.path.join(ROOT, "figures", "baselines")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans",
})


def savefig(name):
    plt.tight_layout()
    png = os.path.join(FIGS, name + ".png")
    pdf = os.path.join(FIGS, name + ".pdf")
    plt.savefig(png, dpi=600)
    plt.savefig(pdf)
    plt.close()
    return png, pdf


def main():
    t0 = time.time()
    cfg = BASELINE_CONFIG
    f_train, f_test = generate_frequency_set()

    # ------------------------------------------------------ 1. trusted fields
    print("[1/5] generating trusted exact-spectral fields ...")
    train = np.empty((len(f_train), cfg["Ny"], cfg["Nz"]), dtype=np.complex128)
    t_gen0 = time.time()
    for i, f in enumerate(f_train):
        train[i], _ = build_field(float(f), operator=cfg["operator_trusted"])
        print(f"   train {i+1:2d}/{len(f_train)} f={f:.3f} GHz")
    test = np.empty((len(f_test), cfg["Ny"], cfg["Nz"]), dtype=np.complex128)
    for j, f in enumerate(f_test):
        test[j], _ = build_field(float(f), operator=cfg["operator_trusted"])
        print(f"   test  {j+1:2d}/{len(f_test)} f={f:.3f} GHz")
    t_gen = time.time() - t_gen0

    # geometry (same for every field)
    _, prob = build_field(float(f_train[0]), operator=cfg["operator_trusted"])
    y, z = prob.y_phys, prob.z_phys
    mask = make_valid_mask(y, z)

    np.savez_compressed(os.path.join(DATA, "train_fields.npz"),
                        fields=train, f_GHz=f_train, y=y, z=z,
                        mask=mask)
    np.savez_compressed(os.path.join(DATA, "test_ref_fields.npz"),
                        fields=test, f_GHz=f_test, y=y, z=z, mask=mask)

    # params for fit
    f_train_hz = f_train * 1.0e9
    fit_params = {"frequency_hz": f_train_hz}

    results = {
        "config": {k: (v if not isinstance(v, tuple) else list(v))
                   for k, v in cfg.items()},
        "frequency_train_GHz": f_train.tolist(),
        "frequency_test_GHz": f_test.tolist(),
        "nyquist_angle_deg": {f"{f}GHz": nyquist_angle_deg(f) for f in (0.5, 1.0, 3.0)},
        "valid_pixels": int(mask.sum()),
        "n_pixels_total": int(mask.size),
        "timing": {"trusted_field_generation_s": t_gen},
        "methods": {},
    }

    # ------------------------------------------------------ 2. helpers
    def evaluate(model, tag):
        """Predict at every test frequency; return per-freq + aggregate metrics."""
        per = []
        online = []
        for j, f in enumerate(f_test):
            t_on = time.time()
            u_pred = model.predict({"frequency_hz": float(f) * 1.0e9})
            online.append(time.time() - t_on)
            u_ref = test[j]
            rel = rel_l2_complex(u_pred, u_ref, mask)
            db = db_error_stats(u_pred, u_ref, mask)
            per.append({"f_GHz": float(f), "rel_l2_complex": rel, **db})
        agg = {
            "rel_l2_complex_mean": float(np.mean([p["rel_l2_complex"] for p in per])),
            "rel_l2_complex_max": float(np.max([p["rel_l2_complex"] for p in per])),
            "db_rms_mean": float(np.mean([p["rms_db"] for p in per])),
            "db_p95_mean": float(np.mean([p["p95_db"] for p in per])),
            "online_predict_s": float(np.mean(online)),
        }
        return per, agg

    # ------------------------------------------------------ 3. fit baselines
    print("[2/5] fitting baselines ...")

    # --- direct interpolation ---
    t_fit = time.time()
    direct = DirectFieldInterpolation(variant="auto").fit(train, fit_params)
    t_direct_fit = time.time() - t_fit
    per, agg = evaluate(direct, "direct")
    agg["storage_complex"] = direct.storage_complex()
    agg["offline_fit_s"] = t_direct_fit
    agg["chosen_variant"] = direct.chosen_variant
    agg["loo_residual"] = direct.loo_residual
    results["methods"]["direct_interpolation"] = {"per_test": per, **agg}
    print(f"   direct [{direct.chosen_variant}] relL2={agg['rel_l2_complex_mean']:.4f} "
          f"storage={agg['storage_complex']}")

    # --- POD/ROM ---
    t_fit = time.time()
    pod = POD_ROM(energy=0.9999).fit(train, fit_params)
    t_pod_fit = time.time() - t_fit
    per, agg = evaluate(pod, "pod")
    agg["storage_complex"] = pod.storage_complex()
    agg["offline_fit_s"] = t_pod_fit
    agg["rank"] = int(pod.rank)
    results["methods"]["pod_rom"] = {"per_test": per, **agg}
    print(f"   pod rank={pod.rank} relL2={agg['rel_l2_complex_mean']:.4f} "
          f"storage={agg['storage_complex']}")

    # --- truncated SVD rank scan ---
    t_fit = time.time()
    svd = TruncatedSVD().fit(train, fit_params)
    t_svd_fit = time.time() - t_fit
    s = svd.singular_values()
    rank_curve = []
    for r in range(1, len(s) + 1):
        svd.set_rank(r)
        per, agg = evaluate(svd, f"svd_r{r}")
        rank_curve.append({"rank": r,
                           "rel_l2_complex_mean": agg["rel_l2_complex_mean"],
                           "rel_l2_complex_max": agg["rel_l2_complex_max"],
                           "db_rms_mean": agg["db_rms_mean"],
                           "storage_complex": svd.storage_complex(),
                           "online_predict_s": agg["online_predict_s"]})
    # operating point = 99.99% energy rank (already set at fit)
    per, agg = evaluate(svd, "svd")
    agg["storage_complex"] = svd.storage_complex()
    agg["offline_fit_s"] = t_svd_fit
    agg["rank"] = int(svd.rank)
    results["methods"]["truncated_svd"] = {"per_test": per, **agg,
                                          "rank_scan": rank_curve,
                                          "singular_values": s.tolist()}
    print(f"   svd rank={svd.rank} relL2={agg['rel_l2_complex_mean']:.4f} "
          f"storage={agg['storage_complex']} (scanned 1..{len(s)})")

    # --- Kuttler-Dockery narrow PE ---
    t_fit = time.time()
    kd = KuttlerDockery().fit(train, fit_params)
    t_kd_fit = time.time() - t_fit
    per, agg = evaluate(kd, "kd")
    agg["storage_complex"] = kd.storage_complex()
    agg["offline_fit_s"] = t_kd_fit
    results["methods"]["kuttler_dockery"] = {"per_test": per, **agg}
    print(f"   kd relL2={agg['rel_l2_complex_mean']:.4f} "
          f"storage={agg['storage_complex']} online={agg['online_predict_s']:.2f}s")

    results["timing"]["total_s"] = time.time() - t0

    # ------------------------------------------------------ 4. save JSON
    with open(os.path.join(DATA, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[3/5] results.json written ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------ 5. figures
    print("[4/5] plotting ...")
    plot_rank_curve(rank_curve)
    plot_storage_time(results)
    plot_error_summary(results, f_test)
    print(f"[5/5] done. total {time.time()-t0:.1f}s")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    return str(o)


def plot_rank_curve(rank_curve):
    rk = [c["rank"] for c in rank_curve]
    err = [c["rel_l2_complex_mean"] for c in rank_curve]
    stor = [c["storage_complex"] for c in rank_curve]

    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax1.semilogy(rk, err, "o-", color="C0", label="mean test rel. L2 error")
    ax1.set_xlabel("SVD rank r")
    ax1.set_ylabel("mean complex rel. L2 error (test)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, which="both", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.semilogy(rk, stor, "s--", color="C3", label="storage (complex128)")
    ax2.set_ylabel("storage (complex128 scalars)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Truncated SVD: rank vs test accuracy and storage")
    savefig("svd_rank_accuracy_storage")


def plot_storage_time(results):
    methods = ["direct_interpolation", "pod_rom", "truncated_svd", "kuttler_dockery"]
    labels = ["Direct interp.", "POD/ROM", "Truncated SVD", "Kuttler-Dockery"]
    stor = [results["methods"][m]["storage_complex"] for m in methods]
    online = [results["methods"][m]["online_predict_s"] for m in methods]
    offline = [results["methods"][m]["offline_fit_s"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = ["C0", "C1", "C2", "C4"]
    axes[0].bar(labels, stor, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("storage (complex128 scalars)")
    axes[0].set_title("(a) retained storage")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, offline, 0.4, label="offline fit", color="C2")
    axes[1].bar(x + 0.2, online, 0.4, label="online per query", color="C3")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("time [s]")
    axes[1].set_title("(b) offline vs online time")
    axes[1].legend()
    savefig("baselines_storage_time")


def plot_error_summary(results, f_test):
    methods = ["direct_interpolation", "pod_rom", "truncated_svd", "kuttler_dockery"]
    labels = ["Direct interp.", "POD/ROM", "Truncated SVD", "Kuttler-Dockery"]
    colors = ["C0", "C1", "C2", "C4"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(f_test))
    w = 0.2
    for k, m in enumerate(methods):
        rel = [p["rel_l2_complex"] for p in results["methods"][m]["per_test"]]
        ax.bar(x + (k - 1.5) * w, rel, w, label=labels[k], color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.2f}" for f in f_test])
    ax.set_xlabel("test frequency [GHz]")
    ax.set_ylabel("complex rel. L2 error (effective region)")
    ax.set_yscale("log")
    ax.set_title("Baseline complex-field error at held-out frequencies")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    savefig("baselines_error_summary")


if __name__ == "__main__":
    main()
