"""Shared helpers for solver benchmarks: paths, plotting style, JSON output."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "solver")
FIGS = os.path.join(ROOT, "figures", "solver")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 100, "savefig.dpi": 600, "font.family": "DejaVu Sans",
})


def savefig(name):
    png = os.path.join(FIGS, name + ".png")
    pdf = os.path.join(FIGS, name + ".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=600)
    plt.savefig(pdf)
    plt.close()
    return png, pdf


def save_json(name, obj):
    path = os.path.join(DATA, name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {"real": o.real, "imag": o.imag}
    return str(o)


def rel_l2(a, b, mask=None):
    a = np.asarray(a); b = np.asarray(b)
    if mask is not None:
        a = a[mask]; b = b[mask]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def interior_mask(y, z, y_half, z_half):
    return (np.abs(y[None, :]) < y_half) & (np.abs(z[:, None]) < z_half) \
        if y.ndim == 1 else None


if __name__ == "__main__":
    print("DATA:", DATA)
    print("FIGS:", FIGS)
