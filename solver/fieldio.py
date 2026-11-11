"""Field + result persistence: complex128 .npz and machine-readable JSON."""
from __future__ import annotations
import json
import os
import numpy as np


def save_field_npz(path, u, y, z, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, u=np.asarray(u, dtype=np.complex128),
                        y=np.asarray(y), z=np.asarray(z),
                        meta_json=json.dumps(meta))
    return path


def load_field_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["u"], d["y"], d["z"], json.loads(str(d["meta_json"]))


def save_results_json(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_default)
    return path


__all__ = ["save_field_npz", "load_field_npz", "save_results_json"]
