"""One-click driver: run all nine solver benchmarks from a clean environment.

Usage (from project root, Python 3.10):
    py -3.10 scripts/run_all_benchmarks.py

Each bench writes its own .npz / .json / figures into data/solver, figures/solver.
A final summary table prints every benchmark's pass/fail and key numbers.
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "solver")

BENCHES = [
    ("bench1_free_space", "bench1_free_space.py"),
    ("bench2_pec_mirror", "bench2_pec_mirror.py"),
    ("bench3_knife_edge", "bench3_knife_edge.py"),
    ("bench4_grid_convergence", "bench4_grid_convergence.py"),
    ("bench5_absorption", "bench5_absorption.py"),
    ("bench6_wide_angle", "bench6_wide_angle.py"),
    ("bench7_3d_gaussian", "bench7_3d_gaussian.py"),
    ("bench8_terrain", "bench8_terrain.py"),
    ("bench9_duct", "bench9_duct.py"),
]

if __name__ == "__main__":
    t0 = time.time()
    summary = {}
    for name, script in BENCHES:
        print(f"\n=== running {script} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            summary[name] = dict(status="FAILED")
        else:
            try:
                d = json.load(open(os.path.join(DATA, name + ".json"), encoding="utf-8"))
                summary[name] = dict(status="OK", pass_=d.get("pass"))
            except Exception as e:
                summary[name] = dict(status="NO_JSON", error=str(e))
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print(f"total {time.time()-t0:.1f} s")
