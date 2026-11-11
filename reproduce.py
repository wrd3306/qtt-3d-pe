"""One-click, end-to-end reproduction driver (code only; no manuscript build).

Run from the repository root with the project interpreter:
    python reproduce.py

Stages (each must exit 0):
  1. solver benchmarks (bench1 ... bench9)
  2. QTT dataset generation
  3. QTT decomposition / accounting
  4. phase-aware vs complex-linear held-out comparison
  5. external baselines
  6. six-scenario head-to-head

All generated fields, results and figures are written under data/ and figures/.
A transcript is mirrored to data/reproduce_run.log.
"""
import os, sys, time, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
LOG = open(os.path.join(ROOT, "data", "reproduce_run.log"), "w", encoding="utf-8")


def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def run_stage(name, argv, cwd=None):
    t0 = time.time()
    log("\n=== STAGE: %s ===" % name)
    log("cmd: %s" % " ".join(argv))
    p = subprocess.run(argv, cwd=cwd or ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                       errors="replace")
    log(p.stdout[-6000:])
    dt = time.time() - t0
    ok = p.returncode == 0
    log("--- %s: %s (%.1f s, exit %d) ---" % (name, "PASS" if ok else "FAIL", dt, p.returncode))
    return ok


def main():
    stages = [
        ("solver benchmarks", [PY, "scripts/run_all_benchmarks.py"]),
        ("qtt dataset", [PY, "scripts/qtt_dataset.py"]),
        ("qtt decomposition", [PY, "scripts/qtt_run.py"]),
        ("phase-aware comparison", [PY, "scripts/phase_aware_run.py"]),
        ("baselines", [PY, "scripts/run_baselines.py"]),
        ("scenarios", [PY, "scripts/run_scenarios.py"]),
    ]
    results = {}
    for name, argv in stages:
        results[name] = run_stage(name, argv)
        if not results[name]:
            break

    log("\n==== REPRODUCTION SUMMARY ====")
    for k, v in results.items():
        log("  %-22s %s" % (k, "PASS" if v else "FAIL"))
    overall = all(results.values())
    log("OVERALL: %s" % ("PASS" if overall else "FAIL"))
    LOG.close()
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
